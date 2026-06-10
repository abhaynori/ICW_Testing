#!/usr/bin/env python3
"""
FPL-GRPO: Fine-tuning-Persistent Watermark Training.

Implements FPL (Fine-tuning-Persistent Learning) via a TrainerCallback that
fires AFTER each GRPO gradient step (clean CUDA context — avoids the CUDA
stream deadlock that occurs when doing backward()/generate() inside TRL's
reward callback on older kernels).

Algorithm (every --fpl-every steps, in on_step_end):
  1. Evaluate watermark quality on probe prompts  → score_before
  2. Simulate M steps of clean fine-tuning in-place (LoRA weights only)
  3. Evaluate watermark quality again              → score_after_attack
  4. REINFORCE recovery: generate from attacked model, reward watermarked
     completions, update LoRA weights toward watermark behaviour
  5. Evaluate again                               → score_after_recovery
  6. Keep the current state — GRPO continues from the attack-perturbed +
     recovered state, training robustness into the model.

Usage:
  python fpl_grpo_train.py \\
      --warm-start-model <path> \\
      --fpl-steps 5 --fpl-every 25 --fpl-recovery-steps 3 \\
      --samples 1000 --epochs 3 \\
      --output-dir fpl_grpo_logs/run_$(date +%Y%m%d_%H%M%S)
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from datasets import Dataset
from peft import LoraConfig, PeftModel, TaskType, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainerCallback

sys.path.insert(0, str(Path(__file__).parent))
from main import (
    acrostics_detector, secret_sequence,
    get_base_system_prompt, set_acrostics_secret_sequence,
)
from grpo_train import (
    WatermarkRewardFunction,
    prepare_dataset,
    build_messages,
    get_prompt_function,
    build_grpo_config,
    build_grpo_trainer,
    compute_baseline_statistics,
    generate_responses_batch,
)
from research_utils import (
    load_causal_lm_with_adapter_support,
    patch_saved_model_config,
    sanitize_generated_text,
)
from sft_train import (
    SFTDataCollator,
    load_sft_pairs,
    prepare_sft_dataset,
)
from finetune_robustness import load_eval_queries, score_queries, summarize


BASE_MODEL = "Qwen/Qwen2.5-7B-Instruct"


# ── FPL callback ───────────────────────────────────────────────────────────────

class FPLCallback(TrainerCallback):
    """
    Fine-tuning-Persistent Learning callback.

    Runs every fpl_every steps in on_step_end (clean CUDA context, after TRL's
    own gradient update). Simulates a fine-tuning attack on the LoRA weights,
    scores the attacked model on a small fixed probe set, then optionally runs
    a REINFORCE recovery step that pushes the attacked model back toward
    watermarked behaviour.  The post-recovery state is KEPT — GRPO continues
    from the perturbed+recovered weights, so the model learns to embed the
    watermark in a basin robust to fine-tuning.
    """

    def __init__(
        self,
        model,
        tokenizer,
        attack_data_loader: DataLoader,
        probe_messages: list,          # fixed small prompt set, loaded once
        fpl_every: int = 25,
        fpl_attack_steps: int = 5,
        fpl_attack_lr: float = 2e-5,
        fpl_recovery_steps: int = 3,
        fpl_recovery_lr: float = 5e-5,
        fpl_max_new_tokens: int = 200,
        output_dir: str | None = None,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.attack_data_loader = attack_data_loader
        self.probe_messages = probe_messages
        self.fpl_every = fpl_every
        self.fpl_attack_steps = fpl_attack_steps
        self.fpl_attack_lr = fpl_attack_lr
        self.fpl_recovery_steps = fpl_recovery_steps
        self.fpl_recovery_lr = fpl_recovery_lr
        self.fpl_max_new_tokens = fpl_max_new_tokens
        self.output_dir = output_dir
        self._attack_iter = iter(attack_data_loader)
        self._log: list[dict] = []

    # ── helpers ────────────────────────────────────────────────────────────────

    def _next_attack_batch(self):
        try:
            return next(self._attack_iter)
        except StopIteration:
            self._attack_iter = iter(self.attack_data_loader)
            return next(self._attack_iter)

    def _score_probes(self) -> tuple[list[str], float]:
        """Generate probe completions and return (texts, mean_acrostics_score)."""
        was_training = self.model.training
        self.model.eval()
        with torch.no_grad():
            texts = generate_responses_batch(
                self.model, self.tokenizer, self.probe_messages,
                max_new_tokens=self.fpl_max_new_tokens,
                temperature=0.7, top_p=0.9,
            )
        if was_training:
            self.model.train()
        scores = [
            float(acrostics_detector(sanitize_generated_text(t), secret_sequence))
            for t in texts
        ]
        return texts, float(np.mean(scores))

    def _simulate_attack(self):
        """Run fpl_attack_steps of clean NLL fine-tuning in-place on LoRA weights."""
        device = next(self.model.parameters()).device
        trainable = [p for p in self.model.parameters() if p.requires_grad]
        opt = torch.optim.AdamW(trainable, lr=self.fpl_attack_lr)
        self.model.train()
        for _ in range(self.fpl_attack_steps):
            batch = self._next_attack_batch()
            batch = {k: v.to(device) for k, v in batch.items() if isinstance(v, torch.Tensor)}
            with torch.enable_grad():
                outputs = self.model(**batch)
                outputs.loss.backward()
                torch.nn.utils.clip_grad_norm_(trainable, 1.0)
                opt.step()
                opt.zero_grad()

    def _recovery_step(self):
        """
        REINFORCE recovery: generate completions from the attacked model, reward
        those that are still watermarked, and do policy-gradient updates.
        """
        if self.fpl_recovery_steps == 0:
            return
        device = next(self.model.parameters()).device
        trainable = [p for p in self.model.parameters() if p.requires_grad]
        opt = torch.optim.AdamW(trainable, lr=self.fpl_recovery_lr)

        for _ in range(self.fpl_recovery_steps):
            opt.zero_grad()

            # Generate completions (eval mode, no grad)
            self.model.eval()
            with torch.no_grad():
                completions = generate_responses_batch(
                    self.model, self.tokenizer, self.probe_messages,
                    max_new_tokens=self.fpl_max_new_tokens,
                    temperature=0.7, top_p=0.9,
                )
            self.model.train()

            # REINFORCE: reward = -acrostics_score (lower = better watermark)
            losses = []
            for messages, completion in zip(self.probe_messages, completions):
                reward = -float(
                    acrostics_detector(sanitize_generated_text(completion), secret_sequence)
                )
                reward_t = torch.tensor(reward, dtype=torch.float32, device=device)

                prompt_text = self.tokenizer.apply_chat_template(
                    messages, add_generation_prompt=True, tokenize=False
                )
                full_text = prompt_text + completion
                prompt_len = self.tokenizer(
                    prompt_text, return_tensors="pt"
                ).input_ids.shape[1]
                full_ids = self.tokenizer(
                    full_text, return_tensors="pt",
                    truncation=True, max_length=1024,
                ).input_ids.to(device)

                if full_ids.shape[1] <= prompt_len:
                    continue

                out = self.model(input_ids=full_ids)
                # logits aligned: position i predicts token i+1
                logits = out.logits[0, prompt_len - 1 : -1]   # [completion_len, vocab]
                target = full_ids[0, prompt_len:]              # [completion_len]
                log_probs = F.log_softmax(logits, dim=-1)
                token_log_probs = log_probs[torch.arange(len(target), device=device), target]
                seq_log_prob = token_log_probs.mean()
                losses.append(-reward_t * seq_log_prob)

            if losses:
                total_loss = torch.stack(losses).mean()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(trainable, 1.0)
                opt.step()

    # ── callback entry point ───────────────────────────────────────────────────

    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step == 0 or state.global_step % self.fpl_every != 0:
            return

        print(f"\n[FPL @ step {state.global_step}] Starting attack simulation...")

        _, score_before = self._score_probes()
        self._simulate_attack()
        _, score_after_attack = self._score_probes()

        if self.fpl_recovery_steps > 0:
            self._recovery_step()
            _, score_after_recovery = self._score_probes()
        else:
            score_after_recovery = score_after_attack

        entry = {
            "step": state.global_step,
            "score_before": score_before,
            "score_after_attack": score_after_attack,
            "score_after_recovery": score_after_recovery,
        }
        self._log.append(entry)

        degradation = score_after_attack - score_before     # positive = watermark lost
        recovery = score_after_attack - score_after_recovery  # positive = recovered

        print(
            f"[FPL @ step {state.global_step}] "
            f"before={score_before:.3f}  attacked={score_after_attack:.3f} "
            f"(+{degradation:.3f})  recovered={score_after_recovery:.3f} "
            f"(-{recovery:.3f})"
        )

        if self.output_dir:
            log_path = os.path.join(self.output_dir, "fpl_log.json")
            with open(log_path, "w") as f:
                json.dump(self._log, f, indent=2)


# ── model loading ──────────────────────────────────────────────────────────────

def load_and_merge(model_path: str, tokenizer, dtype) -> torch.nn.Module:
    adapter_cfg = os.path.join(model_path, "adapter_config.json")
    if os.path.exists(adapter_cfg):
        with open(adapter_cfg) as f:
            base_path = json.load(f)["base_model_name_or_path"]
        base = AutoModelForCausalLM.from_pretrained(
            base_path, device_map="auto", trust_remote_code=True,
            low_cpu_mem_usage=True, torch_dtype=dtype,
        )
        model = PeftModel.from_pretrained(base, model_path)
        model = model.merge_and_unload()
        print(f"  ✓ Merged PEFT adapter from {base_path}")
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_path, device_map="auto", trust_remote_code=True,
            low_cpu_mem_usage=True, torch_dtype=dtype,
        )
        print(f"  ✓ Loaded full model from {model_path}")
    return model


# ── eval ───────────────────────────────────────────────────────────────────────

def run_eval(model, tokenizer, queries_by_ds: dict, gen_kwargs: dict, label: str):
    print(f"\n{'─'*60}")
    print(f"  EVAL: {label}")
    print(f"{'─'*60}")
    for ds, queries in queries_by_ds.items():
        scores = score_queries(model, tokenizer, queries, **gen_kwargs)
        s = summarize(scores)
        print(
            f"  [{ds}] mean={s['mean']:.4f}  std={s['std']:.4f}  "
            f"z={s['z']:.3f}  p={s['p']:.4e}"
        )


# ── main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="FPL-GRPO: fine-tuning-robust watermark training")

    # Model
    parser.add_argument("--warm-start-model", required=True,
                        help="Starting GRPO/SFT model checkpoint to fine-tune with FPL")
    parser.add_argument("--base-model", default=BASE_MODEL)

    # GRPO training
    parser.add_argument("--samples",         type=int,   default=1000)
    parser.add_argument("--epochs",          type=int,   default=3)
    parser.add_argument("--batch-size",      type=int,   default=4)
    parser.add_argument("--learning-rate",   type=float, default=1e-5)
    parser.add_argument("--num-generations", type=int,   default=4)
    parser.add_argument("--max-new-tokens",  type=int,   default=200)
    parser.add_argument("--temperature",     type=float, default=0.7)
    parser.add_argument("--top-p",           type=float, default=0.9)
    parser.add_argument("--beta",            type=float, default=0.04)
    parser.add_argument("--train-dataset",   default="mixed",
                        choices=["eli5", "alpaca", "mixed", "gsm8k"])
    parser.add_argument("--implicit-fraction", type=float, default=0.4)

    # LoRA for GRPO (required for FPL — attack/recovery operate on LoRA weights)
    parser.add_argument("--lora-rank",  type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)

    # FPL hyperparameters
    parser.add_argument("--fpl-every",           type=int,   default=25,
                        help="Run FPL every N GRPO steps (default: 25)")
    parser.add_argument("--fpl-steps",           type=int,   default=5,
                        help="Simulated attack gradient steps per FPL call (default: 5)")
    parser.add_argument("--fpl-lr",              type=float, default=2e-5,
                        help="Attack learning rate (default: 2e-5)")
    parser.add_argument("--fpl-recovery-steps",  type=int,   default=3,
                        help="REINFORCE recovery steps after attack (0 = skip, default: 3)")
    parser.add_argument("--fpl-recovery-lr",     type=float, default=5e-5,
                        help="Recovery learning rate (default: 5e-5)")
    parser.add_argument("--fpl-attack-dataset",  default="alpaca",
                        choices=["eli5", "alpaca", "gsm8k"])
    parser.add_argument("--fpl-attack-samples",  type=int,   default=200)

    # Eval
    parser.add_argument("--eval-samples",    type=int, default=100)
    parser.add_argument("--gen-batch",       type=int, default=4)
    parser.add_argument("--eval-max-tokens", type=int, default=512)
    parser.add_argument("--eval-min-tokens", type=int, default=256)

    # Output
    parser.add_argument("--output-dir",
                        default=f"fpl_grpo_logs/run_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    use_cuda = torch.cuda.is_available()
    use_bf16 = use_cuda and torch.cuda.is_bf16_supported()
    dtype = torch.bfloat16 if use_bf16 else torch.float16

    print("\n" + "=" * 70)
    print("FPL-GRPO: Fine-tuning-Persistent Watermark Training")
    print("=" * 70)
    print(f"Warm start:        {args.warm_start_model}")
    print(f"GRPO dataset:      {args.train_dataset} ({args.samples} samples × {args.epochs} epochs)")
    print(f"GRPO LoRA rank:    {args.lora_rank}")
    print(f"FPL every:         {args.fpl_every} steps")
    print(f"FPL attack steps:  {args.fpl_steps} (lr={args.fpl_lr})")
    print(f"FPL recovery steps:{args.fpl_recovery_steps} (lr={args.fpl_recovery_lr})")
    print(f"FPL attack data:   {args.fpl_attack_dataset} ({args.fpl_attack_samples} samples)")
    print(f"Output:            {args.output_dir}")
    print("=" * 70 + "\n")

    # ── tokenizer ─────────────────────────────────────────────────────────────
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.warm_start_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    print("✓ Tokenizer loaded\n")

    # ── model: merge warm-start adapter, then attach GRPO LoRA ────────────────
    print("Loading model...")
    model = load_and_merge(args.warm_start_model, tokenizer, dtype)
    model.config.pad_token_id = tokenizer.pad_token_id

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "up_proj", "down_proj", "gate_proj"],
        bias="none",
    )
    model = get_peft_model(model, lora_config)
    model.enable_input_require_grads()
    model.print_trainable_parameters()
    print("✓ GRPO LoRA attached\n")

    # ── eval queries ──────────────────────────────────────────────────────────
    print(f"Loading eval queries ({args.eval_samples} per dataset)...")
    queries_by_ds = {
        ds: load_eval_queries(ds, args.eval_samples)
        for ds in ["eli5", "alpaca", "gsm8k"]
    }
    eval_gen_kwargs = dict(
        gen_batch=args.gen_batch,
        max_new_tokens=args.eval_max_tokens,
        min_new_tokens=args.eval_min_tokens,
        temperature=0.7,
        top_p=0.9,
    )
    print()

    # ── baseline eval (before FPL-GRPO) ──────────────────────────────────────
    run_eval(model, tokenizer, queries_by_ds, eval_gen_kwargs, "Before FPL-GRPO")

    # ── FPL attack data loader ────────────────────────────────────────────────
    print(f"\nLoading FPL attack data ({args.fpl_attack_dataset}, {args.fpl_attack_samples} samples)...")
    attack_records = load_sft_pairs(
        dataset_name=args.fpl_attack_dataset,
        split="train",
        num_samples=args.fpl_attack_samples,
    )
    attack_dataset = prepare_sft_dataset(
        records=attack_records, tokenizer=tokenizer,
        prompt_fn=None, include_instruction=False, max_length=1024,
    )
    attack_loader = DataLoader(
        attack_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=SFTDataCollator(tokenizer),
    )
    print(f"✓ {len(attack_dataset)} attack samples loaded\n")

    # ── FPL probe messages: small fixed set used every fpl_every steps ────────
    # Always exactly gen_batch prompts — O(1) cost per FPL call, not O(dataset).
    print(f"Building FPL probe messages ({args.gen_batch} fixed prompts)...")
    _probe_queries = load_eval_queries(args.fpl_attack_dataset, args.gen_batch)
    fpl_probe_messages = [
        build_messages(q, include_instruction=False) for q in _probe_queries
    ]
    print(f"✓ {len(fpl_probe_messages)} probe messages ready\n")

    # ── GRPO training dataset ─────────────────────────────────────────────────
    print(f"Loading GRPO training dataset ({args.train_dataset}, {args.samples} samples)...")
    train_dataset_raw = prepare_dataset(
        num_samples=args.samples,
        split="train",
        dataset_name=args.train_dataset,
        seed=args.seed,
    )

    prompt_fn = get_prompt_function("acrostics")
    _implicit_rng = np.random.default_rng(args.seed + 1000)

    def tokenize_function(examples):
        prompts = []
        for query in examples["query"]:
            if _implicit_rng.random() < args.implicit_fraction:
                messages = build_messages(query, include_instruction=False)
            else:
                messages = prompt_fn(query)
            prompt_text = tokenizer.apply_chat_template(
                messages, add_generation_prompt=True, tokenize=False
            )
            prompts.append(prompt_text)
        return {"prompt": prompts}

    tokenized_dataset = train_dataset_raw.map(tokenize_function, batched=True)
    print(f"✓ {len(tokenized_dataset)} prompts tokenized\n")

    # ── baseline reward statistics ────────────────────────────────────────────
    print("Computing baseline reward statistics...")
    baseline_reward = WatermarkRewardFunction("acrostics")
    baseline_mean, baseline_std = compute_baseline_statistics(
        model, tokenizer, train_dataset_raw, "acrostics",
        num_samples=min(50, args.samples),
        generation_batch_size=args.gen_batch,
        reward_override_fn=baseline_reward._acrostics_training_score,
        max_new_tokens=args.max_new_tokens,
    )
    print(f"  Baseline mean={baseline_mean:.3f}  std={baseline_std:.3f}\n")

    # ── standard watermark reward (no FPL blending needed) ───────────────────
    standard_reward = WatermarkRewardFunction(
        "acrostics", baseline_mean, baseline_std,
        reward_shaping=False,
    )

    # ── FPL callback (attack + recovery outside reward computation) ───────────
    fpl_callback = FPLCallback(
        model=model,
        tokenizer=tokenizer,
        attack_data_loader=attack_loader,
        probe_messages=fpl_probe_messages,
        fpl_every=args.fpl_every,
        fpl_attack_steps=args.fpl_steps,
        fpl_attack_lr=args.fpl_lr,
        fpl_recovery_steps=args.fpl_recovery_steps,
        fpl_recovery_lr=args.fpl_recovery_lr,
        fpl_max_new_tokens=args.max_new_tokens,
        output_dir=args.output_dir,
    )

    # ── GRPO trainer setup ────────────────────────────────────────────────────
    base_training_args = {
        "output_dir": os.path.join(args.output_dir, "grpo_checkpoints"),
        "num_train_epochs": args.epochs,
        "per_device_train_batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "logging_steps": 10,
        "save_steps": 100,
        "save_total_limit": 2,
        "gradient_accumulation_steps": 4,
        "warmup_steps": 10,
        "max_grad_norm": 1.0,
        "seed": args.seed,
        "beta": args.beta,
        "bf16": use_bf16,
        "fp16": bool(use_cuda and not use_bf16),
    }
    generation_args = {
        "max_new_tokens": args.max_new_tokens,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "remove_invalid_values": True,
    }
    training_args = build_grpo_config(base_training_args, generation_args, args.num_generations)

    print("Initializing GRPO trainer...")
    trainer = build_grpo_trainer(
        model=model,
        training_args=training_args,
        train_dataset=tokenized_dataset,
        tokenizer=tokenizer,
        reward_fn=standard_reward,
        reference_model=None,
        require_explicit_reference=False,
    )
    trainer.add_callback(fpl_callback)
    print("✓ Trainer initialized with FPLCallback\n")

    # ── train ─────────────────────────────────────────────────────────────────
    print("=" * 70)
    print("Starting FPL-GRPO training...")
    print("=" * 70 + "\n")
    trainer.train()
    print("\n✓ Training complete\n")

    # ── save final model ──────────────────────────────────────────────────────
    # Merge the FPL LoRA into the weights before saving.  The effective base is
    # (warm-start merged) + (FPL LoRA delta), but adapter_config.json would only
    # record Qwen as the base — making downstream loaders reconstruct the wrong
    # model.  Saving as a plain merged model avoids this entirely.
    final_model_path = os.path.join(args.output_dir, "final_model")
    print(f"Saving final model (merged) → {final_model_path}")
    merged = trainer.model.merge_and_unload()
    merged.save_pretrained(final_model_path)
    tokenizer.save_pretrained(final_model_path)
    patch_saved_model_config(final_model_path, args.base_model)
    print("✓ Saved\n")

    # ── post-training eval ────────────────────────────────────────────────────
    run_eval(model, tokenizer, queries_by_ds, eval_gen_kwargs, "After FPL-GRPO")

    # ── save FPL log ──────────────────────────────────────────────────────────
    log_path = os.path.join(args.output_dir, "fpl_log.json")
    with open(log_path, "w") as f:
        json.dump(fpl_callback._log, f, indent=2)
    print(f"FPL log → {log_path}")

    print("\n" + "=" * 70)
    print("Next step — test fine-tuning robustness of the FPL model:")
    print(f"  python finetune_robustness.py \\")
    print(f"      --grpo-model {final_model_path} \\")
    print(f"      --base-model {args.base_model} \\")
    print(f"      --finetune-dataset alpaca \\")
    print(f"      --finetune-samples 200 \\")
    print(f"      --max-steps 200 \\")
    print(f"      --lora-rank 4 \\")
    print(f"      --skip-base \\")
    print(f"      --output-dir fpl_grpo_logs/robustness_test")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()

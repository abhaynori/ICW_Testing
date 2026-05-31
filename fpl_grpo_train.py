#!/usr/bin/env python3
"""
FPL-GRPO: Fine-tuning-robust Policy Learning with GRPO.

FPL (Fine-tuning-Persistent Learning) makes the watermark robust by training
with simulated fine-tuning attacks inside each reward step:

  For each GRPO training step:
    1. Generate K completions from θ (standard GRPO rollout)
    2. Compute standard watermark rewards r_standard
    3. Save θ's trainable weights (~20MB for LoRA, or full model)
    4. Simulate M steps of clean fine-tuning  →  θ_attacked
    5. Generate completions from θ_attacked on the SAME prompts
    6. Compute r_fpl = watermark quality of θ_attacked completions
    7. Restore θ's original weights
    8. Combined reward = (1-λ) * r_standard + λ * r_fpl
    9. GRPO update on θ using combined reward

Only one model is loaded — FPL works in-place, so no OOM from a second model.

Usage:
  python fpl_grpo_train.py \
      --warm-start-model rerun_acrostics_multidomain_42_20260525_112640/grpo_models/acrostics_full_20260525_230104/final_model \
      --fpl-steps 5 \
      --fpl-lambda 0.5 \
      --fpl-attack-samples 200 \
      --samples 1000 \
      --epochs 3 \
      --output-dir fpl_grpo_logs/run_$(date +%Y%m%d_%H%M%S)
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from datasets import Dataset
from peft import LoraConfig, PeftModel, TaskType, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

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


# ── FPL reward wrapper ─────────────────────────────────────────────────────────

class FPLRewardFunction:
    """
    Wraps WatermarkRewardFunction with FPL: on every reward call, simulates
    M steps of clean fine-tuning (in-place, then restored), generates completions
    from the attacked model on a small FIXED set of representative prompts, and
    uses the attacked model's watermark quality as a robustness bonus.

    The FPL signal is a scalar per batch (average over the fixed probe prompts),
    broadcast uniformly to all completions in the current GRPO batch so it shapes
    the GRPO advantage without creating a per-prompt mismatch.
    """

    def __init__(
        self,
        base_reward: WatermarkRewardFunction,
        model,
        tokenizer,
        clean_data_loader: DataLoader,
        fpl_probe_messages: list,          # fixed small prompt set loaded once at init
        fpl_steps: int = 5,
        fpl_lr: float = 2e-5,
        fpl_lambda: float = 0.5,
        fpl_max_new_tokens: int = 200,
    ):
        self.base = base_reward
        self.model = model
        self.tokenizer = tokenizer
        self.clean_data_loader = clean_data_loader
        self.fpl_probe_messages = fpl_probe_messages   # list of chat-message lists
        self.fpl_steps = fpl_steps
        self.fpl_lr = fpl_lr
        self.fpl_lambda = fpl_lambda
        self.fpl_max_new_tokens = fpl_max_new_tokens
        self._clean_iter = iter(clean_data_loader)
        self.__name__ = f"fpl_watermark_reward_{base_reward.method}"
        self._step = 0
        self._fpl_scores_log: list[float] = []

    def _next_clean_batch(self):
        try:
            return next(self._clean_iter)
        except StopIteration:
            self._clean_iter = iter(self.clean_data_loader)
            return next(self._clean_iter)

    def _save_params(self):
        return {
            k: v.detach().clone()
            for k, v in self.model.named_parameters()
            if v.requires_grad
        }

    def _restore_params(self, saved: dict):
        for k, v in self.model.named_parameters():
            if k in saved:
                v.data.copy_(saved[k])

    def _simulate_attack(self):
        """Apply fpl_steps of clean fine-tuning in-place on the model."""
        device = next(self.model.parameters()).device
        opt = torch.optim.AdamW(
            [p for p in self.model.parameters() if p.requires_grad],
            lr=self.fpl_lr,
        )
        self.model.train()
        for _ in range(self.fpl_steps):
            batch = self._next_clean_batch()
            batch = {
                k: v.to(device)
                for k, v in batch.items()
                if isinstance(v, torch.Tensor)
            }
            outputs = self.model(**batch)
            outputs.loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [p for p in self.model.parameters() if p.requires_grad], 1.0
            )
            opt.step()
            opt.zero_grad()
        self.model.eval()

    def __call__(self, *args, **kwargs):
        self._step += 1

        # Standard watermark reward (on TRL's current-batch completions)
        r_standard = self.base(*args, **kwargs)
        n = len(r_standard)

        if self.fpl_lambda == 0:
            return r_standard

        # ── FPL simulation (fixed cost: always fpl_probe_messages prompts) ────
        saved = self._save_params()
        try:
            self._simulate_attack()

            # Generate from attacked model on the small fixed probe set only.
            # This is O(len(fpl_probe_messages)) = O(fpl_gen_batch), never O(dataset).
            with torch.no_grad():
                attacked_texts = generate_responses_batch(
                    self.model,
                    self.tokenizer,
                    self.fpl_probe_messages,
                    max_new_tokens=self.fpl_max_new_tokens,
                    temperature=0.7,
                    top_p=0.9,
                )

            # Scalar FPL score: average watermark quality across probe prompts
            probe_scores = [
                float(acrostics_detector(sanitize_generated_text(t), secret_sequence))
                for t in attacked_texts
            ]
            avg_probe = float(np.mean(probe_scores))

            # Normalize with same baseline as standard reward
            base = self.base
            if base.baseline_std > 0:
                fpl_scalar = (avg_probe - base.baseline_mean) / base.baseline_std
            else:
                fpl_scalar = avg_probe - base.baseline_mean
            if base.max_abs_reward and base.max_abs_reward > 0:
                fpl_scalar = float(np.clip(fpl_scalar, -base.max_abs_reward, base.max_abs_reward))

            # Broadcast scalar uniformly to all completions in this batch
            fpl_tensor = torch.full((n,), fpl_scalar, dtype=torch.float32)

            self._fpl_scores_log.append(fpl_scalar)
            if self._step % 10 == 0:
                print(
                    f"  [FPL step {self._step}] "
                    f"r_std={float(r_standard.mean()):.3f}  "
                    f"r_fpl={fpl_scalar:.3f}  "
                    f"probe_raw={avg_probe:.3f}"
                )

        finally:
            self._restore_params(saved)
            self.model.train()

        return (1.0 - self.fpl_lambda) * r_standard + self.fpl_lambda * fpl_tensor


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
    parser.add_argument("--samples",         type=int,   default=1000,
                        help="GRPO training samples (default: 1000)")
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

    # LoRA for GRPO (required for FPL — saves/restores only LoRA weights)
    parser.add_argument("--lora-rank",       type=int,   default=16)
    parser.add_argument("--lora-alpha",      type=int,   default=32)

    # FPL hyperparameters
    parser.add_argument("--fpl-steps",          type=int,   default=5,
                        help="Inner fine-tuning steps per GRPO step (default: 5)")
    parser.add_argument("--fpl-lr",             type=float, default=2e-5,
                        help="Learning rate for FPL inner attack (default: 2e-5)")
    parser.add_argument("--fpl-lambda",         type=float, default=0.5,
                        help="Weight on FPL reward: 0=standard only, 1=FPL only (default: 0.5)")
    parser.add_argument("--fpl-attack-dataset", default="alpaca",
                        choices=["eli5", "alpaca", "gsm8k"],
                        help="Clean dataset for simulated attack (default: alpaca)")
    parser.add_argument("--fpl-attack-samples", type=int,   default=200,
                        help="Clean samples for FPL attack data loader (default: 200)")
    parser.add_argument("--fpl-attack-rank",    type=int,   default=4,
                        help="LoRA rank for the FPL inner attack (default: 4)")

    # Eval
    parser.add_argument("--eval-samples",    type=int,   default=100)
    parser.add_argument("--gen-batch",       type=int,   default=4)
    parser.add_argument("--eval-max-tokens", type=int,   default=512)
    parser.add_argument("--eval-min-tokens", type=int,   default=256)

    # Output
    parser.add_argument("--output-dir", default=f"fpl_grpo_logs/run_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    parser.add_argument("--seed",       type=int, default=42)

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
    print(f"FPL lambda:        {args.fpl_lambda}")
    print(f"FPL attack steps:  {args.fpl_steps} per reward call (lr={args.fpl_lr})")
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

    # ── FPL probe messages: small fixed set used every reward call ────────────
    # These are the prompts the attacked model is evaluated on.  Always exactly
    # fpl_gen_batch prompts — never the full training dataset.
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
    from grpo_train import WatermarkRewardFunction
    baseline_reward = WatermarkRewardFunction("acrostics")
    baseline_mean, baseline_std = compute_baseline_statistics(
        model, tokenizer, train_dataset_raw, "acrostics",
        num_samples=min(50, args.samples),
        generation_batch_size=args.gen_batch,
        reward_override_fn=baseline_reward._acrostics_training_score,
        max_new_tokens=args.max_new_tokens,
    )
    print(f"  Baseline mean={baseline_mean:.3f}  std={baseline_std:.3f}\n")

    # ── build reward functions ────────────────────────────────────────────────
    standard_reward = WatermarkRewardFunction(
        "acrostics", baseline_mean, baseline_std,
        reward_shaping=False,
    )

    fpl_reward = FPLRewardFunction(
        base_reward=standard_reward,
        model=model,
        tokenizer=tokenizer,
        clean_data_loader=attack_loader,
        fpl_probe_messages=fpl_probe_messages,
        fpl_steps=args.fpl_steps,
        fpl_lr=args.fpl_lr,
        fpl_lambda=args.fpl_lambda,
        fpl_max_new_tokens=args.max_new_tokens,
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

    print("Initializing GRPO trainer with FPL reward...")
    trainer = build_grpo_trainer(
        model=model,
        training_args=training_args,
        train_dataset=tokenized_dataset,
        tokenizer=tokenizer,
        reward_fn=fpl_reward,
        reference_model=None,
        require_explicit_reference=False,
    )
    print("✓ Trainer initialized\n")

    # ── train ─────────────────────────────────────────────────────────────────
    print("=" * 70)
    print("Starting FPL-GRPO training...")
    print("=" * 70 + "\n")
    trainer.train()
    print("\n✓ Training complete\n")

    # ── save final model (merge LoRA into base) ───────────────────────────────
    final_model_path = os.path.join(args.output_dir, "final_model")
    print(f"Saving final model → {final_model_path}")
    trainer.save_model(final_model_path)
    tokenizer.save_pretrained(final_model_path)
    patch_saved_model_config(final_model_path, args.base_model)
    print("✓ Saved\n")

    # ── post-training eval ────────────────────────────────────────────────────
    run_eval(model, tokenizer, queries_by_ds, eval_gen_kwargs, "After FPL-GRPO (LoRA attached)")

    # ── save FPL scores log ───────────────────────────────────────────────────
    import json as _json
    log_path = os.path.join(args.output_dir, "fpl_scores.json")
    with open(log_path, "w") as f:
        _json.dump({"fpl_scores": fpl_reward._fpl_scores_log}, f)
    print(f"FPL scores log → {log_path}")

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

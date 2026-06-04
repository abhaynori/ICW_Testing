#!/usr/bin/env python3
"""
FRPO-GRPO: Fine-tuning-Robust Policy Optimization for Watermark Training.

Implements FRPO (Sabbaghi et al. 2026, arXiv:2602.08813) as a reward-level
approximation on top of GRPO.  Instead of optimising E[r], FRPO optimises:

    -lambda * log E[exp(-r/lambda)]   (entropic risk / CVaR objective)

which is equivalent to E[r] - (1/2*lambda)*Var(r) for large lambda —
explicitly penalising reward variance so the policy sits in a flat region
of reward space that downstream fine-tuning cannot easily escape.

Implementation:
  Within TRL's GRPOTrainer the reward function is called once per step with
  all B*G completions (B prompts, G generations each).  We wrap the base
  watermark reward to apply per-group softmax reweighting before TRL centres
  within groups:

      w_i  = softmax(-A_i / lambda)   (upweights bad completions)
      r_effective_i = lambda * G * w_i

  TRL then centres these to get advantages.  This gives the correct FRPO
  gradient direction (equivalent to reweighted REINFORCE) with zero extra
  model forward/backward passes.

Usage:
  python frpo_grpo_train.py \\
      --warm-start-model <path> \\
      --lambda-frpo 0.5 \\
      --samples 1000 --epochs 3 \\
      --output-dir frpo_grpo_logs/run_$(date +%Y%m%d_%H%M%S)

  Set --lambda-frpo to a large value (e.g. 1e6) to recover standard GRPO.
  Rule of thumb from paper: lambda ~ std(reward) / sqrt(2 * 0.3) ≈ std/0.77.
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
from finetune_robustness import load_eval_queries, score_queries, summarize


BASE_MODEL = "Qwen/Qwen2.5-7B-Instruct"


# ── FRPO reward wrapper ────────────────────────────────────────────────────────

class FRPORewardWrapper:
    """
    Wraps a per-completion reward function with FRPO group-level reweighting.

    After computing raw rewards for all B*G completions in the current step,
    we reshape to [B, G], centre within groups to get advantages A_i, then
    replace each reward with:

        r_frpo_i = lambda * G * softmax(-A_i / lambda)_i

    This upweights completions with below-average reward so the GRPO gradient
    focuses on lifting the worst-case completions — the FRPO mechanism for
    finding flat reward regions.

    When lambda → ∞, softmax(-A/lambda) → uniform → r_frpo_i = lambda (constant),
    which after TRL centering gives zero advantage. To prevent this collapse we
    fall through to the raw rewards for lambda > lambda_passthrough.
    """

    def __init__(
        self,
        base_fn,
        num_generations: int,
        lambda_frpo: float,
        lambda_passthrough: float = 1e4,
    ):
        self.base = base_fn
        self.G = num_generations
        self.lam = lambda_frpo
        self.lam_pt = lambda_passthrough
        self.__name__ = getattr(base_fn, "__name__", "frpo_reward")

    def __call__(self, *args, **kwargs):
        raw = self.base(*args, **kwargs)          # tensor [B*G] from TRL

        # Pass through for very large lambda (standard GRPO territory)
        if self.lam >= self.lam_pt:
            return raw

        n = raw.numel()
        G = self.G
        if n % G != 0:
            # Can't infer group structure — fall back to raw rewards
            return raw

        B = n // G
        device = raw.device
        r = raw.reshape(B, G).float()

        # Per-group centred advantages
        A = r - r.mean(dim=1, keepdim=True)       # [B, G]

        # FRPO softmax reweighting: bad completions get high weight
        w = torch.softmax(-A / self.lam, dim=1)   # [B, G], sums to 1 per row

        # Effective reward that gives the FRPO gradient when TRL centres it
        r_frpo = self.lam * G * w                  # [B, G]

        return r_frpo.to(raw.dtype).reshape(n)


# ── helpers ────────────────────────────────────────────────────────────────────

def load_and_merge(model_path: str, tokenizer, dtype):
    """Load model; if it's a PEFT adapter, merge into base weights."""
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


def run_eval(model, tokenizer, queries_by_ds, gen_kwargs, label):
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
    parser = argparse.ArgumentParser(
        description="FRPO-GRPO: fine-tuning-robust watermark training"
    )

    # Model
    parser.add_argument("--warm-start-model", required=True,
                        help="Starting GRPO/SFT model checkpoint")
    parser.add_argument("--base-model", default=BASE_MODEL)

    # GRPO training
    parser.add_argument("--samples",          type=int,   default=1000)
    parser.add_argument("--epochs",           type=int,   default=3)
    parser.add_argument("--batch-size",       type=int,   default=4)
    parser.add_argument("--learning-rate",    type=float, default=1e-5)
    parser.add_argument("--num-generations",  type=int,   default=4)
    parser.add_argument("--max-new-tokens",   type=int,   default=200)
    parser.add_argument("--temperature",      type=float, default=0.7)
    parser.add_argument("--top-p",            type=float, default=0.9)
    parser.add_argument("--beta",             type=float, default=0.04)
    parser.add_argument("--train-dataset",    default="mixed",
                        choices=["eli5", "alpaca", "mixed", "gsm8k"])
    parser.add_argument("--implicit-fraction", type=float, default=0.4)

    # LoRA
    parser.add_argument("--lora-rank",  type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)

    # FRPO
    parser.add_argument("--lambda-frpo", type=float, default=0.5,
                        help=(
                            "FRPO risk parameter lambda. Small = strong robustness "
                            "signal (focuses gradient on worst completions). Large "
                            "(e.g. 1e6) = standard GRPO. "
                            "Rule of thumb: std(reward)/sqrt(2*0.3) ~ std/0.77. "
                            "Paper uses 0.2 for safety, 2.0 for math. "
                            "Default: 0.5"
                        ))

    # Eval
    parser.add_argument("--eval-samples",    type=int, default=100)
    parser.add_argument("--gen-batch",       type=int, default=4)
    parser.add_argument("--eval-max-tokens", type=int, default=512)
    parser.add_argument("--eval-min-tokens", type=int, default=256)

    # Output
    parser.add_argument("--output-dir",
                        default=f"frpo_grpo_logs/run_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    use_cuda = torch.cuda.is_available()
    use_bf16 = use_cuda and torch.cuda.is_bf16_supported()
    dtype = torch.bfloat16 if use_bf16 else torch.float16

    print("\n" + "=" * 70)
    print("FRPO-GRPO: Fine-tuning-Robust Watermark Training")
    print("=" * 70)
    print(f"Warm start:        {args.warm_start_model}")
    print(f"GRPO dataset:      {args.train_dataset} ({args.samples} samples x {args.epochs} epochs)")
    print(f"GRPO LoRA rank:    {args.lora_rank}")
    print(f"FRPO lambda:       {args.lambda_frpo}  (large = standard GRPO)")
    print(f"Num generations:   {args.num_generations}")
    print(f"Output:            {args.output_dir}")
    print("=" * 70 + "\n")

    # ── tokenizer ─────────────────────────────────────────────────────────────
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        args.warm_start_model, trust_remote_code=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    print("✓ Tokenizer loaded\n")

    # ── model ─────────────────────────────────────────────────────────────────
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

    # ── baseline eval ─────────────────────────────────────────────────────────
    run_eval(model, tokenizer, queries_by_ds, eval_gen_kwargs, "Before FRPO-GRPO")

    # ── GRPO training dataset ─────────────────────────────────────────────────
    print(f"\nLoading GRPO training dataset ({args.train_dataset}, {args.samples} samples)...")
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
    print(f"  Baseline mean={baseline_mean:.3f}  std={baseline_std:.3f}")

    # Suggest lambda based on reward std: lambda* ~ std / sqrt(2*rho), rho=0.3
    suggested_lambda = baseline_std / (2 * 0.3) ** 0.5
    print(f"  Suggested lambda (std/sqrt(2*0.3)): {suggested_lambda:.3f}")
    print(f"  Using lambda: {args.lambda_frpo}\n")

    # ── reward function with FRPO wrapper ────────────────────────────────────
    base_reward = WatermarkRewardFunction(
        "acrostics", baseline_mean, baseline_std,
        reward_shaping=False,
    )
    reward_fn = FRPORewardWrapper(
        base_fn=base_reward,
        num_generations=args.num_generations,
        lambda_frpo=args.lambda_frpo,
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
    training_args = build_grpo_config(
        base_training_args, generation_args, args.num_generations
    )

    print("Initializing GRPO trainer with FRPO reward wrapper...")
    trainer = build_grpo_trainer(
        model=model,
        training_args=training_args,
        train_dataset=tokenized_dataset,
        tokenizer=tokenizer,
        reward_fn=reward_fn,
        reference_model=None,
        require_explicit_reference=False,
    )
    print("✓ Trainer initialized\n")

    # ── train ─────────────────────────────────────────────────────────────────
    print("=" * 70)
    print("Starting FRPO-GRPO training...")
    print("=" * 70 + "\n")
    trainer.train()
    print("\n✓ Training complete\n")

    # ── save (merge LoRA into weights so downstream loading is correct) ───────
    final_model_path = os.path.join(args.output_dir, "final_model")
    print(f"Saving final model (merged) -> {final_model_path}")
    merged = trainer.model.merge_and_unload()
    merged.save_pretrained(final_model_path)
    tokenizer.save_pretrained(final_model_path)
    patch_saved_model_config(final_model_path, args.base_model)
    print("✓ Saved\n")

    # ── post-training eval ────────────────────────────────────────────────────
    run_eval(merged, tokenizer, queries_by_ds, eval_gen_kwargs, "After FRPO-GRPO")

    # ── save run config ───────────────────────────────────────────────────────
    config_path = os.path.join(args.output_dir, "run_config.json")
    with open(config_path, "w") as f:
        json.dump(vars(args), f, indent=2)
    print(f"Run config -> {config_path}")

    print("\n" + "=" * 70)
    print("Next step - test fine-tuning robustness:")
    print(f"  python finetune_robustness.py \\")
    print(f"      --grpo-model {final_model_path} \\")
    print(f"      --base-model {args.base_model} \\")
    print(f"      --finetune-dataset alpaca \\")
    print(f"      --finetune-samples 200 \\")
    print(f"      --max-steps 200 \\")
    print(f"      --lora-rank 4 \\")
    print(f"      --skip-base \\")
    print(f"      --output-dir {args.output_dir}/robustness_test")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()

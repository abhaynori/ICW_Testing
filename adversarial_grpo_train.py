#!/usr/bin/env python3
"""
Adversarial GRPO Training for Fine-tuning Robustness.

Algorithm: for each round —
  1. Evaluate watermark quality on current model
  2. Attack: apply LoRA fine-tuning on clean data (simulated adversary)
  3. Evaluate watermark quality after attack (should degrade)
  4. Recover: run GRPO training from the attacked model (re-watermark)
  5. Evaluate watermark quality after recovery
Repeat for --rounds rounds.

Final robustness test: attack the final model and compare against the original GRPO
model under the same attack to measure whether adversarial training helped.

Usage:
  python adversarial_grpo_train.py \
      --grpo-model rerun_acrostics_multidomain_42_20260525_112640/grpo_models/acrostics_full_20260525_230104/final_model \
      --rounds 3 \
      --attack-steps 100 \
      --attack-rank 4 \
      --attack-samples 200 \
      --recovery-samples 500 \
      --recovery-epochs 1 \
      --output-dir adversarial_grpo_logs/run_$(date +%Y%m%d_%H%M%S)
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from peft import LoraConfig, PeftModel, TaskType, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainerCallback,
    TrainingArguments,
)

sys.path.insert(0, str(Path(__file__).parent))
from main import acrostics_detector, secret_sequence, get_base_system_prompt
from grpo_train import generate_responses_batch
from finetune_robustness import load_eval_queries, score_queries, summarize
from sft_train import (
    SFTDataCollator,
    build_transformers_trainer,
    load_sft_pairs,
    prepare_sft_dataset,
)

BASE_MODEL = "Qwen/Qwen2.5-7B-Instruct"
EVAL_DATASETS = ["eli5", "alpaca", "gsm8k"]


# ── model utilities ────────────────────────────────────────────────────────────

def load_model(model_path: str, tokenizer, dtype):
    """Load model from path, merging any PEFT adapters into full weights."""
    adapter_cfg = os.path.join(model_path, "adapter_config.json")
    if os.path.exists(adapter_cfg):
        with open(adapter_cfg) as f:
            base_path = json.load(f)["base_model_name_or_path"]
        print(f"  Loading PEFT adapter (base: {base_path}) ...")
        base = AutoModelForCausalLM.from_pretrained(
            base_path, device_map="auto", trust_remote_code=True,
            low_cpu_mem_usage=True, torch_dtype=dtype,
        )
        model = PeftModel.from_pretrained(base, model_path)
        model = model.merge_and_unload()
        print("  ✓ Adapter merged into base weights")
    else:
        print(f"  Loading full model from {model_path} ...")
        model = AutoModelForCausalLM.from_pretrained(
            model_path, device_map="auto", trust_remote_code=True,
            low_cpu_mem_usage=True, torch_dtype=dtype,
        )
    model.config.pad_token_id = tokenizer.pad_token_id
    return model


def save_model(model, tokenizer, path: str):
    """Save model and tokenizer to path."""
    os.makedirs(path, exist_ok=True)
    model.save_pretrained(path)
    tokenizer.save_pretrained(path)
    print(f"  ✓ Model saved → {path}")


def free_model(model):
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ── watermark evaluation ───────────────────────────────────────────────────────

def eval_watermark(model, tokenizer, queries_by_ds: dict, gen_kwargs: dict) -> dict:
    """Score implicit watermark on each dataset. Returns summary dict."""
    results = {}
    for ds_name, queries in queries_by_ds.items():
        print(f"  [{ds_name}] Scoring {len(queries)} samples (implicit)...")
        scores = score_queries(model, tokenizer, queries, **gen_kwargs)
        s = summarize(scores)
        results[ds_name] = s
        print(f"    mean={s['mean']:.4f}  std={s['std']:.4f}  z={s['z']:.3f}  p={s['p']:.4e}")
    return results


def flat_eval_row(eval_results: dict, round_idx: int, phase: str) -> dict:
    row: dict = {"round": round_idx, "phase": phase, "timestamp": datetime.now().isoformat()}
    for ds, s in eval_results.items():
        for k, v in s.items():
            row[f"{ds}_{k}"] = v
    return row


# ── attack phase ───────────────────────────────────────────────────────────────

def run_attack(model, tokenizer, attack_dataset, args, round_idx: int):
    """
    Simulate adversarial fine-tuning: attach LoRA, fine-tune for attack_steps
    on clean data, merge LoRA back into model weights (in-place modification).

    Returns the modified model (same object, weights changed).
    """
    use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=args.attack_rank,
        lora_alpha=args.attack_rank * 2,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "up_proj", "down_proj", "gate_proj"],
        bias="none",
    )
    model = get_peft_model(model, lora_config)
    model.enable_input_require_grads()
    trainable, total = sum(p.numel() for p in model.parameters() if p.requires_grad), \
                       sum(p.numel() for p in model.parameters())
    print(f"  Attack LoRA: {trainable:,} trainable / {total:,} total params")

    ckpt_dir = os.path.join(args.output_dir, f"round_{round_idx}", "attack_ckpt")
    training_args = TrainingArguments(
        output_dir=ckpt_dir,
        max_steps=args.attack_steps,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=4,
        learning_rate=args.attack_lr,
        save_steps=args.attack_steps,
        save_total_limit=1,
        logging_steps=50,
        warmup_steps=min(10, args.attack_steps // 10),
        lr_scheduler_type="cosine",
        weight_decay=0.01,
        max_grad_norm=1.0,
        fp16=bool(torch.cuda.is_available() and not use_bf16),
        bf16=use_bf16,
        report_to=[],
        remove_unused_columns=False,
        gradient_checkpointing=True,
        seed=args.seed + round_idx,
    )

    trainer = build_transformers_trainer(
        model=model,
        training_args=training_args,
        train_dataset=attack_dataset,
        tokenizer=tokenizer,
        data_collator=SFTDataCollator(tokenizer),
    )
    trainer.train()

    # Merge attack LoRA back into base weights
    model = model.merge_and_unload()
    model.config.pad_token_id = tokenizer.pad_token_id
    print("  ✓ Attack LoRA merged")
    return model


# ── GRPO recovery phase ────────────────────────────────────────────────────────

def run_grpo_recovery(attacked_model_path: str, recovery_outdir: str, args, round_idx: int) -> str:
    """
    Run GRPO training starting from the attacked model via subprocess.
    Returns the path to the recovered model's final_model directory.
    """
    os.makedirs(recovery_outdir, exist_ok=True)
    log_path = os.path.join(recovery_outdir, "grpo_recovery.log")

    cmd = [
        "python", "grpo_train.py",
        "--model",              "full",
        "--method",             "acrostics",
        "--secret-sequence",    secret_sequence,
        "--train-dataset",      args.recovery_dataset,
        "--samples",            str(args.recovery_samples),
        "--epochs",             str(args.recovery_epochs),
        "--batch-size",         str(args.batch_size),
        "--learning-rate",      str(args.recovery_lr),
        "--num-generations",    "4",
        "--max-new-tokens",     "200",
        "--temperature",        "0.7",
        "--top-p",              "0.9",
        "--warm-start-model",   attacked_model_path,
        "--eval-datasets",      "gsm8k,eli5,alpaca",
        "--eval-samples",       "100",
        "--eval-splits",        "validation",
        "--eval-profiles",      "natural",
        "--eval-modes",         "implicit",
        "--output-dir",         recovery_outdir,
        "--seed",               str(args.seed + round_idx * 1000),
        "--allow-implicit-reference",
        "--no-reward-shaping",
    ]

    print(f"\n  Running GRPO recovery (round {round_idx}) — log: {log_path}")
    print(f"  Command: {' '.join(cmd)}\n")

    with open(log_path, "w") as logf:
        proc = subprocess.run(cmd, stdout=logf, stderr=subprocess.STDOUT, text=True)

    if proc.returncode != 0:
        print(f"  ⚠️  GRPO recovery subprocess exited with code {proc.returncode}")
        print(f"  Check log: {log_path}")
    else:
        print(f"  ✓ GRPO recovery complete")

    # Find final_model in the recovery output dir
    final_models = list(Path(recovery_outdir).glob("*/final_model"))
    if not final_models:
        raise FileNotFoundError(
            f"No final_model found under {recovery_outdir}. "
            f"Check {log_path} for errors."
        )
    final_model_path = str(sorted(final_models)[-1])
    print(f"  ✓ Recovered model: {final_model_path}")
    return final_model_path


# ── main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Adversarial GRPO Training")
    parser.add_argument("--grpo-model",
                        default="rerun_acrostics_multidomain_42_20260525_112640/grpo_models/acrostics_full_20260525_230104/final_model",
                        help="Pre-trained watermarked GRPO model to start from")
    parser.add_argument("--rounds",          type=int,   default=3,
                        help="Number of attack+recovery rounds (default: 3)")
    parser.add_argument("--attack-steps",    type=int,   default=100,
                        help="Fine-tuning steps per attack round (default: 100)")
    parser.add_argument("--attack-rank",     type=int,   default=4,
                        help="LoRA rank for adversarial fine-tuning (default: 4)")
    parser.add_argument("--attack-samples",  type=int,   default=200,
                        help="Clean data samples for each attack (default: 200)")
    parser.add_argument("--attack-dataset",  type=str,   default="alpaca",
                        choices=["eli5", "alpaca", "gsm8k"],
                        help="Dataset used for the adversarial attack (default: alpaca)")
    parser.add_argument("--attack-lr",       type=float, default=2e-5,
                        help="Learning rate for adversarial fine-tuning (default: 2e-5)")
    parser.add_argument("--recovery-samples",type=int,   default=500,
                        help="GRPO training samples per recovery round (default: 500)")
    parser.add_argument("--recovery-epochs", type=int,   default=1,
                        help="GRPO training epochs per recovery round (default: 1)")
    parser.add_argument("--recovery-dataset",type=str,   default="mixed",
                        choices=["eli5", "alpaca", "mixed", "gsm8k"],
                        help="Dataset for GRPO recovery (default: mixed)")
    parser.add_argument("--recovery-lr",     type=float, default=1e-5,
                        help="Learning rate for GRPO recovery (default: 1e-5)")
    parser.add_argument("--eval-samples",    type=int,   default=200,
                        help="Eval samples per dataset per eval (default: 200)")
    parser.add_argument("--gen-batch",       type=int,   default=4,
                        help="Generation batch size for eval (default: 4)")
    parser.add_argument("--max-new-tokens",  type=int,   default=512)
    parser.add_argument("--min-new-tokens",  type=int,   default=256)
    parser.add_argument("--temperature",     type=float, default=0.7)
    parser.add_argument("--top-p",           type=float, default=0.9)
    parser.add_argument("--batch-size",      type=int,   default=4,
                        help="Per-device batch size for training (default: 4)")
    parser.add_argument("--final-attack-steps", type=int, default=200,
                        help="Attack steps for the final robustness test (default: 200)")
    parser.add_argument("--output-dir",      type=str,
                        default=f"adversarial_grpo_logs/run_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    parser.add_argument("--seed",            type=int,   default=42)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    dtype = torch.bfloat16 if use_bf16 else torch.float16

    print("\n" + "=" * 70)
    print("ADVERSARIAL GRPO TRAINING")
    print("=" * 70)
    print(f"Starting model:    {args.grpo_model}")
    print(f"Rounds:            {args.rounds}")
    print(f"Attack steps/rank: {args.attack_steps} / rank {args.attack_rank}")
    print(f"Attack dataset:    {args.attack_dataset} ({args.attack_samples} samples)")
    print(f"Recovery:          {args.recovery_samples} samples × {args.recovery_epochs} epoch(s) GRPO")
    print(f"Recovery dataset:  {args.recovery_dataset}")
    print(f"Output:            {args.output_dir}")
    print("=" * 70 + "\n")

    # ── shared tokenizer ───────────────────────────────────────────────────────
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.grpo_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    print("✓ Tokenizer loaded\n")

    # ── shared eval queries ────────────────────────────────────────────────────
    print(f"Loading eval queries ({args.eval_samples} per dataset)...")
    queries_by_ds = {ds: load_eval_queries(ds, args.eval_samples) for ds in EVAL_DATASETS}
    for ds, q in queries_by_ds.items():
        print(f"  {ds}: {len(q)} queries")
    print()

    gen_kwargs = dict(
        gen_batch=args.gen_batch,
        max_new_tokens=args.max_new_tokens,
        min_new_tokens=args.min_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
    )

    # ── shared attack dataset (regenerated each round with different slice) ────
    def load_attack_data(round_idx: int):
        """Load attack dataset — different slice each round to vary the attack."""
        print(f"  Loading attack data ({args.attack_dataset}, {args.attack_samples} samples, round {round_idx})...")
        # Use offset to get different samples each round
        offset = round_idx * args.attack_samples
        raw_records = load_sft_pairs(
            dataset_name=args.attack_dataset,
            split="test",
            num_samples=args.attack_samples + offset,
        )
        # Slice to get this round's unique samples
        raw_records = raw_records[offset:offset + args.attack_samples]
        if len(raw_records) < args.attack_samples:
            # Wrap around if we run out of data
            raw_records = load_sft_pairs(
                dataset_name=args.attack_dataset, split="test",
                num_samples=args.attack_samples,
            )
        dataset = prepare_sft_dataset(
            records=raw_records, tokenizer=tokenizer,
            prompt_fn=None, include_instruction=False, max_length=1024,
        )
        print(f"  ✓ {len(dataset)} tokenized attack examples")
        return dataset

    # ── tracking ───────────────────────────────────────────────────────────────
    all_results: list[dict] = []
    csv_path = os.path.join(args.output_dir, "adversarial_grpo_results.csv")

    def save_results():
        pd.DataFrame(all_results).to_csv(csv_path, index=False)

    # ── load initial model ─────────────────────────────────────────────────────
    print("Loading initial GRPO model...")
    model = load_model(args.grpo_model, tokenizer, dtype)
    current_model_path = args.grpo_model  # track on-disk path for GRPO subprocess
    print()

    # ── initial eval ───────────────────────────────────────────────────────────
    print("=" * 60)
    print("INITIAL EVAL (before any attack)")
    print("=" * 60)
    eval_results = eval_watermark(model, tokenizer, queries_by_ds, gen_kwargs)
    all_results.append(flat_eval_row(eval_results, round_idx=0, phase="init"))
    save_results()

    # ── adversarial rounds ─────────────────────────────────────────────────────
    for round_idx in range(1, args.rounds + 1):
        print("\n" + "=" * 70)
        print(f"ROUND {round_idx} / {args.rounds}")
        print("=" * 70)

        round_dir = os.path.join(args.output_dir, f"round_{round_idx}")
        os.makedirs(round_dir, exist_ok=True)

        # ── Attack ────────────────────────────────────────────────────────────
        print(f"\n--- [Round {round_idx}] ATTACK ({args.attack_steps} steps, rank {args.attack_rank}) ---")
        attack_data = load_attack_data(round_idx - 1)
        model = run_attack(model, tokenizer, attack_data, args, round_idx)

        print(f"\n--- [Round {round_idx}] EVAL AFTER ATTACK ---")
        eval_results = eval_watermark(model, tokenizer, queries_by_ds, gen_kwargs)
        all_results.append(flat_eval_row(eval_results, round_idx=round_idx, phase="attacked"))
        save_results()

        # ── Save attacked model ────────────────────────────────────────────────
        attacked_model_path = os.path.join(round_dir, "attacked_model")
        print(f"\n  Saving attacked model → {attacked_model_path}")
        save_model(model, tokenizer, attacked_model_path)
        free_model(model)

        # ── GRPO Recovery ─────────────────────────────────────────────────────
        print(f"\n--- [Round {round_idx}] GRPO RECOVERY ({args.recovery_epochs} epoch(s)) ---")
        recovery_outdir = os.path.join(round_dir, "grpo_recovery")
        recovered_model_path = run_grpo_recovery(
            attacked_model_path, recovery_outdir, args, round_idx
        )

        # ── Reload recovered model ────────────────────────────────────────────
        print(f"\n  Reloading recovered model for eval...")
        model = load_model(recovered_model_path, tokenizer, dtype)
        current_model_path = recovered_model_path

        print(f"\n--- [Round {round_idx}] EVAL AFTER RECOVERY ---")
        eval_results = eval_watermark(model, tokenizer, queries_by_ds, gen_kwargs)
        all_results.append(flat_eval_row(eval_results, round_idx=round_idx, phase="recovered"))
        save_results()

    # ── Final robustness test ──────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print(f"FINAL ROBUSTNESS TEST ({args.final_attack_steps} steps, rank {args.attack_rank})")
    print("Testing: does the adversarially-trained model survive a fresh attack?")
    print("=" * 70)

    # Prepare a fresh attack dataset (test split, not seen during training)
    print("\nLoading fresh attack dataset (different split)...")
    final_attack_records = load_sft_pairs(
        dataset_name=args.attack_dataset,
        split="test",
        num_samples=args.attack_samples,
    )
    final_attack_data = prepare_sft_dataset(
        records=final_attack_records, tokenizer=tokenizer,
        prompt_fn=None, include_instruction=False, max_length=1024,
    )

    # Attack the final adversarially-trained model
    final_args = argparse.Namespace(**vars(args))
    final_args.attack_steps = args.final_attack_steps
    model = run_attack(model, tokenizer, final_attack_data, final_args, round_idx=99)

    print("\n--- EVAL: Final adversarial model after fresh attack ---")
    eval_results = eval_watermark(model, tokenizer, queries_by_ds, gen_kwargs)
    all_results.append(flat_eval_row(eval_results, round_idx=args.rounds + 1, phase="final_attacked"))
    save_results()
    free_model(model)

    # ── Print summary ──────────────────────────────────────────────────────────
    df = pd.DataFrame(all_results)
    print("\n" + "=" * 70)
    print("ADVERSARIAL GRPO SUMMARY")
    print("=" * 70)
    print(f"\n{'Phase':<20} {'Round':>5}  {'ELI5 mean':>10}  {'Alpaca mean':>12}  {'GSM8K mean':>11}")
    print("-" * 65)
    for _, row in df.iterrows():
        phase = row["phase"]
        rnd   = int(row["round"])
        eli5  = row.get("eli5_mean",   float("nan"))
        alp   = row.get("alpaca_mean", float("nan"))
        gsm   = row.get("gsm8k_mean",  float("nan"))
        print(f"{phase:<20} {rnd:>5}  {eli5:>10.4f}  {alp:>12.4f}  {gsm:>11.4f}")
    print("=" * 70)
    print(f"\n✓ Results saved → {csv_path}")
    print(f"\nKey question: do 'attacked' scores improve across rounds?")
    print("If so, adversarial GRPO training is increasing fine-tuning robustness.\n")


if __name__ == "__main__":
    main()

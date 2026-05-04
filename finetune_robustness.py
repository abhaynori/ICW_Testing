#!/usr/bin/env python3
"""
Phase 3: Fine-tuning robustness for the ICW acrostics watermark.

Runs the same LoRA fine-tuning sweep on BOTH the GRPO watermarked model
and the base (unwatermarked) model, evaluating implicit watermark signal
at steps: 0, 100, 250, 500, 1000.

Usage:
  python finetune_robustness.py                          # defaults
  python finetune_robustness.py --max-steps 500 --eval-samples 100
"""

from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from peft import LoraConfig, TaskType, get_peft_model
from scipy import stats as scipy_stats
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainerCallback,
    TrainingArguments,
)

sys.path.insert(0, str(Path(__file__).parent))
from main import acrostics_detector, secret_sequence, get_base_system_prompt
from grpo_train import generate_responses_batch
from sft_train import (
    SFTDataCollator,
    build_transformers_trainer,
    load_sft_pairs,
    prepare_sft_dataset,
    _format_alpaca_query,
)

# ── globals ────────────────────────────────────────────────────────────────────
EVAL_STEPS = [0, 100, 250, 500, 1000]
DEFAULT_GRPO_MODEL = "grpo_models/acrostics_full_20260318_234510/final_model"
DEFAULT_BASE_MODEL = "Qwen/Qwen2.5-7B-Instruct"


# ── data loading ───────────────────────────────────────────────────────────────

def load_eval_queries(dataset_name: str, n: int) -> list[str]:
    """Load N validation-split queries for watermark evaluation."""
    from datasets import load_dataset

    if dataset_name == "eli5":
        ds = load_dataset("sentence-transformers/eli5", "pair", split="train")
        start = int(len(ds) * 0.8)
        end = int(len(ds) * 0.9)
        subset = ds.select(range(start, min(start + n, end)))
        queries = [row["question"].strip() for row in subset if row.get("question")]
    elif dataset_name == "alpaca":
        ds = load_dataset("yahma/alpaca-cleaned", split="train")
        start = int(len(ds) * 0.8)
        end = int(len(ds) * 0.9)
        subset = ds.select(range(start, min(start + n, end)))
        queries = [_format_alpaca_query(row) for row in subset]
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    return [q for q in queries if q][:n]


# ── scoring ────────────────────────────────────────────────────────────────────

def score_queries(
    model,
    tokenizer,
    queries: list[str],
    gen_batch: int = 4,
    max_new_tokens: int = 512,
    min_new_tokens: int = 256,
    temperature: float = 0.7,
    top_p: float = 0.9,
) -> list[float]:
    """Generate responses (implicit mode) and return per-sample acrostics scores."""
    scores: list[float] = []
    was_training = model.training
    model.eval()

    for i in range(0, len(queries), gen_batch):
        batch = [
            [
                {"role": "system", "content": get_base_system_prompt()},
                {"role": "user", "content": q},
            ]
            for q in queries[i : i + gen_batch]
        ]
        responses = generate_responses_batch(
            model, tokenizer, batch,
            max_new_tokens=max_new_tokens,
            min_new_tokens=min_new_tokens,
            temperature=temperature,
            top_p=top_p,
        )
        for resp in responses:
            scores.append(acrostics_detector(resp, secret_sequence))

    if was_training:
        model.train()
    return scores


def summarize(scores: list[float]) -> dict:
    arr = np.array(scores)
    mean, std, n = float(arr.mean()), float(arr.std(ddof=1)), len(arr)
    z = mean / (std / np.sqrt(n)) if (std > 0 and n > 1) else 0.0
    p = float(2 * scipy_stats.norm.sf(abs(z)))
    return {"mean": mean, "std": std, "n": n, "z": z, "p": p}


# ── callback ───────────────────────────────────────────────────────────────────

class WatermarkEvalCallback(TrainerCallback):
    """Runs implicit watermark eval at each milestone step."""

    def __init__(self, *, eval_steps, model, tokenizer,
                 eli5_queries, alpaca_queries, results, out_dir,
                 model_label: str, gen_kwargs: dict):
        self.eval_steps = set(eval_steps)
        self.model = model
        self.tokenizer = tokenizer
        self.eli5_queries = eli5_queries
        self.alpaca_queries = alpaca_queries
        self.results = results
        self.out_dir = out_dir
        self.model_label = model_label
        self.gen_kwargs = gen_kwargs

    def _run_eval(self, step: int) -> None:
        print(f"\n{'─'*60}")
        print(f"  [{self.model_label}] Watermark eval at step {step}")
        print(f"{'─'*60}")
        row: dict = {
            "model": self.model_label,
            "step": step,
            "timestamp": datetime.now().isoformat(),
        }

        for ds_name, queries in [("eli5", self.eli5_queries), ("alpaca", self.alpaca_queries)]:
            print(f"  [{ds_name}] Scoring {len(queries)} samples (implicit)...")
            scores = score_queries(self.model, self.tokenizer, queries, **self.gen_kwargs)
            s = summarize(scores)
            for k, v in s.items():
                row[f"{ds_name}_val_implicit_{k}"] = v
            print(f"    mean={s['mean']:.4f}  std={s['std']:.4f}  z={s['z']:.3f}  p={s['p']:.4e}")

        self.results.append(row)
        csv_path = os.path.join(self.out_dir, "finetune_robustness_results.csv")
        pd.DataFrame(self.results).to_csv(csv_path, index=False)
        print(f"  Saved → {csv_path}")

    def on_train_begin(self, args, state, control, **kwargs):
        if 0 in self.eval_steps:
            self._run_eval(0)

    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step in self.eval_steps:
            self._run_eval(state.global_step)


# ── sweep for one model ────────────────────────────────────────────────────────

def run_sweep(
    *,
    model_path: str,
    model_label: str,
    train_dataset,
    tokenizer,
    eli5_queries: list[str],
    alpaca_queries: list[str],
    results: list[dict],
    args,
    use_bf16: bool,
) -> None:
    """Load model_path, attach LoRA, fine-tune, eval at EVAL_STEPS."""
    print(f"\n{'='*70}")
    print(f"  Running sweep: {model_label}")
    print(f"  Model path:   {model_path}")
    print(f"{'='*70}\n")

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map="auto",
        trust_remote_code=True,
        low_cpu_mem_usage=True,
        torch_dtype=torch.bfloat16 if use_bf16 else torch.float16,
    )
    model.config.pad_token_id = tokenizer.pad_token_id

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=args.lora_rank,
        lora_alpha=args.lora_rank * 2,
        lora_dropout=0.05,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "up_proj", "down_proj", "gate_proj",
        ],
        bias="none",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    model.enable_input_require_grads()
    print(f"✓ LoRA attached to {model_label}\n")

    gen_kwargs = dict(
        gen_batch=args.gen_batch,
        max_new_tokens=args.max_new_tokens,
        min_new_tokens=args.min_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
    )
    callback = WatermarkEvalCallback(
        eval_steps=EVAL_STEPS,
        model=model,
        tokenizer=tokenizer,
        eli5_queries=eli5_queries,
        alpaca_queries=alpaca_queries,
        results=results,
        out_dir=args.output_dir,
        model_label=model_label,
        gen_kwargs=gen_kwargs,
    )

    use_cuda = torch.cuda.is_available()
    ckpt_dir = os.path.join(args.output_dir, f"checkpoints_{model_label}")
    training_args = TrainingArguments(
        output_dir=ckpt_dir,
        max_steps=args.max_steps,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=4,
        learning_rate=args.learning_rate,
        save_steps=args.max_steps,
        save_total_limit=1,
        logging_steps=50,
        warmup_steps=20,
        lr_scheduler_type="cosine",
        weight_decay=0.01,
        max_grad_norm=1.0,
        fp16=bool(use_cuda and not use_bf16),
        bf16=use_bf16,
        report_to=[],
        remove_unused_columns=False,
        gradient_checkpointing=True,
        seed=args.seed,
    )

    trainer = build_transformers_trainer(
        model=model,
        training_args=training_args,
        train_dataset=train_dataset,
        tokenizer=tokenizer,
        data_collator=SFTDataCollator(tokenizer),
    )
    trainer.add_callback(callback)
    trainer.train()

    # Free VRAM before loading the next model
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ── main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 3: Fine-tuning robustness")
    parser.add_argument("--grpo-model", default=DEFAULT_GRPO_MODEL,
                        help="Path to the watermarked GRPO model checkpoint")
    parser.add_argument("--base-model", default=DEFAULT_BASE_MODEL,
                        help="Path or HF hub name of the unwatermarked base model")
    parser.add_argument("--skip-base", action="store_true",
                        help="Skip the base model sweep (run GRPO only)")
    parser.add_argument("--finetune-dataset", default="alpaca",
                        choices=["eli5", "alpaca"],
                        help="Clean dataset for adversarial fine-tuning (test split)")
    parser.add_argument("--finetune-samples", type=int, default=2000)
    parser.add_argument("--max-steps", type=int, default=1000)
    parser.add_argument("--eval-samples", type=int, default=200)
    parser.add_argument("--gen-batch", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--min-new-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--lora-rank", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--output-dir", default="robustness_logs/finetune")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()

    print(f"\n{'='*70}")
    print("Phase 3: Fine-tuning Robustness  (GRPO vs Base)")
    print(f"{'='*70}")
    print(f"GRPO model:     {args.grpo_model}")
    print(f"Base model:     {'(skipped)' if args.skip_base else args.base_model}")
    print(f"Fine-tune data: {args.finetune_dataset} (test split, {args.finetune_samples} samples)")
    print(f"Max steps:      {args.max_steps}")
    print(f"Eval at steps:  {EVAL_STEPS}")
    print(f"Eval samples:   {args.eval_samples} per dataset")
    print(f"LoRA rank:      {args.lora_rank}")
    print(f"Learning rate:  {args.learning_rate}")
    print(f"Output:         {args.output_dir}")
    print(f"{'='*70}\n")

    # ── shared tokenizer (both models use the same Qwen tokenizer) ─────────────
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        args.grpo_model, trust_remote_code=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    print("✓ Tokenizer loaded\n")

    # ── shared eval queries ────────────────────────────────────────────────────
    print(f"Loading eval queries ({args.eval_samples} per dataset)...")
    eli5_queries   = load_eval_queries("eli5",   args.eval_samples)
    alpaca_queries = load_eval_queries("alpaca", args.eval_samples)
    print(f"✓ ELI5: {len(eli5_queries)}  Alpaca: {len(alpaca_queries)}\n")

    # ── shared fine-tuning dataset (same data, same order for both runs) ───────
    print(f"Loading clean fine-tuning data ({args.finetune_dataset}, test split)...")
    raw_records = load_sft_pairs(
        dataset_name=args.finetune_dataset,
        split="test",
        num_samples=args.finetune_samples,
    )
    train_dataset = prepare_sft_dataset(
        records=raw_records,
        tokenizer=tokenizer,
        prompt_fn=None,
        include_instruction=False,
        max_length=1024,
    )
    print(f"✓ Built {len(train_dataset)} tokenized training examples\n")

    # ── run sweeps ─────────────────────────────────────────────────────────────
    results: list[dict] = []

    run_sweep(
        model_path=args.grpo_model,
        model_label="grpo",
        train_dataset=train_dataset,
        tokenizer=tokenizer,
        eli5_queries=eli5_queries,
        alpaca_queries=alpaca_queries,
        results=results,
        args=args,
        use_bf16=use_bf16,
    )

    if not args.skip_base:
        run_sweep(
            model_path=args.base_model,
            model_label="base",
            train_dataset=train_dataset,
            tokenizer=tokenizer,
            eli5_queries=eli5_queries,
            alpaca_queries=alpaca_queries,
            results=results,
            args=args,
            use_bf16=use_bf16,
        )

    # ── final summary ──────────────────────────────────────────────────────────
    df = pd.DataFrame(results)
    csv_path = os.path.join(args.output_dir, "finetune_robustness_results.csv")
    df.to_csv(csv_path, index=False)

    print("\n" + "=" * 70)
    print("FINE-TUNING ROBUSTNESS SUMMARY  (implicit, validation split)")
    for label in df["model"].unique():
        sub = df[df["model"] == label]
        print(f"\n  Model: {label}")
        print(f"  {'Step':>6}  {'ELI5 mean':>10}  {'ELI5 z':>8}  {'ELI5 p':>12}  "
              f"{'Alpaca mean':>12}  {'Alpaca z':>8}  {'Alpaca p':>12}")
        print("  " + "-" * 76)
        for _, row in sub.iterrows():
            print(f"  {int(row['step']):>6}  "
                  f"{row['eli5_val_implicit_mean']:>10.4f}  "
                  f"{row['eli5_val_implicit_z']:>8.3f}  "
                  f"{row['eli5_val_implicit_p']:>12.4e}  "
                  f"{row['alpaca_val_implicit_mean']:>12.4f}  "
                  f"{row['alpaca_val_implicit_z']:>8.3f}  "
                  f"{row['alpaca_val_implicit_p']:>12.4e}")
    print("=" * 70)
    print(f"\n✓ Full results saved to: {csv_path}")


if __name__ == "__main__":
    main()

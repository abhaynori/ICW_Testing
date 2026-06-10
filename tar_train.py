#!/usr/bin/env python3
"""
TAR-style tamper-resistance training for the ICW acrostics watermark.

Adapts "Tamper-Resistant Safeguards for Open-Weight LLMs" (Tamirisa et al.,
ICLR 2025) to watermark preservation.  The detector reward is not
differentiable, so the tamper-resistance (TR) loss is cross-entropy on
self-distilled watermarked completions: minimizing post-attack CE on
watermarked text preserves the watermark after fine-tuning.

Each outer step (first-order MAML, as in TAR):
  1. Snapshot the defense LoRA adapter weights.
  2. Inner loop: simulate a fine-tuning attack -- K steps on benign alpaca
     data (TRAIN split; the robustness eval attacks with the TEST split so
     there is no leakage), with lr / K / optimizer sampled per outer step.
  3. Accumulate TR gradients (CE on watermarked data) at points along the
     attack trajectory.
  4. Restore the adapter snapshot, compute the retain gradient (CE on
     watermarked data at the unattacked point), combine, and take the
     outer optimizer step.

Pipeline position: run on the watermarked model (FRPO- or GRPO-trained),
then evaluate with finetune_robustness.py as usual.

Usage:
  python tar_train.py \
      --model frpo_grpo_logs/run_frpo_from_sft_20260609_114832/final_model \
      --output-dir tar_logs/run1
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from peft import LoraConfig, PeftModel, TaskType, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).parent))
from main import acrostics_detector, secret_sequence, get_base_system_prompt
from grpo_train import generate_responses_batch
from sft_train import SFTDataCollator, load_sft_pairs, prepare_sft_dataset
from finetune_robustness import load_eval_queries, score_queries, summarize


# ── model loading ──────────────────────────────────────────────────────────────

def load_watermarked_model(model_path: str, dtype):
    """Load the watermarked model, merging a PEFT adapter if present."""
    adapter_cfg = os.path.join(model_path, "adapter_config.json")
    if os.path.exists(adapter_cfg):
        with open(adapter_cfg) as f:
            base_path = json.load(f)["base_model_name_or_path"]
        print(f"  Detected PEFT adapter. Base model: {base_path}")
        base = AutoModelForCausalLM.from_pretrained(
            base_path, device_map="auto", trust_remote_code=True,
            low_cpu_mem_usage=True, torch_dtype=dtype,
        )
        model = PeftModel.from_pretrained(base, model_path)
        model = model.merge_and_unload()
        print("  ✓ Adapter merged into base weights")
        return model
    return AutoModelForCausalLM.from_pretrained(
        model_path, device_map="auto", trust_remote_code=True,
        low_cpu_mem_usage=True, torch_dtype=dtype,
    )


# ── watermarked self-distillation data ─────────────────────────────────────────

def load_wm_prompts(datasets: list[str], n_per: int) -> list[dict]:
    """TRAIN-split queries for self-distilling watermarked completions."""
    prompts = []
    for ds_name in datasets:
        pairs = load_sft_pairs(dataset_name=ds_name, split="train", num_samples=n_per)
        for p in pairs:
            prompts.append({"dataset": ds_name, "query": p["query"]})
    return prompts


def generate_wm_data(
    model, tokenizer, prompts: list[dict], *,
    threshold: float, gen_batch: int, max_new_tokens: int,
    min_new_tokens: int, temperature: float, top_p: float,
) -> list[dict]:
    """Sample implicit-mode completions and keep detector-confirmed ones."""
    records: list[dict] = []
    was_training = model.training
    model.eval()
    orig_padding_side = tokenizer.padding_side
    tokenizer.padding_side = "left"

    for i in range(0, len(prompts), gen_batch):
        chunk = prompts[i : i + gen_batch]
        batch = [
            [
                {"role": "system", "content": get_base_system_prompt()},
                {"role": "user", "content": p["query"]},
            ]
            for p in chunk
        ]
        responses = generate_responses_batch(
            model, tokenizer, batch,
            max_new_tokens=max_new_tokens,
            min_new_tokens=min_new_tokens,
            temperature=temperature,
            top_p=top_p,
        )
        for p, resp in zip(chunk, responses):
            score = acrostics_detector(resp, secret_sequence)
            if score < threshold:
                records.append({
                    "dataset": p["dataset"],
                    "query": p["query"],
                    "target": resp,
                    "detector_score": score,
                })
        if (i // gen_batch) % 10 == 0:
            kept = len(records)
            print(f"  Progress: {min(i + gen_batch, len(prompts))}/{len(prompts)}"
                  f"  kept={kept}")

    tokenizer.padding_side = orig_padding_side
    if was_training:
        model.train()
    return records


# ── infinite batch iterators ───────────────────────────────────────────────────

def cycle_loader(dataset, collator, batch_size: int, seed: int):
    g = torch.Generator()
    g.manual_seed(seed)
    while True:
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=True,
            collate_fn=collator, generator=g,
        )
        for batch in loader:
            yield batch


def to_device(batch: dict, device) -> dict:
    return {k: v.to(device) for k, v in batch.items()}


# ── TAR outer loop ─────────────────────────────────────────────────────────────

def ce_loss(model, batch) -> torch.Tensor:
    out = model(
        input_ids=batch["input_ids"],
        attention_mask=batch["attention_mask"],
        labels=batch["labels"],
    )
    return out.loss


def snapshot_params(params: dict) -> dict:
    return {n: p.detach().clone() for n, p in params.items()}


def restore_params(params: dict, snapshot: dict) -> None:
    with torch.no_grad():
        for n, p in params.items():
            p.data.copy_(snapshot[n])


def main() -> None:
    parser = argparse.ArgumentParser(description="TAR-style watermark tamper-resistance")
    parser.add_argument("--model", required=True,
                        help="Path to the watermarked model (FRPO/GRPO final_model)")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--seed", type=int, default=42)

    # watermarked self-distillation data
    parser.add_argument("--wm-data", default=None,
                        help="Cached watermarked completions JSON (generated if missing)")
    parser.add_argument("--wm-datasets", default="eli5,alpaca,gsm8k")
    parser.add_argument("--wm-samples-per", type=int, default=300,
                        help="Prompts per dataset for self-distillation")
    parser.add_argument("--wm-threshold", type=float, default=0.05,
                        help="Keep completions with detector p-value below this")

    # generation (self-distill + monitoring evals)
    parser.add_argument("--gen-batch", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--min-new-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.9)

    # TAR loop
    parser.add_argument("--outer-steps", type=int, default=300)
    parser.add_argument("--outer-lr", type=float, default=1e-4)
    parser.add_argument("--lora-rank", type=int, default=32)
    parser.add_argument("--inner-steps-choices", default="4,8,16",
                        help="Attack length K sampled from these each outer step")
    parser.add_argument("--attack-lr-choices", default="1e-5,2e-5,5e-5",
                        help="Attack lr sampled from these each outer step")
    parser.add_argument("--attack-batch", type=int, default=4)
    parser.add_argument("--attack-samples", type=int, default=2000,
                        help="Alpaca TRAIN-split samples for simulated attacks")
    parser.add_argument("--tr-every", type=int, default=2,
                        help="Accumulate TR gradient every N inner attack steps")
    parser.add_argument("--tr-batch", type=int, default=2)
    parser.add_argument("--retain-batch", type=int, default=2)
    parser.add_argument("--lambda-tr", type=float, default=1.0)
    parser.add_argument("--lambda-retain", type=float, default=1.0)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)

    # monitoring
    parser.add_argument("--eval-every", type=int, default=50)
    parser.add_argument("--eval-samples", type=int, default=50)
    parser.add_argument("--save-every", type=int, default=100)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    rng = random.Random(args.seed)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = args.output_dir or f"tar_logs/run_{timestamp}"
    os.makedirs(out_dir, exist_ok=True)

    wm_datasets = [d.strip() for d in args.wm_datasets.split(",") if d.strip()]
    inner_choices = [int(x) for x in args.inner_steps_choices.split(",")]
    attack_lrs = [float(x) for x in args.attack_lr_choices.split(",")]

    use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    dtype = torch.bfloat16 if use_bf16 else torch.float16

    print(f"\n{'='*70}")
    print("TAR-style Tamper-Resistance Training (watermark preservation)")
    print(f"{'='*70}")
    print(f"Watermarked model: {args.model}")
    print(f"Output:            {out_dir}")
    print(f"Outer steps:       {args.outer_steps}  (lr={args.outer_lr}, LoRA r={args.lora_rank})")
    print(f"Attack sampling:   K∈{inner_choices}  lr∈{attack_lrs}  data=alpaca[train]")
    print(f"TR/retain lambdas: {args.lambda_tr} / {args.lambda_retain}")
    print(f"{'='*70}\n")

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    print("Loading watermarked model...")
    model = load_watermarked_model(args.model, dtype)
    model.config.pad_token_id = tokenizer.pad_token_id

    # ── watermarked self-distillation data ─────────────────────────────────────
    wm_data_path = args.wm_data or os.path.join(out_dir, "wm_selfdistill_data.json")
    if os.path.exists(wm_data_path):
        print(f"\nLoading cached watermarked data: {wm_data_path}")
        with open(wm_data_path) as f:
            wm_records = json.load(f)
    else:
        print(f"\nSelf-distilling watermarked completions "
              f"({args.wm_samples_per} prompts x {wm_datasets})...")
        prompts = load_wm_prompts(wm_datasets, args.wm_samples_per)
        wm_records = generate_wm_data(
            model, tokenizer, prompts,
            threshold=args.wm_threshold,
            gen_batch=args.gen_batch,
            max_new_tokens=args.max_new_tokens,
            min_new_tokens=args.min_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
        )
        with open(wm_data_path, "w") as f:
            json.dump(wm_records, f, indent=2)
        print(f"✓ Saved {len(wm_records)} watermarked records → {wm_data_path}")
    kept_frac = len(wm_records) / max(1, len(wm_datasets) * args.wm_samples_per)
    print(f"✓ Watermarked data: {len(wm_records)} records "
          f"(keep rate ≈ {kept_frac:.1%}, threshold p<{args.wm_threshold})")
    if len(wm_records) < 100:
        print("⚠️  Fewer than 100 watermarked records — consider lowering "
              "--wm-threshold strictness or raising --wm-samples-per")

    wm_dataset = prepare_sft_dataset(
        records=wm_records, tokenizer=tokenizer,
        prompt_fn=None, include_instruction=False, max_length=1024,
    )

    # ── attack data (alpaca TRAIN split; eval attack uses TEST split) ──────────
    print(f"\nLoading attack data (alpaca, train split, {args.attack_samples})...")
    attack_records = load_sft_pairs(
        dataset_name="alpaca", split="train", num_samples=args.attack_samples,
    )
    attack_dataset = prepare_sft_dataset(
        records=attack_records, tokenizer=tokenizer,
        prompt_fn=None, include_instruction=False, max_length=1024,
    )
    print(f"✓ Attack dataset: {len(attack_dataset)} examples")

    collator = SFTDataCollator(tokenizer)
    wm_iter = cycle_loader(wm_dataset, collator, args.tr_batch, args.seed)
    retain_iter = cycle_loader(wm_dataset, collator, args.retain_batch, args.seed + 1)
    attack_iter = cycle_loader(attack_dataset, collator, args.attack_batch, args.seed + 2)

    # ── eval queries for monitoring ────────────────────────────────────────────
    print(f"\nLoading monitoring eval queries ({args.eval_samples} per dataset)...")
    eval_queries = {
        "eli5": load_eval_queries("eli5", args.eval_samples),
        "alpaca": load_eval_queries("alpaca", args.eval_samples),
    }

    # ── defense adapter ────────────────────────────────────────────────────────
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=args.lora_rank,
        lora_alpha=args.lora_rank * 2,
        lora_dropout=0.0,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "up_proj", "down_proj", "gate_proj",
        ],
        bias="none",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    model.enable_input_require_grads()
    # Non-reentrant checkpointing is required for torch.autograd.grad (the
    # reentrant variant only populates .grad, breaking the TR gradient).
    model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False}
    )
    model.train()
    device = next(model.parameters()).device

    trainable = {n: p for n, p in model.named_parameters() if p.requires_grad}
    param_list = list(trainable.values())
    outer_opt = torch.optim.AdamW(param_list, lr=args.outer_lr, weight_decay=0.0)

    gen_kwargs = dict(
        gen_batch=args.gen_batch,
        max_new_tokens=args.max_new_tokens,
        min_new_tokens=args.min_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
    )

    def quick_eval(step: int) -> None:
        print(f"\n{'─'*60}\n  Monitoring eval at outer step {step}\n{'─'*60}")
        row = {"step": step}
        for ds_name, queries in eval_queries.items():
            scores = score_queries(model, tokenizer, queries, **gen_kwargs)
            s = summarize(scores)
            row[ds_name] = s
            print(f"  [{ds_name}] mean={s['mean']:.4f}  z={s['z']:.3f}  p={s['p']:.4e}")
        with open(os.path.join(out_dir, "monitor_evals.jsonl"), "a") as f:
            f.write(json.dumps(row) + "\n")
        model.train()

    quick_eval(0)

    # ── TAR outer loop ─────────────────────────────────────────────────────────
    log_path = os.path.join(out_dir, "tar_train_log.jsonl")
    for outer_step in range(1, args.outer_steps + 1):
        k_attack = rng.choice(inner_choices)
        attack_lr = rng.choice(attack_lrs)
        attack_opt_name = rng.choice(["adamw", "sgd"])

        snapshot = snapshot_params(trainable)

        # inner loop: simulated fine-tuning attack
        if attack_opt_name == "adamw":
            attack_opt = torch.optim.AdamW(param_list, lr=attack_lr)
        else:
            attack_opt = torch.optim.SGD(param_list, lr=attack_lr * 10)

        tr_grads = {n: torch.zeros_like(p) for n, p in trainable.items()}
        n_tr_points = 0
        attack_losses, tr_losses = [], []

        for k in range(1, k_attack + 1):
            batch = to_device(next(attack_iter), device)
            loss = ce_loss(model, batch)
            attack_opt.zero_grad(set_to_none=True)
            loss.backward()
            attack_opt.step()
            attack_losses.append(loss.item())

            # TR gradient at this point of the attack trajectory
            if k % args.tr_every == 0 or k == k_attack:
                tr_batch = to_device(next(wm_iter), device)
                tr_loss = ce_loss(model, tr_batch)
                grads = torch.autograd.grad(
                    tr_loss, param_list, allow_unused=True,
                )
                for (n, _), g in zip(trainable.items(), grads):
                    if g is not None:
                        tr_grads[n] += g.detach()
                n_tr_points += 1
                tr_losses.append(tr_loss.item())

        attack_opt.zero_grad(set_to_none=True)
        del attack_opt

        # restore pre-attack adapter weights
        restore_params(trainable, snapshot)
        del snapshot

        # retain gradient at the unattacked point
        retain_batch = to_device(next(retain_iter), device)
        retain_loss = ce_loss(model, retain_batch)
        outer_opt.zero_grad(set_to_none=True)
        (args.lambda_retain * retain_loss).backward()

        # combine: retain grad (in .grad) + averaged TR trajectory grads
        with torch.no_grad():
            for n, p in trainable.items():
                tr_g = tr_grads[n] / max(1, n_tr_points)
                if p.grad is None:
                    p.grad = args.lambda_tr * tr_g
                else:
                    p.grad += args.lambda_tr * tr_g
        grad_norm = torch.nn.utils.clip_grad_norm_(param_list, args.max_grad_norm)
        outer_opt.step()
        outer_opt.zero_grad(set_to_none=True)
        del tr_grads

        rec = {
            "outer_step": outer_step,
            "k_attack": k_attack,
            "attack_lr": attack_lr,
            "attack_opt": attack_opt_name,
            "attack_loss_first": attack_losses[0],
            "attack_loss_last": attack_losses[-1],
            "tr_loss_mean": float(np.mean(tr_losses)) if tr_losses else None,
            "retain_loss": retain_loss.item(),
            "grad_norm": float(grad_norm),
        }
        with open(log_path, "a") as f:
            f.write(json.dumps(rec) + "\n")
        if outer_step % 10 == 0 or outer_step == 1:
            print(f"[{outer_step:>4}/{args.outer_steps}] "
                  f"K={k_attack:<2} lr={attack_lr:.0e} opt={attack_opt_name:<5} "
                  f"tr_ce={rec['tr_loss_mean']:.4f} "
                  f"retain_ce={rec['retain_loss']:.4f} "
                  f"gnorm={rec['grad_norm']:.3f}")

        if outer_step % args.eval_every == 0:
            quick_eval(outer_step)

        if outer_step % args.save_every == 0:
            ckpt = os.path.join(out_dir, f"adapter_step{outer_step}")
            model.save_pretrained(ckpt)
            print(f"  ✓ Adapter checkpoint → {ckpt}")

    # ── save merged final model ────────────────────────────────────────────────
    final_path = os.path.join(out_dir, "final_model")
    print(f"\nMerging defense adapter and saving → {final_path}")
    merged = model.merge_and_unload()
    merged.save_pretrained(final_path)
    tokenizer.save_pretrained(final_path)

    print(f"\n✓ Done. Test robustness with:")
    print(f"  python finetune_robustness.py "
          f"--grpo-model {final_path} "
          f"--base-model Qwen/Qwen2.5-7B-Instruct "
          f"--finetune-dataset alpaca --finetune-samples 200 "
          f"--max-steps 200 --lora-rank 4 --skip-base "
          f"--output-dir {os.path.join(out_dir, 'robustness_test')}")


if __name__ == "__main__":
    main()

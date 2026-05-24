#!/usr/bin/env python3
"""
Paraphrase robustness evaluation for ICW acrostics watermark.

Takes outputs from a GRPO model eval JSON, paraphrases each response using
a local Qwen model (or optionally OpenAI if OPENAI_API_KEY is set), then
re-runs the acrostics detector to measure how much signal survives.

Usage:
  # Use a local paraphraser (Qwen2.5-7B, no API key needed)
  python paraphrase_robustness.py \\
      --eval-json rerun_acrostics_42_20260522_235846/eval/grpo/eval_gsm8k_test_implicit_natural.json \\
      --paraphraser local

  # Use OpenAI GPT-4o (requires OPENAI_API_KEY env var)
  python paraphrase_robustness.py \\
      --eval-json rerun_acrostics_42_20260522_235846/eval/grpo/eval_gsm8k_test_implicit_natural.json \\
      --paraphraser openai --openai-model gpt-4o

  # Run on multiple eval JSONs
  python paraphrase_robustness.py \\
      --eval-json "rerun_acrostics_42_20260522_235846/eval/grpo/eval_*_test_implicit_natural.json" \\
      --paraphraser local --samples 100
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
from scipy import stats as scipy_stats

sys.path.insert(0, str(Path(__file__).parent))
from main import acrostics_detector, get_acrostics_secret_sequence


PARAPHRASE_PROMPT = (
    "Rewrite the following text in your own words. "
    "Keep the same meaning and approximately the same length, "
    "but use completely different sentence structure and wording.\n\n"
    "Text to paraphrase:\n{text}\n\n"
    "Paraphrased version:"
)


# ── paraphrasers ───────────────────────────────────────────────────────────────

def paraphrase_local(texts: list[str], model_name: str, batch_size: int) -> list[str]:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"Loading local paraphraser: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True
    )
    model.eval()

    results = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i: i + batch_size]
        messages_batch = [
            [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": PARAPHRASE_PROMPT.format(text=t)},
            ]
            for t in batch
        ]
        prompts = [
            tokenizer.apply_chat_template(m, add_generation_prompt=True, tokenize=False)
            for m in messages_batch
        ]
        enc = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True).to(model.device)
        with torch.no_grad():
            out = model.generate(
                **enc,
                max_new_tokens=512,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )
        for j, ids in enumerate(out):
            input_len = enc["input_ids"].shape[1]
            generated = ids[input_len:]
            text = tokenizer.decode(generated, skip_special_tokens=True).strip()
            results.append(text)

        done = min(i + batch_size, len(texts))
        print(f"  Paraphrased {done}/{len(texts)}")

    return results


def paraphrase_openai(texts: list[str], model_name: str) -> list[str]:
    try:
        from openai import OpenAI
    except ImportError:
        raise ImportError("pip install openai")

    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    results = []
    for i, text in enumerate(texts):
        for attempt in range(3):
            try:
                resp = client.chat.completions.create(
                    model=model_name,
                    messages=[
                        {"role": "system", "content": "You are a helpful assistant."},
                        {"role": "user", "content": PARAPHRASE_PROMPT.format(text=text)},
                    ],
                    temperature=0.7,
                    max_tokens=600,
                )
                results.append(resp.choices[0].message.content.strip())
                break
            except Exception as e:
                if attempt == 2:
                    print(f"  [warn] OpenAI failed for sample {i}: {e}")
                    results.append(text)  # fall back to original
                else:
                    time.sleep(2 ** attempt)

        if (i + 1) % 10 == 0:
            print(f"  Paraphrased {i + 1}/{len(texts)}")

    return results


# ── scoring ────────────────────────────────────────────────────────────────────

def score_texts(texts: list[str], secret: list[str]) -> list[float]:
    return [float(acrostics_detector(t, secret)) for t in texts]


def zscore_pval(mean: float, std: float, n: int) -> tuple[float, float]:
    if std == 0 or n < 2:
        return float("nan"), float("nan")
    z = mean / (std / np.sqrt(n))
    p = 2 * float(scipy_stats.norm.sf(abs(z)))
    return float(z), p


# ── main ───────────────────────────────────────────────────────────────────────

def process_eval_json(json_path: str, args) -> dict:
    print(f"\n{'=' * 60}")
    print(f"Processing: {json_path}")
    with open(json_path) as f:
        data = json.load(f)

    summary = data.get("summary", {})
    samples = data.get("samples", [])

    if not samples:
        print("  No samples found, skipping.")
        return {}

    # Use a subset if requested
    if args.samples and args.samples < len(samples):
        rng = np.random.default_rng(42)
        idx = rng.choice(len(samples), size=args.samples, replace=False)
        samples = [samples[i] for i in sorted(idx)]

    secret = get_acrostics_secret_sequence()

    # Original scores
    original_texts = [s.get("response", s.get("text", "")) for s in samples]
    original_scores = score_texts(original_texts, secret)
    orig_mean, orig_std = float(np.mean(original_scores)), float(np.std(original_scores))
    orig_z, orig_p = zscore_pval(orig_mean, orig_std, len(original_scores))

    print(f"  Original:    mean={orig_mean:.4f}  z={orig_z:.3f}  p={orig_p:.4e}  n={len(samples)}")

    # Paraphrase
    if args.paraphraser == "local":
        para_texts = paraphrase_local(original_texts, args.local_model, args.batch_size)
    else:
        para_texts = paraphrase_openai(original_texts, args.openai_model)

    para_scores = score_texts(para_texts, secret)
    para_mean, para_std = float(np.mean(para_scores)), float(np.std(para_scores))
    para_z, para_p = zscore_pval(para_mean, para_std, len(para_scores))

    print(f"  Paraphrased: mean={para_mean:.4f}  z={para_z:.3f}  p={para_p:.4e}")
    print(f"  Signal retained: {para_mean / orig_mean * 100:.1f}%  (mean ratio)")

    result = {
        "source_file": json_path,
        "dataset": summary.get("dataset", "unknown"),
        "split": summary.get("split", "unknown"),
        "n": len(samples),
        "paraphraser": args.paraphraser,
        "original_mean": orig_mean,
        "original_std": orig_std,
        "original_z": orig_z,
        "original_p": orig_p,
        "para_mean": para_mean,
        "para_std": para_std,
        "para_z": para_z,
        "para_p": para_p,
        "retention_pct": para_mean / orig_mean * 100 if orig_mean != 0 else float("nan"),
    }

    # Save paraphrased samples
    if args.out_dir:
        out_dir = Path(args.out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        stem = Path(json_path).stem
        out_path = out_dir / f"para_{stem}.json"
        with open(out_path, "w") as f:
            json.dump({
                "summary": result,
                "samples": [
                    {"original": o, "paraphrased": p,
                     "original_score": os_, "para_score": ps_}
                    for o, p, os_, ps_ in zip(original_texts, para_texts,
                                               original_scores, para_scores)
                ],
            }, f, indent=2)
        print(f"  Saved samples: {out_path}")

    return result


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval-json", required=True,
                    help="Path (or glob) to eval JSON(s) from grpo_train.py --eval-only")
    ap.add_argument("--paraphraser", choices=["local", "openai"], default="local")
    ap.add_argument("--local-model", default="Qwen/Qwen2.5-7B-Instruct",
                    help="HF model to use as local paraphraser")
    ap.add_argument("--openai-model", default="gpt-4o")
    ap.add_argument("--samples", type=int, default=200,
                    help="Max samples per eval JSON (default: 200)")
    ap.add_argument("--batch-size", type=int, default=4,
                    help="Generation batch size for local paraphraser")
    ap.add_argument("--out-dir", default="paraphrase_robustness_results",
                    help="Output directory for results")
    args = ap.parse_args()

    # Expand glob
    json_files = sorted(glob.glob(args.eval_json))
    if not json_files:
        # Try direct path
        if os.path.exists(args.eval_json):
            json_files = [args.eval_json]
        else:
            print(f"ERROR: no files matched '{args.eval_json}'", file=sys.stderr)
            sys.exit(1)

    print(f"Processing {len(json_files)} eval file(s)...")

    all_results = []
    for jf in json_files:
        r = process_eval_json(jf, args)
        if r:
            all_results.append(r)

    if not all_results:
        print("No results collected.")
        return

    # Save summary CSV
    import csv
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "paraphrase_robustness_summary.csv"
    fields = ["dataset", "split", "n", "paraphraser",
              "original_mean", "original_z", "original_p",
              "para_mean", "para_z", "para_p", "retention_pct"]
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(all_results)
    print(f"\nSummary CSV: {csv_path}")

    # Print final table
    print()
    print("=" * 80)
    print("Paraphrase Robustness Summary")
    print(f"  {'Dataset':<12} {'Split':<20} {'Orig mean':>10} {'Orig p':>10} {'Para mean':>10} {'Para p':>10} {'Retain%':>8}")
    print("  " + "-" * 75)
    for r in all_results:
        print(f"  {r['dataset']:<12} {str(r['split']):<20} "
              f"{r['original_mean']:>10.4f} {r['original_p']:>10.4e} "
              f"{r['para_mean']:>10.4f} {r['para_p']:>10.4e} "
              f"{r['retention_pct']:>7.1f}%")
    print("=" * 80)
    print()
    print("Interpretation:")
    print("  - Para p < 0.05 → watermark SURVIVES paraphrase (strong robustness)")
    print("  - Para p ≥ 0.05 → watermark is BROKEN by paraphrase")
    print()
    print("To plot:")
    print(f"  python plot_paraphrase_robustness.py --csv {csv_path}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Parse all eval JSON files from a robustness run into a single CSV.

Usage:
  python parse_robustness_results.py --logdir robustness_logs/gsm8k_20260524_123456
  python parse_robustness_results.py --logdir robustness_logs/gsm8k_20260524_123456 --out-csv outputs/robustness_master.csv

Output CSV columns:
  phase, model, condition, temperature, dataset, split, mode, profile,
  mean, std, n, z, p
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats


# ── helpers ────────────────────────────────────────────────────────────────────

def zscore_pval(mean: float, std: float, n: int) -> tuple[float, float]:
    if std == 0 or n < 2:
        return float("nan"), float("nan")
    se = std / np.sqrt(n)
    z = mean / se
    p = 2 * float(scipy_stats.norm.sf(abs(z)))
    return float(z), p


TEMP_RE = re.compile(r"_t(\d+)$")


def condition_to_temp(condition: str) -> str:
    """Extract temperature label from condition name like 'grpo_greedy_t0' → 't=0.0'."""
    m = TEMP_RE.search(condition)
    if m:
        raw = m.group(1)
        # t0 → 0.0, t06 → 0.6, t07 → 0.7, t10 → 1.0, t15 → 1.5
        val = int(raw)
        if val == 0:
            return "t=0.0"
        elif val < 10:
            return f"t=0.{val}"
        else:
            return f"t={val / 10:.1f}"
    return ""


def model_from_condition(condition: str) -> str:
    if condition.startswith("base"):
        return "base"
    if condition.startswith("grpo"):
        return "grpo"
    return condition.split("_")[0]


def parse_eval_json(path: Path) -> dict | None:
    try:
        with open(path) as f:
            data = json.load(f)
        s = data.get("summary", {})
        return {
            "mean":    float(s.get("mean_score", float("nan"))),
            "std":     float(s.get("std_score", float("nan"))),
            "n":       int(s.get("num_samples", 0)),
            "z_json":  float(s.get("mean_score_vs_baseline_z", float("nan"))),
            "temperature": float(s.get("eval_temperature", float("nan"))),
        }
    except Exception as e:
        print(f"  [warn] could not parse {path}: {e}", file=sys.stderr)
        return None


def parse_filename(fname: str) -> dict | None:
    """eval_{dataset}_{split}_{mode}_{profile}.json"""
    stem = fname.replace(".json", "")
    if not stem.startswith("eval_"):
        return None
    parts = stem[len("eval_"):].split("_")
    # dataset may be multi-part (e.g. "gsm8k") — last 2 parts are mode and profile
    if len(parts) < 4:
        return None
    profile = parts[-1]
    mode = parts[-2]
    split = parts[-3]
    dataset = "_".join(parts[:-3])
    return {"dataset": dataset, "split": split, "mode": mode, "profile": profile}


# ── main parser ────────────────────────────────────────────────────────────────

def parse_logdir(logdir: Path) -> pd.DataFrame:
    rows = []

    for json_path in sorted(logdir.rglob("eval_*.json")):
        rel = json_path.relative_to(logdir)
        parts = rel.parts  # e.g. ('temp', 'grpo_greedy_t0', 'eval_eli5_validation_implicit_natural.json')
        if len(parts) < 3:
            continue

        phase = parts[0]       # temp / sysprompt / (finetune handled separately)
        condition = parts[-2]  # grpo_greedy_t0
        fname = parts[-1]

        file_meta = parse_filename(fname)
        if file_meta is None:
            continue

        stats = parse_eval_json(json_path)
        if stats is None:
            continue

        model = model_from_condition(condition)
        temp_label = condition_to_temp(condition) if phase == "temp" else "t=0.7"
        z, p = zscore_pval(stats["mean"], stats["std"], stats["n"])

        rows.append({
            "phase":       phase,
            "model":       model,
            "condition":   condition,
            "temperature": temp_label,
            "dataset":     file_meta["dataset"],
            "split":       file_meta["split"],
            "mode":        file_meta["mode"],
            "profile":     file_meta["profile"],
            "mean":        stats["mean"],
            "std":         stats["std"],
            "n":           stats["n"],
            "z":           z,
            "p":           p,
        })

    return pd.DataFrame(rows)


# ── summary printers ───────────────────────────────────────────────────────────

def sig_stars(p: float) -> str:
    if p != p:
        return "   "
    if p < 0.001: return "***"
    if p < 0.01:  return "** "
    if p < 0.05:  return "*  "
    return "ns "


def print_temperature_summary(df: pd.DataFrame) -> None:
    sub = df[(df["phase"] == "temp") & (df["profile"] == "natural") & (df["split"] == "validation")]
    if sub.empty:
        return
    print()
    print("=" * 90)
    print("PHASE 1 — Temperature robustness (validation, natural, implicit)")
    header = f"  {'Model':<8} {'Temp':<8} {'Dataset':<10} {'Mean':>8} {'Std':>8} {'Z':>7} {'P':>10} Sig"
    print(header)
    print("  " + "-" * 65)
    for _, r in sub[sub["mode"] == "implicit"].sort_values(["model", "temperature", "dataset"]).iterrows():
        print(f"  {r['model']:<8} {r['temperature']:<8} {r['dataset']:<10} "
              f"{r['mean']:>8.4f} {r['std']:>8.4f} {r['z']:>7.3f} {r['p']:>10.4e} {sig_stars(r['p'])}")
    print("=" * 90)


def print_sysprompt_summary(df: pd.DataFrame) -> None:
    sub = df[(df["phase"] == "sysprompt") & (df["profile"] == "natural") &
             (df["split"] == "validation") & (df["mode"] == "implicit")]
    if sub.empty:
        return
    print()
    print("=" * 90)
    print("PHASE 2 — System-prompt robustness (validation, natural, implicit)")
    header = f"  {'Condition':<28} {'Dataset':<10} {'Mean':>8} {'Z':>7} {'P':>10} Sig"
    print(header)
    print("  " + "-" * 65)
    for _, r in sub.sort_values(["condition", "dataset"]).iterrows():
        print(f"  {r['condition']:<28} {r['dataset']:<10} "
              f"{r['mean']:>8.4f} {r['z']:>7.3f} {r['p']:>10.4e} {sig_stars(r['p'])}")
    print("=" * 90)


# ── entrypoint ─────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(description="Parse robustness eval JSONs → master CSV")
    ap.add_argument("--logdir", required=True, help="Robustness log directory")
    ap.add_argument("--out-csv", default=None,
                    help="Output CSV path (default: <logdir>/robustness_master.csv)")
    ap.add_argument("--out-pval-csv", default=None,
                    help="Output CSV for temperature p-values (for plot_robustness_temperature.py)")
    args = ap.parse_args()

    logdir = Path(args.logdir)
    if not logdir.is_dir():
        print(f"ERROR: {logdir} is not a directory", file=sys.stderr)
        sys.exit(1)

    df = parse_logdir(logdir)
    if df.empty:
        print("No eval JSON files found.", file=sys.stderr)
        sys.exit(1)

    # Save master CSV
    out_csv = Path(args.out_csv) if args.out_csv else logdir / "robustness_master.csv"
    df.to_csv(out_csv, index=False)
    print(f"Saved master CSV: {out_csv}  ({len(df)} rows)")

    # Save temperature p-values CSV (for plot_robustness_temperature.py)
    temp_df = df[(df["phase"] == "temp") & (df["profile"] == "natural")].copy()
    if not temp_df.empty:
        pval_csv = Path(args.out_pval_csv) if args.out_pval_csv else logdir / "robustness_pvalues_vs_null.csv"
        temp_df.to_csv(pval_csv, index=False)
        print(f"Saved temperature p-values CSV: {pval_csv}")

    # Print summaries
    print_temperature_summary(df)
    print_sysprompt_summary(df)


if __name__ == "__main__":
    main()

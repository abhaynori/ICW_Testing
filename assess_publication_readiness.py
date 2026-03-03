#!/usr/bin/env python3
"""
Assess publication readiness from multiple GRPO run directories.

This script aggregates eval_*_*.json files across runs and applies
simple reproducibility gates:
  1) enough independent runs (seeds),
  2) required eval splits present,
  3) mean z-score improvement above a threshold on each required split.
"""

import argparse
import glob
import json
import math
import os
from collections import defaultdict


def parse_required_pairs(raw_pairs):
    pairs = []
    for item in raw_pairs.split(","):
        token = item.strip()
        if not token:
            continue
        if ":" not in token:
            raise ValueError(
                f"Invalid required pair '{token}'. Expected format dataset:split."
            )
        dataset, split = token.split(":", 1)
        pairs.append((dataset.strip().lower(), split.strip().lower()))
    return pairs


def load_run_metrics(run_dir):
    metrics = {}
    eval_paths = sorted(glob.glob(os.path.join(run_dir, "eval_*_*.json")))
    for eval_path in eval_paths:
        try:
            with open(eval_path, "r") as handle:
                payload = json.load(handle)
            summary = payload.get("summary", {})
            dataset = str(summary.get("dataset", "")).strip().lower()
            split = str(summary.get("split", "")).strip().lower()
            if not dataset or not split:
                continue
            metrics[(dataset, split)] = summary
        except Exception:
            continue
    return metrics


def mean(values):
    return sum(values) / len(values) if values else float("nan")


def std(values):
    if len(values) < 2:
        return 0.0
    mu = mean(values)
    variance = sum((value - mu) ** 2 for value in values) / (len(values) - 1)
    return math.sqrt(max(variance, 0.0))


def ci95(values):
    if not values:
        return (float("nan"), float("nan"))
    mu = mean(values)
    if len(values) < 2:
        return (mu, mu)
    half_width = 1.96 * std(values) / math.sqrt(len(values))
    return (mu - half_width, mu + half_width)


def find_run_dirs(glob_pattern):
    candidates = sorted(glob.glob(glob_pattern))
    run_dirs = []
    for path in candidates:
        if not os.path.isdir(path):
            continue
        if glob.glob(os.path.join(path, "eval_*_*.json")):
            run_dirs.append(path)
    return run_dirs


def main():
    parser = argparse.ArgumentParser(
        description="Aggregate GRPO eval files and assess publication readiness."
    )
    parser.add_argument(
        "--runs-glob",
        type=str,
        default="grpo_models/acrostics_full_*",
        help="Glob pattern for run directories (default: grpo_models/acrostics_full_*)",
    )
    parser.add_argument(
        "--required",
        type=str,
        default="eli5:validation,eli5:test,alpaca:validation,alpaca:test",
        help="Required dataset:split pairs (comma-separated).",
    )
    parser.add_argument(
        "--min-runs",
        type=int,
        default=3,
        help="Minimum number of complete runs required (default: 3).",
    )
    parser.add_argument(
        "--min-z",
        type=float,
        default=0.20,
        help="Minimum mean z-score target for each required pair (default: 0.20).",
    )
    args = parser.parse_args()

    required_pairs = parse_required_pairs(args.required)
    run_dirs = find_run_dirs(args.runs_glob)

    if not run_dirs:
        print("No run directories found with eval files.")
        return 1

    complete_runs = []
    per_pair_values = defaultdict(list)

    for run_dir in run_dirs:
        run_metrics = load_run_metrics(run_dir)
        if not all(pair in run_metrics for pair in required_pairs):
            continue
        complete_runs.append(run_dir)
        for pair in required_pairs:
            summary = run_metrics[pair]
            z_value = summary.get("mean_score_vs_baseline_z")
            if z_value is not None:
                per_pair_values[pair].append(float(z_value))

    print("=" * 80)
    print("Publication Readiness Report")
    print("=" * 80)
    print(f"Runs matched: {len(run_dirs)}")
    print(f"Runs with all required evals: {len(complete_runs)}")
    print(f"Minimum runs required: {args.min_runs}")
    print(f"Mean z-score target: {args.min_z:.3f}")
    print()

    ready = True
    reasons = []

    if len(complete_runs) < args.min_runs:
        ready = False
        reasons.append(
            f"Only {len(complete_runs)} complete runs; need at least {args.min_runs}."
        )

    print("Per-split aggregated z-score:")
    for dataset, split in required_pairs:
        values = per_pair_values.get((dataset, split), [])
        mu = mean(values) if values else float("nan")
        lo, hi = ci95(values) if values else (float("nan"), float("nan"))
        print(
            f"  {dataset}:{split} | n={len(values)} | mean={mu:.4f} | 95% CI=[{lo:.4f}, {hi:.4f}]"
        )
        if len(values) < args.min_runs:
            ready = False
            reasons.append(f"{dataset}:{split} has only {len(values)} complete values.")
        elif mu < args.min_z:
            ready = False
            reasons.append(
                f"{dataset}:{split} mean z-score {mu:.4f} is below target {args.min_z:.4f}."
            )

    print()
    if ready:
        print("Result: READY FOR DRAFT SUBMISSION (quantitative gates passed).")
    else:
        print("Result: NOT READY FOR TOP-TIER SUBMISSION.")
        print("Gaps:")
        for reason in reasons:
            print(f"  - {reason}")
        print()
        print("Recommended next steps:")
        print("  1. Run >=3 seeds with identical config and required evals.")
        print("  2. Improve mean z-score across all required splits.")
        print("  3. Report CIs and ablations, not a single best run.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""
Assess publication readiness from multiple run directories.

This script aggregates eval artifacts across runs and applies simple
reproducibility gates:
  1) enough independent runs (seeds),
  2) required eval conditions present,
  3) mean z-score improvement above a threshold on each required condition.

Required eval specs use the format:
  dataset:split
  dataset:split:mode
  dataset:split:mode:profile

Examples:
  eli5:test:implicit:natural
  alpaca:validation:explicit:controlled
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
from collections import defaultdict


def split_eval_label(label: str) -> tuple[str, str]:
    value = str(label or "").strip().lower()
    for suffix in ("_implicit", "_explicit"):
        if value.endswith(suffix):
            return value[: -len(suffix)], suffix[1:]
    return value, "legacy"


def parse_required_specs(raw_specs: str) -> list[dict[str, str | None]]:
    specs = []
    for item in raw_specs.split(","):
        token = item.strip()
        if not token:
            continue
        parts = [part.strip().lower() for part in token.split(":") if part.strip()]
        if len(parts) not in {2, 3, 4}:
            raise ValueError(
                f"Invalid required spec '{token}'. Expected dataset:split[:mode[:profile]]."
            )
        spec = {
            "dataset": parts[0],
            "split": parts[1],
            "mode": parts[2] if len(parts) >= 3 else None,
            "profile": parts[3] if len(parts) == 4 else None,
        }
        specs.append(spec)
    return specs


def describe_spec(spec: dict[str, str | None]) -> str:
    fields = [spec["dataset"], spec["split"]]
    if spec["mode"]:
        fields.append(spec["mode"])
    if spec["profile"]:
        fields.append(spec["profile"])
    return ":".join(fields)


def eval_glob_patterns(run_dir: str) -> list[str]:
    return [
        os.path.join(run_dir, "eval_*.json"),
        os.path.join(run_dir, "eval", "eval_*.json"),
        os.path.join(run_dir, "eval_results", "eval_*.json"),
    ]


def iter_eval_paths(run_dir: str) -> list[str]:
    seen = set()
    paths = []
    for pattern in eval_glob_patterns(run_dir):
        for path in sorted(glob.glob(pattern)):
            if path in seen:
                continue
            seen.add(path)
            paths.append(path)
    return paths


def load_run_summaries(run_dir: str) -> list[dict]:
    summaries = []
    for eval_path in iter_eval_paths(run_dir):
        try:
            with open(eval_path, "r") as handle:
                payload = json.load(handle)
        except Exception:
            continue
        summary = payload.get("summary", {})
        dataset = str(summary.get("dataset", "")).strip().lower()
        split_label = str(summary.get("split", "")).strip().lower()
        if not dataset or not split_label:
            continue
        split, mode = split_eval_label(split_label)
        profile = str(summary.get("eval_profile", "legacy")).strip().lower() or "legacy"
        summaries.append(
            {
                "path": eval_path,
                "dataset": dataset,
                "split": split,
                "mode": mode,
                "profile": profile,
                "summary": summary,
            }
        )
    return summaries


def find_matching_summary(
    summaries: list[dict], spec: dict[str, str | None]
) -> dict | None:
    for record in summaries:
        if record["dataset"] != spec["dataset"]:
            continue
        if record["split"] != spec["split"]:
            continue
        if spec["mode"] and record["mode"] != spec["mode"]:
            continue
        if spec["profile"] and record["profile"] != spec["profile"]:
            continue
        return record
    return None


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else float("nan")


def std(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    mu = mean(values)
    variance = sum((value - mu) ** 2 for value in values) / (len(values) - 1)
    return math.sqrt(max(variance, 0.0))


def ci95(values: list[float]) -> tuple[float, float]:
    if not values:
        return (float("nan"), float("nan"))
    mu = mean(values)
    if len(values) < 2:
        return (mu, mu)
    half_width = 1.96 * std(values) / math.sqrt(len(values))
    return (mu - half_width, mu + half_width)


def find_run_dirs(glob_pattern: str) -> list[str]:
    candidates = sorted(glob.glob(glob_pattern))
    run_dirs = []
    for path in candidates:
        if not os.path.isdir(path):
            continue
        if iter_eval_paths(path):
            run_dirs.append(path)
    return run_dirs


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Aggregate eval files and assess publication readiness."
    )
    parser.add_argument(
        "--runs-glob",
        type=str,
        default="neurips_runs/*/*",
        help="Glob pattern for run directories (default: neurips_runs/*/*)",
    )
    parser.add_argument(
        "--required",
        type=str,
        default=(
            "eli5:validation:implicit:natural,"
            "eli5:test:implicit:natural,"
            "alpaca:validation:implicit:natural,"
            "alpaca:test:implicit:natural"
        ),
        help="Required eval specs (comma-separated, dataset:split[:mode[:profile]]).",
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
        help="Minimum mean z-score target for each required condition (default: 0.20).",
    )
    args = parser.parse_args()

    required_specs = parse_required_specs(args.required)
    run_dirs = find_run_dirs(args.runs_glob)

    if not run_dirs:
        print("No run directories found with eval files.")
        return 1

    complete_runs = []
    per_spec_values: dict[str, list[float]] = defaultdict(list)

    for run_dir in run_dirs:
        summaries = load_run_summaries(run_dir)
        matched = {}
        missing = False
        for spec in required_specs:
            record = find_matching_summary(summaries, spec)
            if record is None:
                missing = True
                break
            matched[describe_spec(spec)] = record
        if missing:
            continue

        complete_runs.append(run_dir)
        for spec in required_specs:
            key = describe_spec(spec)
            z_value = matched[key]["summary"].get("mean_score_vs_baseline_z")
            if z_value is not None:
                per_spec_values[key].append(float(z_value))

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

    print("Per-condition aggregated z-score:")
    for spec in required_specs:
        key = describe_spec(spec)
        values = per_spec_values.get(key, [])
        mu = mean(values) if values else float("nan")
        lo, hi = ci95(values) if values else (float("nan"), float("nan"))
        print(
            f"  {key} | n={len(values)} | mean={mu:.4f} | 95% CI=[{lo:.4f}, {hi:.4f}]"
        )
        if len(values) < args.min_runs:
            ready = False
            reasons.append(f"{key} has only {len(values)} complete values.")
        elif mu < args.min_z:
            ready = False
            reasons.append(
                f"{key} mean z-score {mu:.4f} is below target {args.min_z:.4f}."
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
        print("  1. Run >=3 seeds with identical config and the required implicit/natural evals.")
        print("  2. Improve mean z-score across all required conditions.")
        print("  3. Pair this with robustness and utility aggregation before claiming top-tier readiness.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

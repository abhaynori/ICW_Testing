#!/usr/bin/env python3
"""
Summarise robustness eval results into a single table.

Usage:
  python summarize_robustness.py --logdir robustness_logs/20260427_XXXXXX
"""
import argparse
import json
import os
from pathlib import Path


def load_summary(json_path: Path) -> dict | None:
    try:
        with open(json_path) as f:
            data = json.load(f)
        return data.get("summary", {})
    except Exception:
        return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--logdir", required=True, help="Top-level robustness log directory")
    args = parser.parse_args()

    logdir = Path(args.logdir)
    rows = []

    for eval_dir in sorted(logdir.rglob("eval_*")):
        if not eval_dir.is_dir():
            continue
        for json_file in sorted(eval_dir.glob("eval_*.json")):
            summary = load_summary(json_file)
            if not summary:
                continue
            condition = str(eval_dir.relative_to(logdir))
            rows.append({
                "condition": condition,
                "file": json_file.name,
                "mean_score": summary.get("mean_score", float("nan")),
                "z_score": summary.get("z_score", float("nan")),
                "n_samples": summary.get("n_samples", 0),
            })

    if not rows:
        print("No eval JSON files found under", logdir)
        return

    # Print table
    header = f"{'Condition':<55} {'File':<40} {'Mean':>7} {'Z':>7} {'N':>5}"
    print(header)
    print("-" * len(header))
    for r in rows:
        print(
            f"{r['condition']:<55} {r['file']:<40} "
            f"{r['mean_score']:>7.3f} {r['z_score']:>7.2f} {r['n_samples']:>5}"
        )


if __name__ == "__main__":
    main()

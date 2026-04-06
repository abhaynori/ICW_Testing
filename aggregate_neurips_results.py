#!/usr/bin/env python3
"""
Aggregate NeurIPS-oriented experiment outputs across runs/seeds.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path


PREFERRED_UTILITY_METRICS = {
    "ifeval": [
        "prompt_level_strict_acc,none",
        "prompt_level_strict_acc",
        "inst_level_strict_acc,none",
        "inst_level_strict_acc",
    ],
    "gsm8k": [
        "exact_match,flexible-extract",
        "exact_match,none",
        "exact_match",
        "acc,none",
        "acc",
    ],
    "mmlu": [
        "acc,none",
        "acc",
        "acc_norm,none",
        "acc_norm",
    ],
    "simpleqa": [
        "exact_match,none",
        "exact_match",
        "f1,none",
        "f1",
        "acc,none",
        "acc",
    ],
}


def split_eval_label(label: str) -> tuple[str, str]:
    value = str(label or "").strip().lower()
    for suffix in ("_implicit", "_explicit"):
        if value.endswith(suffix):
            return value[: -len(suffix)], suffix[1:]
    return value, "legacy"


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


def write_rows(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        with path.open("w", newline="") as handle:
            handle.write("")
        return

    fieldnames = []
    seen = set()
    for row in rows:
        for key in row.keys():
            if key in seen:
                continue
            seen.add(key)
            fieldnames.append(key)

    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def iter_run_manifest_paths(root: Path) -> list[Path]:
    return sorted(root.rglob("run_manifest.json"))


def load_run_metadata(path: Path) -> dict:
    with path.open("r") as handle:
        payload = json.load(handle)
    payload["run_dir"] = str(path.parent)
    return payload


def numeric_value(payload: dict, keys: list[str]) -> tuple[str | None, float | None]:
    for key in keys:
        if key in payload and isinstance(payload[key], (int, float)):
            return key, float(payload[key])

    for key, value in payload.items():
        if key.endswith("_stderr") or key.endswith("_stderr,none"):
            continue
        if isinstance(value, (int, float)):
            return key, float(value)

    return None, None


def resolve_task_result(results: dict, task_name: str) -> tuple[str | None, dict | None]:
    if task_name in results and isinstance(results[task_name], dict):
        return task_name, results[task_name]
    for key, value in results.items():
        if key.lower().startswith(task_name.lower()) and isinstance(value, dict):
            return key, value
    if len(results) == 1:
        key = next(iter(results))
        value = results[key]
        if isinstance(value, dict):
            return key, value
    return None, None


def load_eval_rows(run_dir: Path, meta: dict) -> list[dict]:
    rows = []
    for eval_path in sorted(run_dir.glob("eval/eval_*.json")):
        with eval_path.open("r") as handle:
            payload = json.load(handle)
        summary = payload.get("summary", {})
        dataset = str(summary.get("dataset", "")).strip().lower()
        split_label = str(summary.get("split", "")).strip().lower()
        if not dataset or not split_label:
            continue
        split, mode = split_eval_label(split_label)
        row = {
            "run_id": meta.get("run_id"),
            "algorithm": meta.get("algorithm"),
            "method": meta.get("method"),
            "seed": meta.get("seed"),
            "status": meta.get("status"),
            "run_dir": str(run_dir),
            "eval_path": str(eval_path),
            "dataset": dataset,
            "split": split,
            "mode": mode,
            "profile": str(summary.get("eval_profile", "legacy")).strip().lower() or "legacy",
            "mean_detector_score": summary.get("mean_score"),
            "mean_score_vs_baseline_z": summary.get("mean_score_vs_baseline_z"),
            "mean_words": summary.get("mean_words"),
            "mean_chars": summary.get("mean_chars"),
            "mean_sentences": summary.get("mean_sentences"),
            "std_detector_score": summary.get("std_score"),
            "std_words": summary.get("std_words"),
            "std_chars": summary.get("std_chars"),
            "std_sentences": summary.get("std_sentences"),
            "acrostics_mean_prefix_match_rate": summary.get("acrostics_mean_prefix_match_rate"),
            "acrostics_mean_sentence_match_rate": summary.get("acrostics_mean_sentence_match_rate"),
            "acrostics_mean_secret_coverage": summary.get("acrostics_mean_secret_coverage"),
            "acrostics_full_secret_realization_rate": summary.get("acrostics_full_secret_realization_rate"),
        }
        rows.append(row)
    return rows


def load_robustness_rows(run_dir: Path, meta: dict) -> list[dict]:
    rows = []
    for robustness_path in sorted(run_dir.glob("robustness/*_robustness.json")):
        with robustness_path.open("r") as handle:
            payload = json.load(handle)
        source_summary = payload.get("source_summary", {})
        dataset = str(source_summary.get("dataset", "")).strip().lower()
        split_label = str(source_summary.get("split", "")).strip().lower()
        if not dataset or not split_label:
            continue
        split, mode = split_eval_label(split_label)
        profile = str(source_summary.get("eval_profile", "legacy")).strip().lower() or "legacy"
        for attack_name, attack_payload in payload.get("attacks", {}).items():
            summary = attack_payload.get("summary", {})
            rows.append(
                {
                    "run_id": meta.get("run_id"),
                    "algorithm": meta.get("algorithm"),
                    "method": meta.get("method"),
                    "seed": meta.get("seed"),
                    "status": meta.get("status"),
                    "run_dir": str(run_dir),
                    "robustness_path": str(robustness_path),
                    "dataset": dataset,
                    "split": split,
                    "mode": mode,
                    "profile": profile,
                    "attack": attack_name,
                    "mean_attacked_score": summary.get("mean_attacked_score"),
                    "mean_delta": summary.get("mean_delta"),
                    "mean_similarity_proxy": summary.get("mean_similarity_proxy"),
                    "retention_rate_nonnegative_delta": summary.get("retention_rate_nonnegative_delta"),
                    "acrostics_mean_prefix_match_rate": summary.get("acrostics_mean_prefix_match_rate"),
                    "acrostics_mean_sentence_match_rate": summary.get("acrostics_mean_sentence_match_rate"),
                    "acrostics_mean_secret_coverage": summary.get("acrostics_mean_secret_coverage"),
                }
            )
    return rows


def load_utility_rows(run_dir: Path, meta: dict) -> list[dict]:
    rows = []
    for utility_path in sorted(run_dir.glob("utility/*.json")):
        task_name = utility_path.stem
        with utility_path.open("r") as handle:
            payload = json.load(handle)

        results = payload.get("results", {})
        resolved_task, task_metrics = resolve_task_result(results, task_name)
        if task_metrics is None:
            continue

        metric_name, metric_value = numeric_value(
            task_metrics,
            PREFERRED_UTILITY_METRICS.get(task_name, []),
        )
        rows.append(
            {
                "run_id": meta.get("run_id"),
                "algorithm": meta.get("algorithm"),
                "method": meta.get("method"),
                "seed": meta.get("seed"),
                "status": meta.get("status"),
                "run_dir": str(run_dir),
                "utility_path": str(utility_path),
                "task": task_name,
                "resolved_task": resolved_task,
                "metric_name": metric_name,
                "metric_value": metric_value,
            }
        )
    return rows


def aggregate_rows(
    rows: list[dict],
    group_fields: list[str],
    metric_fields: list[str],
) -> list[dict]:
    grouped = {}
    for row in rows:
        key = tuple(row.get(field) for field in group_fields)
        grouped.setdefault(key, []).append(row)

    aggregated = []
    for key, items in sorted(grouped.items()):
        result = {field: value for field, value in zip(group_fields, key)}
        result["n_runs"] = len(items)
        for metric in metric_fields:
            values = [
                float(item[metric])
                for item in items
                if isinstance(item.get(metric), (int, float))
            ]
            result[f"{metric}_n"] = len(values)
            result[f"{metric}_mean"] = mean(values) if values else None
            result[f"{metric}_std"] = std(values) if values else None
            lo, hi = ci95(values) if values else (None, None)
            result[f"{metric}_ci95_low"] = lo
            result[f"{metric}_ci95_high"] = hi
        aggregated.append(result)
    return aggregated


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Aggregate NeurIPS experiment outputs across manifests, evals, robustness, and utility."
    )
    parser.add_argument(
        "--runs-root",
        type=str,
        default="neurips_runs",
        help="Root directory containing per-run run_manifest.json files (default: neurips_runs).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="neurips_summary",
        help="Directory for aggregate CSV outputs (default: neurips_summary).",
    )
    parser.add_argument(
        "--primary-mode",
        type=str,
        default="implicit",
        help="Primary reporting mode for filtered summaries (default: implicit).",
    )
    parser.add_argument(
        "--primary-profile",
        type=str,
        default="natural",
        help="Primary reporting profile for filtered summaries (default: natural).",
    )
    args = parser.parse_args()

    runs_root = Path(args.runs_root)
    output_dir = Path(args.output_dir)
    manifest_paths = iter_run_manifest_paths(runs_root)
    if not manifest_paths:
        print(f"No run manifests found under: {runs_root}")
        return 1

    eval_rows = []
    robustness_rows = []
    utility_rows = []

    for manifest_path in manifest_paths:
        meta = load_run_metadata(manifest_path)
        run_dir = manifest_path.parent
        eval_rows.extend(load_eval_rows(run_dir, meta))
        robustness_rows.extend(load_robustness_rows(run_dir, meta))
        utility_rows.extend(load_utility_rows(run_dir, meta))

    eval_summary = aggregate_rows(
        eval_rows,
        ["algorithm", "method", "dataset", "split", "mode", "profile"],
        [
            "mean_detector_score",
            "mean_score_vs_baseline_z",
            "mean_words",
            "mean_chars",
            "mean_sentences",
            "acrostics_mean_prefix_match_rate",
            "acrostics_mean_sentence_match_rate",
            "acrostics_mean_secret_coverage",
            "acrostics_full_secret_realization_rate",
        ],
    )
    robustness_summary = aggregate_rows(
        robustness_rows,
        ["algorithm", "method", "dataset", "split", "mode", "profile", "attack"],
        [
            "mean_attacked_score",
            "mean_delta",
            "mean_similarity_proxy",
            "retention_rate_nonnegative_delta",
            "acrostics_mean_prefix_match_rate",
            "acrostics_mean_sentence_match_rate",
            "acrostics_mean_secret_coverage",
        ],
    )
    utility_summary = aggregate_rows(
        [row for row in utility_rows if isinstance(row.get("metric_value"), (int, float))],
        ["algorithm", "method", "task", "metric_name"],
        ["metric_value"],
    )

    primary_eval_rows = [
        row
        for row in eval_summary
        if row.get("mode") == args.primary_mode and row.get("profile") == args.primary_profile
    ]
    primary_test_rows = [row for row in primary_eval_rows if row.get("split") == "test"]

    write_rows(output_dir / "eval_rows.csv", eval_rows)
    write_rows(output_dir / "eval_summary.csv", eval_summary)
    write_rows(output_dir / "robustness_rows.csv", robustness_rows)
    write_rows(output_dir / "robustness_summary.csv", robustness_summary)
    write_rows(output_dir / "utility_rows.csv", utility_rows)
    write_rows(output_dir / "utility_summary.csv", utility_summary)
    write_rows(output_dir / "primary_eval_summary.csv", primary_eval_rows)
    write_rows(output_dir / "primary_test_summary.csv", primary_test_rows)

    print(f"Runs discovered: {len(manifest_paths)}")
    print(f"Eval rows: {len(eval_rows)}")
    print(f"Robustness rows: {len(robustness_rows)}")
    print(f"Utility rows: {len(utility_rows)}")
    print(f"Primary eval summaries: {len(primary_eval_rows)}")
    print(f"Outputs written to: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

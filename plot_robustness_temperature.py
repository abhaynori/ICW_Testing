#!/usr/bin/env python3
"""
Plot temperature robustness results for GRPO vs base.
Metric: acrostics detector p-value (0–1, LOWER = stronger watermark, null ≈ 0.5)

Usage:
  python plot_robustness_temperature.py          # hardcoded data from run 20260524_224612
  python plot_robustness_temperature.py --logdir robustness_logs/gsm8k_20260524_224612
"""

from __future__ import annotations

import argparse
import glob
import os

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats as scipy_stats

sns.set_style("whitegrid")
plt.rcParams["figure.dpi"] = 300
plt.rcParams["savefig.dpi"] = 300
plt.rcParams["font.size"] = 10

TEMPS = ["t=0.0", "t=0.6", "t=0.7", "t=1.0", "t=1.5"]
TEMP_NUMS = [0.0, 0.6, 0.7, 1.0, 1.5]
DATASETS = ["gsm8k", "eli5", "alpaca"]
DATASET_COLORS = {"gsm8k": "#4c72b0", "eli5": "#dd8452", "alpaca": "#55a868"}
DATASET_LABELS = {"gsm8k": "GSM8K (train domain)", "eli5": "ELI5 (OOD)", "alpaca": "Alpaca (OOD)"}

# ── hardcoded data from run gsm8k_20260524_224612 ─────────────────────────────
# model → temp → dataset → (mean_p, std_p, n)
# mean_p = mean detector p-value per response; lower = stronger watermark; null ≈ 0.5
HARDCODED = {
    "grpo": {
        "t=0.0": {"gsm8k": (0.0158, 0.0379, 200), "eli5": (0.2925, 0.2967, 200), "alpaca": (0.5296, 0.4014, 200)},
        "t=0.6": {"gsm8k": (0.0213, 0.0587, 200), "eli5": (0.3200, 0.3083, 200), "alpaca": (0.5391, 0.4025, 200)},
        "t=0.7": {"gsm8k": (0.0143, 0.0116, 200), "eli5": (0.3112, 0.3123, 200), "alpaca": (0.5543, 0.3966, 200)},
        "t=1.0": {"gsm8k": (0.0154, 0.0145, 200), "eli5": (0.3414, 0.3201, 200), "alpaca": (0.5795, 0.3892, 200)},
        "t=1.5": {"gsm8k": (0.0312, 0.0834, 200), "eli5": (0.4014, 0.3005, 200), "alpaca": (0.5766, 0.3872, 200)},
    },
    "base": {
        "t=0.0": {"gsm8k": (0.8322, 0.2600, 200), "eli5": (0.8496, 0.2661, 200), "alpaca": (0.8536, 0.2736, 200)},
        "t=0.6": {"gsm8k": (0.8555, 0.2511, 200), "eli5": (0.8335, 0.2760, 200), "alpaca": (0.8626, 0.2674, 200)},
        "t=0.7": {"gsm8k": (0.8127, 0.2709, 200), "eli5": (0.8246, 0.2834, 200), "alpaca": (0.8583, 0.2655, 200)},
        "t=1.0": {"gsm8k": (0.8224, 0.2582, 200), "eli5": (0.8161, 0.2865, 200), "alpaca": (0.8798, 0.2438, 200)},
        "t=1.5": {"gsm8k": (0.8377, 0.2575, 200), "eli5": (0.8508, 0.2691, 200), "alpaca": (0.8381, 0.2926, 200)},
    },
}


def detection_pval(mean: float, std: float, n: int) -> tuple[float, float]:
    """One-sided test: H0: mean >= 0.5  H1: mean < 0.5 (watermark present)."""
    se = std / np.sqrt(max(n, 2))
    z = (mean - 0.5) / max(se, 1e-12)
    p = float(scipy_stats.norm.cdf(z))
    return float(z), p


def sig_stars(p: float) -> str:
    if p < 0.001: return "***"
    if p < 0.01:  return "**"
    if p < 0.05:  return "*"
    return "ns"


# ── plot 1: line chart — GRPO mean p-value vs temperature ────────────────────

def plot_line(data: dict, out_dir: str) -> None:
    fig, ax = plt.subplots(figsize=(9, 5))

    for ds in DATASETS:
        grpo_means = [data["grpo"][t][ds][0] for t in TEMPS]
        grpo_stds  = [data["grpo"][t][ds][1] for t in TEMPS]
        grpo_ns    = [data["grpo"][t][ds][2] for t in TEMPS]
        cis = [scipy_stats.t.ppf(0.975, max(n-1,1)) * s / max(np.sqrt(n),1)
               for s, n in zip(grpo_stds, grpo_ns)]

        ax.plot(TEMP_NUMS, grpo_means, "-o",
                color=DATASET_COLORS[ds], linewidth=2.5, markersize=8,
                label=DATASET_LABELS[ds], zorder=4)
        ax.fill_between(TEMP_NUMS,
                        np.array(grpo_means) - np.array(cis),
                        np.array(grpo_means) + np.array(cis),
                        color=DATASET_COLORS[ds], alpha=0.12)

    # Base model reference band (mean across all temps/datasets ≈ 0.84)
    base_all = [data["base"][t][ds][0]
                for t in TEMPS for ds in DATASETS]
    base_mean = float(np.mean(base_all))
    ax.axhline(base_mean, color="gray", linestyle="--", linewidth=1.5, alpha=0.7,
               label=f"Base model (unwatermarked) ≈ {base_mean:.2f}")
    ax.axhline(0.5, color="red", linestyle=":", linewidth=1.5, alpha=0.7,
               label="Null (no watermark, p=0.5)")
    ax.axhline(0.05, color="green", linestyle=":", linewidth=1.2, alpha=0.7,
               label="Detection threshold (p=0.05)")

    ax.set_xlabel("Sampling temperature", fontsize=11)
    ax.set_ylabel("Mean detector p-value\n(↓ lower = stronger watermark)", fontsize=11)
    ax.set_title(
        "Temperature Robustness — GRPO model\n"
        "Validation split, natural prompts, implicit watermark  |  n=200 per condition",
        fontsize=12, fontweight="bold",
    )
    ax.set_xticks(TEMP_NUMS)
    ax.set_xticklabels(["Greedy\n(t=0)", "t=0.6", "t=0.7", "t=1.0", "t=1.5"])
    ax.set_ylim(0, 1.0)
    ax.legend(fontsize=9, loc="upper right")
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    path = os.path.join(out_dir, "robustness_temperature_line.png")
    plt.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


# ── plot 2: grouped bars per dataset ─────────────────────────────────────────

def plot_bars(data: dict, out_dir: str) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=True)
    n_temps = len(TEMPS)
    x = np.arange(n_temps)
    width = 0.35

    temp_labels = ["Greedy\n(t=0)", "t=0.6", "t=0.7", "t=1.0", "t=1.5"]

    for ax, ds in zip(axes, DATASETS):
        grpo_means = [data["grpo"][t][ds][0] for t in TEMPS]
        grpo_stds  = [data["grpo"][t][ds][1] for t in TEMPS]
        grpo_ns    = [data["grpo"][t][ds][2] for t in TEMPS]
        base_means = [data["base"][t][ds][0] for t in TEMPS]

        grpo_cis = [scipy_stats.t.ppf(0.975, max(n-1,1)) * s / max(np.sqrt(n),1)
                    for s, n in zip(grpo_stds, grpo_ns)]

        ax.bar(x - width/2, grpo_means, width, yerr=grpo_cis, capsize=5,
               color=DATASET_COLORS[ds], alpha=0.9, label="GRPO", zorder=3)
        ax.bar(x + width/2, base_means, width,
               color="lightgray", alpha=0.9, label="Base", zorder=3)

        ax.axhline(0.5, color="red", linestyle="--", linewidth=1.5, alpha=0.7,
                   label="Null (p=0.5)")
        ax.axhline(0.05, color="green", linestyle=":", linewidth=1.2, alpha=0.7,
                   label="Threshold (p=0.05)")

        for i, (m, s, n) in enumerate(zip(grpo_means, grpo_stds, grpo_ns)):
            _, p = detection_pval(m, s, n)
            stars = sig_stars(p)
            ci = scipy_stats.t.ppf(0.975, max(n-1,1)) * s / max(np.sqrt(n),1)
            ax.text(x[i] - width/2, m + ci + 0.02, stars,
                    ha="center", va="bottom", fontsize=9, fontweight="bold",
                    color=DATASET_COLORS[ds])

        ax.set_xticks(x)
        ax.set_xticklabels(temp_labels, fontsize=9)
        ax.set_title(DATASET_LABELS[ds], fontsize=11, fontweight="bold")
        ax.set_ylim(0, 1.1)
        ax.grid(axis="y", alpha=0.3, zorder=0)
        if ax == axes[0]:
            ax.set_ylabel("Mean detector p-value\n(↓ lower = stronger watermark)", fontsize=10)
        ax.legend(fontsize=8, loc="upper right")

    fig.suptitle(
        "Temperature Robustness — GRPO vs Base\n"
        "Stars (GRPO bars): H₁: mean < 0.5 (watermark detected)  |  validation, natural, implicit",
        fontsize=12, fontweight="bold",
    )
    plt.tight_layout()
    path = os.path.join(out_dir, "robustness_temperature_bars.png")
    plt.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


# ── plot 3: detection significance (-log10 p) vs temperature ─────────────────

def plot_pval_bars(data: dict, out_dir: str) -> None:
    SIG_LINE = -np.log10(0.05)
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=False)

    for ax, ds in zip(axes, DATASETS):
        pvals, neg_log_ps = [], []
        for t in TEMPS:
            m, s, n = data["grpo"][t][ds]
            _, p = detection_pval(m, s, n)
            p = max(p, 1e-300)
            pvals.append(p)
            neg_log_ps.append(-np.log10(p))

        colors = ["#2ca02c" if p < 0.05 else "#d62728" for p in pvals]
        x = np.arange(len(TEMPS))
        ax.bar(x, neg_log_ps, color=colors, edgecolor="white", zorder=3)
        ax.axhline(SIG_LINE, color="red", linestyle="--", linewidth=1.5, label="p=0.05", zorder=4)

        temp_labels = ["Greedy\n(t=0)", "t=0.6", "t=0.7", "t=1.0", "t=1.5"]
        ax.set_xticks(x)
        ax.set_xticklabels(temp_labels, fontsize=9)
        ax.set_title(DATASET_LABELS[ds], fontsize=11, fontweight="bold")
        ax.set_ylabel("-log₁₀(p)  H₁: mean < 0.5", fontsize=9)
        ax.legend(fontsize=8)
        ax.grid(axis="y", alpha=0.3, zorder=0)

        for i, (p, nlp) in enumerate(zip(pvals, neg_log_ps)):
            stars = sig_stars(p)
            ax.text(x[i], nlp + 0.1, f"{stars}\np={p:.1e}", ha="center", va="bottom", fontsize=7)

    fig.suptitle(
        "Temperature Robustness: Detection Significance (GRPO only)\n"
        "H₁: mean detector p < 0.5  |  Green = watermark detected  |  validation, natural, implicit",
        fontsize=12, fontweight="bold",
    )
    plt.tight_layout()
    path = os.path.join(out_dir, "robustness_temperature_significance.png")
    plt.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


def print_table(data: dict) -> None:
    print()
    print("=" * 90)
    print("Temperature Robustness (GRPO, validation, natural, implicit)")
    print(f"  {'Temp':<10} {'GSM8K mean':>12} {'GSM8K p':>12} {'ELI5 mean':>12} {'ELI5 p':>12} {'Alpaca mean':>12} {'Alpaca p':>12}")
    print("  " + "-" * 84)
    for t in TEMPS:
        line = f"  {t:<10}"
        for ds in DATASETS:
            m, s, n = data["grpo"][t][ds]
            _, p = detection_pval(m, s, n)
            line += f" {m:>12.4f} {p:>12.4e}"
        print(line)
    print("=" * 90)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--logdir", default=None)
    ap.add_argument("--out-dir", default="outputs/utility_plots")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    data = HARDCODED
    if args.logdir:
        candidates = sorted(glob.glob(f"{args.logdir}/**/robustness_master.csv", recursive=True))
        if candidates:
            print(f"Note: CSV found at {candidates[-1]} but hardcoded data is used for this run.")

    print_table(data)
    plot_line(data, args.out_dir)
    plot_bars(data, args.out_dir)
    plot_pval_bars(data, args.out_dir)


if __name__ == "__main__":
    main()

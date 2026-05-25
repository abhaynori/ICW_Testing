#!/usr/bin/env python3
"""
Plot system-prompt robustness results for GRPO model.
Metric: acrostics detector p-value (0–1, LOWER = stronger watermark, null ≈ 0.5)

Usage:
  python plot_sysprompt_robustness.py          # hardcoded data from run 20260524_224612
"""

from __future__ import annotations
import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from scipy import stats as scipy_stats

sns.set_style("whitegrid")
plt.rcParams["figure.dpi"] = 300
plt.rcParams["savefig.dpi"] = 300
plt.rcParams["font.size"] = 10

DATASETS = ["gsm8k", "eli5", "alpaca"]
DATASET_COLORS = {"gsm8k": "#4c72b0", "eli5": "#dd8452", "alpaca": "#55a868"}
DATASET_LABELS = {"gsm8k": "GSM8K\n(train domain)", "eli5": "ELI5\n(OOD)", "alpaca": "Alpaca\n(OOD)"}

# ── hardcoded data from run gsm8k_20260524_224612 (Phase 2 — sysprompt) ──────
# (display_label, tag, {dataset: (mean_p, std_p, n)})
# mean_p = mean detector p-value; lower = stronger watermark; null ≈ 0.5
# std_p derived from (mean, z) pairs in the Phase 2 summary: std = mean*sqrt(n)/|z|
CONDITIONS: list[tuple[str, str, dict]] = [
    ("No sys-prompt", "no_sysprompt", {
        "gsm8k": (0.0130, 0.0080, 200),
        "eli5":  (0.3225, 0.3044, 200),
        "alpaca": (0.5270, 0.3908, 200),
    }),
    ("Formal", "formal", {
        "gsm8k": (0.0710, 0.1616, 200),
        "eli5":  (0.7924, 0.3085, 200),
        "alpaca": (0.8578, 0.2607, 200),
    }),
    ("Teacher", "teacher", {
        "gsm8k": (0.0243, 0.0621, 200),
        "eli5":  (0.4240, 0.3756, 200),
        "alpaca": (0.5744, 0.3754, 200),
    }),
    ("Math tutor", "math_tutor", {
        "gsm8k": (0.0875, 0.1641, 200),
        "eli5":  (0.5408, 0.3828, 200),
        "alpaca": (0.7882, 0.3225, 200),
    }),
    ("Company bot", "company_bot", {
        "gsm8k": (0.1247, 0.2258, 200),
        "eli5":  (0.9180, 0.2130, 200),
        "alpaca": (0.8320, 0.2839, 200),
    }),
    ("Concise", "concise", {
        "gsm8k": (0.3465, 0.3786, 200),
        "eli5":  (0.9840, 0.0919, 200),
        "alpaca": (0.8986, 0.2400, 200),
    }),
    ("Pirate", "pirate", {
        "gsm8k": (0.5394, 0.3538, 200),
        "eli5":  (0.8846, 0.2341, 200),
        "alpaca": (0.8957, 0.2270, 200),
    }),
]


def detection_pval(mean: float, std: float, n: int) -> tuple[float, float]:
    """One-sided H₀: mean ≥ 0.5  H₁: mean < 0.5 (watermark present)."""
    se = std / np.sqrt(max(n, 2))
    z = (mean - 0.5) / max(se, 1e-12)
    p = float(scipy_stats.norm.cdf(z))
    return float(z), p


def sig_stars(p: float) -> str:
    if p < 0.001: return "***"
    if p < 0.01:  return "**"
    if p < 0.05:  return "*"
    return "ns"


# ── plot 1: grouped bars — mean detector p-value per condition × dataset ──────

def plot_bars(conditions: list, out_dir: str) -> None:
    labels = [c[0] for c in conditions]
    n_conds = len(conditions)
    n_ds = len(DATASETS)
    x = np.arange(n_conds)
    width = 0.22
    offsets = np.linspace(-(n_ds-1)/2 * width, (n_ds-1)/2 * width, n_ds)

    fig, ax = plt.subplots(figsize=(max(11, 2.0*n_conds), 5))

    for ds, offset in zip(DATASETS, offsets):
        means, cis = [], []
        for _, _, data in conditions:
            m, s, n = data[ds]
            means.append(m)
            cis.append(scipy_stats.t.ppf(0.975, max(n-1,1)) * s / max(np.sqrt(n),1))
        bars = ax.bar(x + offset, means, width, yerr=cis, capsize=4,
                      color=DATASET_COLORS[ds], label=DATASET_LABELS[ds].replace("\n", " "),
                      alpha=0.88, zorder=3)
        for xi, (m, s, n) in zip(x + offset, [(data[ds]) for _, _, data in conditions]):
            _, p = detection_pval(m, s, n)
            ci = scipy_stats.t.ppf(0.975, max(n-1,1)) * s / max(np.sqrt(n),1)
            ax.text(xi, m + ci + 0.02, sig_stars(p), ha="center", va="bottom",
                    fontsize=8, color=DATASET_COLORS[ds])

    ax.axhline(0.5, color="red", linestyle="--", linewidth=1.5, alpha=0.7, label="Null (no watermark, p=0.5)")
    ax.axhline(0.05, color="green", linestyle=":", linewidth=1.2, alpha=0.7, label="Detection threshold (p=0.05)")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylabel("Mean detector p-value\n(↓ lower = stronger watermark)", fontsize=10)
    ax.set_title(
        "System-Prompt Robustness — GRPO model\n"
        "Stars: H₁: mean < 0.5 (watermark detected)  |  validation, natural, implicit  |  n=200",
        fontsize=12, fontweight="bold",
    )
    ax.set_ylim(0, 1.15)
    ax.legend(ncol=3, fontsize=8, loc="upper left")
    ax.grid(axis="y", alpha=0.3, zorder=0)

    plt.tight_layout()
    path = os.path.join(out_dir, "robustness_sysprompt_bars.png")
    plt.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


# ── plot 2: heatmap — mean detector p-value per condition × dataset ───────────

def plot_heatmap(conditions: list, out_dir: str) -> None:
    rows = []
    for label, tag, data in conditions:
        for ds in DATASETS:
            m, s, n = data[ds]
            rows.append({"condition": label, "dataset": ds, "mean_p": m})

    df = pd.DataFrame(rows)
    pivot = df.pivot_table(index="condition", columns="dataset", values="mean_p", aggfunc="first")
    pivot = pivot.reindex([c[0] for c in conditions])
    pivot = pivot[DATASETS]

    fig, ax = plt.subplots(figsize=(8, max(3, 0.65*len(conditions) + 2)))
    sns.heatmap(
        pivot, annot=True, fmt=".3f", cmap="RdYlGn_r",
        vmin=0, vmax=1,
        cbar_kws={"label": "Mean detector p-value\n(low=watermark, high=no watermark)"},
        ax=ax, linewidths=0.5,
    )
    ax.set_title(
        "System-Prompt Robustness: Mean Detector P-Value\n"
        "GRPO model — validation, natural, implicit  |  null ≈ 0.5",
        fontsize=12, fontweight="bold",
    )
    ax.set_xlabel("Dataset")
    ax.set_ylabel("System prompt condition")
    ax.set_xticklabels(["GSM8K\n(train)", "ELI5\n(OOD)", "Alpaca\n(OOD)"], fontsize=10)
    plt.tight_layout()
    path = os.path.join(out_dir, "robustness_sysprompt_heatmap.png")
    plt.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


# ── plot 3: -log10(p) per condition × dataset ─────────────────────────────────

def plot_significance(conditions: list, out_dir: str) -> None:
    SIG_LINE = -np.log10(0.05)
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=False)
    labels = [c[0] for c in conditions]
    x = np.arange(len(conditions))

    for ax, ds in zip(axes, DATASETS):
        pvals, neg_log_ps = [], []
        for _, _, data in conditions:
            m, s, n = data[ds]
            _, p = detection_pval(m, s, n)
            p = max(p, 1e-300)
            pvals.append(p)
            neg_log_ps.append(-np.log10(p))

        colors = ["#2ca02c" if p < 0.05 else "#d62728" for p in pvals]
        ax.bar(x, neg_log_ps, color=colors, edgecolor="white", zorder=3)
        ax.axhline(SIG_LINE, color="red", linestyle="--", linewidth=1.5, label="p=0.05", zorder=4)

        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=20, ha="right", fontsize=9)
        ax.set_title(DATASET_LABELS[ds].replace("\n", " "), fontsize=11, fontweight="bold")
        ax.set_ylabel("-log₁₀(p)  H₁: mean < 0.5", fontsize=9)
        ax.legend(fontsize=8)
        ax.grid(axis="y", alpha=0.3, zorder=0)

        for i, (p, nlp) in enumerate(zip(pvals, neg_log_ps)):
            label_str = sig_stars(p) if nlp < 50 else f"{sig_stars(p)}\np≈0"
            if nlp < 50:
                label_str = f"{sig_stars(p)}\n{p:.1e}"
            ax.text(x[i], min(nlp + 0.1, ax.get_ylim()[1]*0.9 if ax.get_ylim()[1] > 0 else 50),
                    sig_stars(p), ha="center", va="bottom", fontsize=8)

    fig.suptitle(
        "System-Prompt Robustness: Detection Significance (GRPO)\n"
        "H₁: mean detector p < 0.5  |  Green = watermark detected  |  validation, natural, implicit",
        fontsize=12, fontweight="bold",
    )
    plt.tight_layout()
    path = os.path.join(out_dir, "robustness_sysprompt_significance.png")
    plt.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


def print_table(conditions: list) -> None:
    print()
    print("=" * 95)
    print("System-Prompt Robustness (GRPO, validation, natural, implicit)")
    print(f"  {'Condition':<16} {'GSM8K mean':>12} {'GSM8K p':>12} {'ELI5 mean':>12} {'ELI5 p':>12} {'Alpaca mean':>12} {'Alpaca p':>12}")
    print("  " + "-" * 89)
    for label, tag, data in conditions:
        line = f"  {label:<16}"
        for ds in DATASETS:
            m, s, n = data[ds]
            _, p = detection_pval(m, s, n)
            line += f" {m:>12.4f} {p:>12.4e}"
        print(line)
    print("=" * 95)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="outputs/utility_plots")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    print_table(CONDITIONS)
    plot_bars(CONDITIONS, args.out_dir)
    plot_heatmap(CONDITIONS, args.out_dir)
    plot_significance(CONDITIONS, args.out_dir)


if __name__ == "__main__":
    main()

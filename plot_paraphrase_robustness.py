#!/usr/bin/env python3
"""
Plot paraphrase robustness results.

Usage:
  python plot_paraphrase_robustness.py \\
      --csv paraphrase_robustness_results/paraphrase_robustness_summary.csv
"""

from __future__ import annotations

import argparse
import glob
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

sns.set_style("whitegrid")
plt.rcParams["figure.dpi"] = 300
plt.rcParams["savefig.dpi"] = 300
plt.rcParams["font.size"] = 11

SIG_LINE = -np.log10(0.05)

DATASET_COLORS = {
    "gsm8k":  "#4c72b0",
    "eli5":   "#dd8452",
    "alpaca": "#55a868",
}


def plot_before_after(df: pd.DataFrame, out_dir: str) -> None:
    datasets = df["dataset"].unique()
    n = len(datasets)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 5), sharey=False)
    if n == 1:
        axes = [axes]

    for ax, ds in zip(axes, datasets):
        row = df[df["dataset"] == ds].iloc[0]
        conditions = ["Original", "Paraphrased"]
        means  = [row["original_mean"], row["para_mean"]]
        pvals  = [row["original_p"],   row["para_p"]]
        colors = [DATASET_COLORS.get(ds, "#888")] * 2

        x = np.arange(2)
        bars = ax.bar(x, [-np.log10(max(p, 1e-300)) for p in pvals],
                      color=colors, alpha=[1.0, 0.55],
                      edgecolor="white", zorder=3)
        ax.axhline(SIG_LINE, color="red", linestyle="--", linewidth=1.5,
                   label="p = 0.05", zorder=4)

        ax.set_xticks(x)
        ax.set_xticklabels(conditions, fontsize=11)
        ax.set_title(f"{ds.upper()}", fontsize=12, fontweight="bold")
        ax.set_ylabel("-log10(p-value)", fontsize=10)
        ax.legend(fontsize=8)
        ax.grid(axis="y", alpha=0.3, zorder=0)

        for bar, p, m in zip(bars, pvals, means):
            stars = "***" if p < 0.001 else ("**" if p < 0.01 else ("*" if p < 0.05 else "ns"))
            nlp = -np.log10(max(p, 1e-300))
            ax.text(bar.get_x() + bar.get_width() / 2, nlp + 0.15,
                    f"{stars}\nmean={m:.3f}\np={p:.2e}",
                    ha="center", va="bottom", fontsize=8)

    fig.suptitle(
        "Paraphrase Robustness — Watermark Signal Before vs After Paraphrase\n"
        "GRPO model (implicit acrostics watermark)",
        fontsize=12, fontweight="bold",
    )
    plt.tight_layout()
    path = os.path.join(out_dir, "paraphrase_robustness_pvalues.png")
    plt.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


def plot_retention(df: pd.DataFrame, out_dir: str) -> None:
    datasets = list(df["dataset"].unique())
    retentions = [float(df[df["dataset"] == ds]["retention_pct"]) for ds in datasets]
    colors = [DATASET_COLORS.get(ds, "#888") for ds in datasets]

    fig, ax = plt.subplots(figsize=(7, 4.5))
    x = np.arange(len(datasets))
    bars = ax.bar(x, retentions, color=colors, edgecolor="white", zorder=3)
    ax.axhline(100, color="gray", linestyle=":", alpha=0.6, label="100% retention")
    ax.axhline(0, color="black", linewidth=0.8)

    for bar, r in zip(bars, retentions):
        ax.text(bar.get_x() + bar.get_width() / 2, r + 1,
                f"{r:.1f}%", ha="center", va="bottom", fontsize=10, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels([ds.upper() for ds in datasets], fontsize=11)
    ax.set_ylabel("Signal retention after paraphrase (%)")
    ax.set_title(
        "Watermark Signal Retention Under Paraphrase Attack\n"
        "(100% = no degradation, 0% = fully erased)",
        fontsize=12, fontweight="bold",
    )
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3, zorder=0)
    ax.set_ylim(0, max(130, max(retentions) * 1.2))

    plt.tight_layout()
    path = os.path.join(out_dir, "paraphrase_robustness_retention.png")
    plt.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=None)
    ap.add_argument("--out-dir", default="outputs/utility_plots")
    args = ap.parse_args()

    csv_path = args.csv
    if csv_path is None:
        candidates = sorted(glob.glob(
            "paraphrase_robustness_results/paraphrase_robustness_summary.csv"
        ))
        if candidates:
            csv_path = candidates[-1]
            print(f"Auto-found: {csv_path}")
        else:
            print("No CSV found. Run paraphrase_robustness.py first.")
            return

    os.makedirs(args.out_dir, exist_ok=True)
    df = pd.read_csv(csv_path)

    print()
    print("=" * 70)
    print("Paraphrase Robustness")
    print(f"  {'Dataset':<12} {'Orig mean':>10} {'Orig p':>10} {'Para mean':>10} {'Para p':>10} {'Retain%':>8}")
    print("  " + "-" * 60)
    for _, r in df.iterrows():
        print(f"  {r['dataset']:<12} {r['original_mean']:>10.4f} {r['original_p']:>10.4e} "
              f"{r['para_mean']:>10.4f} {r['para_p']:>10.4e} {r['retention_pct']:>7.1f}%")
    print("=" * 70)

    plot_before_after(df, args.out_dir)
    plot_retention(df, args.out_dir)


if __name__ == "__main__":
    main()

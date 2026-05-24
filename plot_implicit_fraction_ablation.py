#!/usr/bin/env python3
"""
Plot implicit-fraction ablation results.

Usage:
  python plot_implicit_fraction_ablation.py \\
      --csv ablation_implicit_fraction_<ts>/ablation_implicit_fraction.csv
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
plt.rcParams["font.size"] = 11

DATASET_COLORS = {
    "gsm8k":  "#4c72b0",
    "eli5":   "#dd8452",
    "alpaca": "#55a868",
}
FRACTION_ORDER = ["sft_only", "0.0", "0.3", "0.5", "0.7", "1.0"]
FRACTION_LABELS = {
    "sft_only": "SFT only\n(no GRPO)",
    "0.0":      "frac=0.0\n(explicit)",
    "0.3":      "frac=0.3",
    "0.5":      "frac=0.5",
    "0.7":      "frac=0.7",
    "1.0":      "frac=1.0\n(fully implicit)",
}


def load_data(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df["fraction"] = df["fraction"].astype(str)
    return df


def plot_mean_scores(df: pd.DataFrame, out_dir: str) -> None:
    datasets = [d for d in ["gsm8k", "eli5", "alpaca"] if d in df["dataset"].values]
    fracs = [f for f in FRACTION_ORDER if f in df["fraction"].values]

    x = np.arange(len(fracs))
    width = 0.25
    n_ds = len(datasets)
    offsets = np.linspace(-(n_ds - 1) / 2 * width, (n_ds - 1) / 2 * width, n_ds)

    fig, ax = plt.subplots(figsize=(12, 5))

    for ds, offset in zip(datasets, offsets):
        sub = df[df["dataset"] == ds]
        means, cis = [], []
        for f in fracs:
            row = sub[sub["fraction"] == f]
            if row.empty:
                means.append(float("nan"))
                cis.append(0)
            else:
                m, s, n = float(row["mean"]), float(row["std"]), int(row["n"])
                means.append(m)
                ci = scipy_stats.t.ppf(0.975, max(n - 1, 1)) * s / max(np.sqrt(n), 1)
                cis.append(ci)

        ax.bar(x + offset, means, width, yerr=cis, capsize=4,
               label=ds.upper(), color=DATASET_COLORS.get(ds, "#888"),
               alpha=0.85, edgecolor="white", zorder=3)

    ax.set_xticks(x)
    ax.set_xticklabels([FRACTION_LABELS.get(f, f) for f in fracs], fontsize=10)
    ax.axhline(0, color="gray", linestyle=":", alpha=0.6)
    ax.set_ylabel("Mean acrostics score (implicit, validation)")
    ax.set_title(
        "Implicit-Fraction Ablation — GRPO trained from GSM8K SFT checkpoint\n"
        "Higher = stronger watermark signal",
        fontsize=12, fontweight="bold",
    )
    ax.legend(title="Eval dataset", fontsize=9)
    ax.grid(axis="y", alpha=0.3, zorder=0)

    plt.tight_layout()
    path = os.path.join(out_dir, "ablation_implicit_fraction_scores.png")
    plt.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


def plot_pvalues(df: pd.DataFrame, out_dir: str) -> None:
    datasets = [d for d in ["gsm8k", "eli5", "alpaca"] if d in df["dataset"].values]
    fracs = [f for f in FRACTION_ORDER if f in df["fraction"].values]

    SIG_LINE = -np.log10(0.05)
    fig, axes = plt.subplots(1, len(datasets), figsize=(6.5 * len(datasets), 5), sharey=False)
    if len(datasets) == 1:
        axes = [axes]

    for ax, ds in zip(axes, datasets):
        sub = df[df["dataset"] == ds]
        neg_log_ps, pvals = [], []
        for f in fracs:
            row = sub[sub["fraction"] == f]
            if row.empty or np.isnan(float(row["p"])):
                neg_log_ps.append(0)
                pvals.append(1.0)
            else:
                p = max(float(row["p"]), 1e-300)
                neg_log_ps.append(-np.log10(p))
                pvals.append(p)

        colors = ["#2ca02c" if p < 0.05 else "#d62728" for p in pvals]
        x = np.arange(len(fracs))
        ax.bar(x, neg_log_ps, color=colors, edgecolor="white", zorder=3)
        ax.axhline(SIG_LINE, color="red", linestyle="--", linewidth=1.5,
                   label="p = 0.05", zorder=4)

        ax.set_xticks(x)
        ax.set_xticklabels([FRACTION_LABELS.get(f, f) for f in fracs], fontsize=9)
        ax.set_title(f"{ds.upper()} (implicit validation)", fontsize=11, fontweight="bold")
        ax.set_ylabel("-log10(p-value)")
        ax.legend(fontsize=8)
        ax.grid(axis="y", alpha=0.3, zorder=0)

        for i, (p, nlp) in enumerate(zip(pvals, neg_log_ps)):
            stars = "***" if p < 0.001 else ("**" if p < 0.01 else ("*" if p < 0.05 else "ns"))
            ax.text(x[i], nlp + 0.1, f"{stars}\np={p:.2e}", ha="center", va="bottom",
                    fontsize=8)

    fig.suptitle(
        "Implicit-Fraction Ablation: Watermark Significance vs Null\n"
        "H₀: mean score = 0  |  Green = p<0.05  Red = not significant",
        fontsize=12, fontweight="bold",
    )
    plt.tight_layout()
    path = os.path.join(out_dir, "ablation_implicit_fraction_pvalues.png")
    plt.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


def plot_line(df: pd.DataFrame, out_dir: str) -> None:
    """Line plot showing signal vs fraction for each dataset."""
    grpo_only = df[df["fraction"] != "sft_only"].copy()
    try:
        grpo_only["frac_num"] = grpo_only["fraction"].astype(float)
    except ValueError:
        return

    datasets = [d for d in ["gsm8k", "eli5", "alpaca"] if d in grpo_only["dataset"].values]

    fig, ax = plt.subplots(figsize=(9, 5))
    for ds in datasets:
        sub = grpo_only[grpo_only["dataset"] == ds].sort_values("frac_num")
        ax.plot(sub["frac_num"], sub["mean"], "-o",
                color=DATASET_COLORS.get(ds, "#888"),
                linewidth=2, markersize=8, label=ds.upper(), zorder=3)
        ax.fill_between(sub["frac_num"],
                        sub["mean"] - sub["std"] / np.sqrt(sub["n"]),
                        sub["mean"] + sub["std"] / np.sqrt(sub["n"]),
                        color=DATASET_COLORS.get(ds, "#888"), alpha=0.15)

    # Mark SFT-only baseline
    sft = df[df["fraction"] == "sft_only"]
    for ds in datasets:
        row = sft[sft["dataset"] == ds]
        if not row.empty:
            ax.axhline(float(row["mean"]), color=DATASET_COLORS.get(ds, "#888"),
                       linestyle=":", alpha=0.5, linewidth=1)

    ax.axhline(0, color="black", linestyle="-", linewidth=0.8, alpha=0.4)
    ax.set_xlabel("GRPO implicit fraction\n(0 = always told to watermark, 1 = never told)")
    ax.set_ylabel("Mean acrostics score (implicit validation)")
    ax.set_title(
        "Watermark Signal vs Implicit Fraction\n"
        "Dotted lines = SFT-only baseline for each dataset",
        fontsize=12, fontweight="bold",
    )
    ax.legend(title="Eval dataset", fontsize=9)
    ax.set_xticks([0.0, 0.3, 0.5, 0.7, 1.0])
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    path = os.path.join(out_dir, "ablation_implicit_fraction_line.png")
    plt.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


def print_table(df: pd.DataFrame) -> None:
    fracs = [f for f in FRACTION_ORDER if f in df["fraction"].values]
    datasets = [d for d in ["gsm8k", "eli5", "alpaca"] if d in df["dataset"].values]

    print()
    print("=" * 80)
    print("Implicit-Fraction Ablation Summary (validation, natural, implicit)")
    header = f"  {'Fraction':<14}"
    for ds in datasets:
        header += f"  {ds.upper():<18}"
    print(header)
    print("  " + "-" * (14 + 20 * len(datasets)))

    for f in fracs:
        line = f"  {FRACTION_LABELS.get(f, f).replace(chr(10), ' '):<14}"
        for ds in datasets:
            row = df[(df["fraction"] == f) & (df["dataset"] == ds)]
            if row.empty:
                line += "  " + "n/a".rjust(18)
            else:
                m, p = float(row["mean"]), float(row["p"])
                stars = "***" if p < 0.001 else ("**" if p < 0.01 else ("*" if p < 0.05 else "ns"))
                line += f"  {m:>7.4f} (p={p:.2e}) {stars:>3}"
        print(line)
    print("=" * 80)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=None)
    ap.add_argument("--out-dir", default="outputs/utility_plots")
    args = ap.parse_args()

    csv_path = args.csv
    if csv_path is None:
        candidates = sorted(glob.glob("ablation_implicit_fraction_*/ablation_implicit_fraction.csv"))
        if candidates:
            csv_path = candidates[-1]
            print(f"Auto-found: {csv_path}")
        else:
            print("No CSV found. Run run_implicit_fraction_ablation.sh first.")
            return

    os.makedirs(args.out_dir, exist_ok=True)
    df = load_data(csv_path)

    print_table(df)
    plot_mean_scores(df, args.out_dir)
    plot_pvalues(df, args.out_dir)
    plot_line(df, args.out_dir)


if __name__ == "__main__":
    main()

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

# ── hardcoded data from ablation_implicit_fraction_20260524_122442 ─────────────
# mean = mean acrostics detector p-value (LOWER = stronger watermark, null = 0.5)
HARDCODED_ROWS = [
    # fraction, dataset, mean, std, n
    ("sft_only", "gsm8k",  0.0163, 0.0292, 200),
    ("sft_only", "eli5",   0.3878, 0.3110, 200),
    ("sft_only", "alpaca", 0.6065, 0.3903, 200),
    ("0.0",      "gsm8k",  0.0168, 0.0242, 200),
    ("0.0",      "eli5",   0.3519, 0.3188, 200),
    ("0.0",      "alpaca", 0.5223, 0.3896, 200),
    ("0.3",      "gsm8k",  0.0147, 0.0177, 200),
    ("0.3",      "eli5",   0.3130, 0.3145, 200),
    ("0.3",      "alpaca", 0.5259, 0.3861, 200),
    ("0.5",      "gsm8k",  0.0173, 0.0293, 200),
    ("0.5",      "eli5",   0.3770, 0.3220, 200),
    ("0.5",      "alpaca", 0.5194, 0.3848, 200),
    ("0.7",      "gsm8k",  0.0138, 0.0098, 200),
    ("0.7",      "eli5",   0.2994, 0.2993, 200),
    ("0.7",      "alpaca", 0.5183, 0.3891, 200),
    ("1.0",      "gsm8k",  0.0126, 0.0050, 200),
    ("1.0",      "eli5",   0.3500, 0.3048, 200),
    ("1.0",      "alpaca", 0.5114, 0.3833, 200),
]

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


def load_data(csv_path: str | None) -> pd.DataFrame:
    if csv_path and os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        df["fraction"] = df["fraction"].astype(str)
        return df
    # Fall back to hardcoded data
    rows = []
    for frac, ds, mean, std, n in HARDCODED_ROWS:
        # z and p vs H0: mean = 0.5 (null detector p-value), one-sided lower
        se = std / np.sqrt(n)
        z_vs_null = (mean - 0.5) / se   # negative = below null = watermark works
        p_vs_null = float(scipy_stats.norm.cdf(z_vs_null))  # one-sided: P(Z < z)
        rows.append({"fraction": frac, "dataset": ds,
                     "mean": mean, "std": std, "n": n,
                     "z": z_vs_null, "p": p_vs_null})
    return pd.DataFrame(rows)


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
    ax.axhline(0.5, color="red", linestyle="--", linewidth=1.5, alpha=0.7,
               label="Null (p=0.5, no watermark)")
    ax.axhline(0.05, color="green", linestyle=":", linewidth=1.2, alpha=0.7,
               label="Detection threshold (p=0.05)")
    ax.set_ylabel("Mean detector p-value per response\n(↓ lower = stronger watermark)")
    ax.set_title(
        "Implicit-Fraction Ablation — GRPO from GSM8K SFT checkpoint\n"
        "Lower score = stronger watermark  |  Red dashed = null (no watermark)",
        fontsize=12, fontweight="bold",
    )
    ax.legend(title="Eval dataset", fontsize=9, loc="upper right")
    ax.grid(axis="y", alpha=0.3, zorder=0)
    ax.set_ylim(0, 0.75)

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
            if row.empty:
                neg_log_ps.append(0)
                pvals.append(1.0)
            else:
                m, s, n = float(row["mean"]), float(row["std"]), int(row["n"])
                # One-sided test: H0: mean >= 0.5, H1: mean < 0.5 (watermark present)
                se = s / np.sqrt(n)
                z = (m - 0.5) / se
                p = float(scipy_stats.norm.cdf(z))  # P(Z <= z), lower tail
                p = max(p, 1e-300)
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
        ax.set_ylabel("-log10(p-value)\nH₁: mean detector p < 0.5 (watermark present)")
        ax.legend(fontsize=8)
        ax.grid(axis="y", alpha=0.3, zorder=0)

        for i, (p, nlp) in enumerate(zip(pvals, neg_log_ps)):
            stars = "***" if p < 0.001 else ("**" if p < 0.01 else ("*" if p < 0.05 else "ns"))
            ax.text(x[i], nlp + 0.1, f"{stars}\np={p:.2e}", ha="center", va="bottom",
                    fontsize=8)

    fig.suptitle(
        "Implicit-Fraction Ablation: Watermark Detection Significance\n"
        "H₀: mean detector p-value ≥ 0.5 (no watermark)  |  Green = significant",
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

    ax.axhline(0.5, color="red", linestyle="--", linewidth=1.5, alpha=0.7,
               label="Null (p=0.5, random)")
    ax.axhline(0.05, color="green", linestyle=":", linewidth=1.2, alpha=0.7,
               label="Detection threshold (p=0.05)")
    ax.set_xlabel("GRPO implicit fraction\n(0 = model always told to watermark, 1 = never told)")
    ax.set_ylabel("Mean detector p-value per response\n(↓ lower = stronger watermark)")
    ax.set_title(
        "Watermark Signal vs Implicit Fraction\n"
        "Dotted coloured lines = SFT-only baseline  |  Red dashed = null",
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
            print("No CSV found — using hardcoded data from ablation run 20260524_122442.")

    os.makedirs(args.out_dir, exist_ok=True)
    df = load_data(csv_path)

    print_table(df)
    plot_mean_scores(df, args.out_dir)
    plot_pvalues(df, args.out_dir)
    plot_line(df, args.out_dir)


if __name__ == "__main__":
    main()

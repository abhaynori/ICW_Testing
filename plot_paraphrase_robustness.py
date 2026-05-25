#!/usr/bin/env python3
"""
Plot paraphrase robustness results.
Metric: acrostics detector p-value (0-1, LOWER = stronger watermark, null ≈ 0.5)

Key result: paraphrase (Qwen2.5-7B local) completely destroys the watermark.
Note: run used right-padding (before fix); paraphrase quality may be slightly degraded.

Usage:
  python plot_paraphrase_robustness.py          # hardcoded data from run 20260525
  python plot_paraphrase_robustness.py --csv paraphrase_robustness_results/paraphrase_robustness_summary.csv
"""

from __future__ import annotations

import argparse
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
DATASET_LABELS = {
    "gsm8k":  "GSM8K (train domain)",
    "eli5":   "ELI5 (OOD)",
    "alpaca": "Alpaca (OOD, no original signal)",
}
DATASETS = ["gsm8k", "eli5", "alpaca"]

# ── hardcoded data from run 20260525 (Qwen2.5-7B local paraphraser) ──────────
# mean_p: mean detector p-value (lower = stronger watermark, null ≈ 0.5)
# std_p: std of detector p-values (derived from z_output = mean*sqrt(n)/std)
# z_output from paraphrase_robustness.py tests H0: mean=0 — not the detection test.
# Detection p-values recomputed here: one-sided H0: mean>=0.5, H1: mean<0.5.
HARDCODED = {
    "gsm8k": {
        "orig_mean": 0.0143, "orig_std": 0.01375, "n": 200,
        "para_mean": 0.8630, "para_std": 0.2390,
    },
    "eli5": {
        "orig_mean": 0.3303, "orig_std": 0.3068, "n": 200,
        "para_mean": 0.7533, "para_std": 0.3011,
    },
    "alpaca": {
        "orig_mean": 0.5085, "orig_std": 0.3988, "n": 200,
        "para_mean": 0.8239, "para_std": 0.2821,
    },
}


def detection_pval(mean: float, std: float, n: int) -> tuple[float, float]:
    """One-sided H0: mean >= 0.5  H1: mean < 0.5 (watermark present)."""
    se = std / np.sqrt(max(n, 2))
    z = (mean - 0.5) / max(se, 1e-12)
    p = float(scipy_stats.norm.cdf(z))
    return float(z), p


def sig_stars(p: float) -> str:
    if p < 0.001: return "***"
    if p < 0.01:  return "**"
    if p < 0.05:  return "*"
    return "ns"


def signal_distance(mean: float) -> float:
    """Distance from null=0.5; positive = watermark present, negative = no watermark."""
    return 0.5 - mean


def build_df(csv_path: str | None) -> pd.DataFrame:
    rows = []
    if csv_path and os.path.exists(csv_path):
        raw = pd.read_csv(csv_path)
        for _, r in raw.iterrows():
            ds = str(r["dataset"]).strip()
            # Derive std from z_output (H0: mean=0): std = mean*sqrt(n)/|z|
            n = int(r["n"])
            om = float(r["original_mean"])
            oz = abs(float(r.get("original_z", r.get("original_z", 1))))
            os_ = om * np.sqrt(n) / max(oz, 1e-9)
            pm = float(r["para_mean"])
            pz = abs(float(r.get("para_z", 1)))
            ps = pm * np.sqrt(n) / max(pz, 1e-9)
            rows.append({"dataset": ds, "orig_mean": om, "orig_std": os_,
                         "para_mean": pm, "para_std": ps, "n": n})
    else:
        for ds, d in HARDCODED.items():
            rows.append({"dataset": ds, "orig_mean": d["orig_mean"],
                         "orig_std": d["orig_std"], "para_mean": d["para_mean"],
                         "para_std": d["para_std"], "n": d["n"]})

    df = pd.DataFrame(rows)
    df["dataset"] = pd.Categorical(df["dataset"], categories=DATASETS, ordered=True)
    df = df.sort_values("dataset").reset_index(drop=True)

    # Compute detection p-values (H0: mean>=0.5, one-sided lower tail)
    orig_ps, para_ps = [], []
    for _, r in df.iterrows():
        _, op = detection_pval(r["orig_mean"], r["orig_std"], int(r["n"]))
        _, pp = detection_pval(r["para_mean"], r["para_std"], int(r["n"]))
        orig_ps.append(op)
        para_ps.append(pp)
    df["orig_p_det"] = orig_ps
    df["para_p_det"] = para_ps

    # Signal distance: 0.5 - mean (positive = watermark present)
    df["orig_signal"] = 0.5 - df["orig_mean"]
    df["para_signal"] = 0.5 - df["para_mean"]

    return df


# ── plot 1: before/after mean detector p-value (the clearest story) ───────────

def plot_before_after_means(df: pd.DataFrame, out_dir: str) -> None:
    fig, ax = plt.subplots(figsize=(10, 5))

    n_ds = len(DATASETS)
    x = np.arange(n_ds)
    width = 0.32

    orig_means, para_means = [], []
    orig_cis, para_cis = [], []
    orig_ps, para_ps = [], []

    for ds in DATASETS:
        row = df[df["dataset"] == ds].iloc[0]
        n = int(row["n"])
        orig_means.append(row["orig_mean"])
        para_means.append(row["para_mean"])
        orig_cis.append(scipy_stats.t.ppf(0.975, n-1) * row["orig_std"] / np.sqrt(n))
        para_cis.append(scipy_stats.t.ppf(0.975, n-1) * row["para_std"] / np.sqrt(n))
        orig_ps.append(row["orig_p_det"])
        para_ps.append(row["para_p_det"])

    colors = [DATASET_COLORS[ds] for ds in DATASETS]

    bars_orig = ax.bar(x - width/2, orig_means, width, yerr=orig_cis, capsize=5,
                       color=colors, alpha=0.95, label="Original", zorder=3,
                       edgecolor="white")
    bars_para = ax.bar(x + width/2, para_means, width, yerr=para_cis, capsize=5,
                       color=colors, alpha=0.45, label="After paraphrase", zorder=3,
                       edgecolor="white", hatch="//")

    ax.axhline(0.5, color="red", linestyle="--", linewidth=1.8, alpha=0.8,
               label="Null (no watermark, p=0.5)")
    ax.axhline(0.05, color="green", linestyle=":", linewidth=1.3, alpha=0.8,
               label="Detection threshold (p=0.05)")

    # Significance stars above each bar
    for i, (om, op, pm, pp, ci_o, ci_p) in enumerate(
            zip(orig_means, orig_ps, para_means, para_ps, orig_cis, para_cis)):
        ax.text(x[i] - width/2, om + ci_o + 0.02, sig_stars(op),
                ha="center", va="bottom", fontsize=11, fontweight="bold",
                color=colors[i])
        ax.text(x[i] + width/2, pm + ci_p + 0.02, sig_stars(pp),
                ha="center", va="bottom", fontsize=10, color="gray")

    ax.set_xticks(x)
    ax.set_xticklabels([DATASET_LABELS[ds] for ds in DATASETS], fontsize=10)
    ax.set_ylabel("Mean detector p-value\n(↓ lower = stronger watermark, null = 0.5)", fontsize=10)
    ax.set_ylim(0, 1.15)
    ax.set_title(
        "Paraphrase Attack — Watermark Before vs After\n"
        "Stars on original: H₁: mean < 0.5 (detected)  |  GRPO model, implicit, test split, n=200",
        fontsize=12, fontweight="bold",
    )
    ax.legend(fontsize=9, loc="upper left")
    ax.grid(axis="y", alpha=0.3, zorder=0)

    plt.tight_layout()
    path = os.path.join(out_dir, "paraphrase_robustness_means.png")
    plt.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


# ── plot 2: -log10(detection p-value) before and after ───────────────────────

def plot_before_after_pvals(df: pd.DataFrame, out_dir: str) -> None:
    SIG_LINE = -np.log10(0.05)
    # Only show datasets where original watermark was present or partially present
    datasets_to_show = [ds for ds in DATASETS
                        if df[df["dataset"] == ds].iloc[0]["orig_p_det"] < 0.5]

    n = len(datasets_to_show) if datasets_to_show else 1
    fig, axes = plt.subplots(1, n, figsize=(6.5 * n, 5), sharey=False)
    if n == 1:
        axes = [axes]

    for ax, ds in zip(axes, datasets_to_show):
        row = df[df["dataset"] == ds].iloc[0]
        pvals = [row["orig_p_det"], row["para_p_det"]]
        means = [row["orig_mean"], row["para_mean"]]
        neg_log_ps = [-np.log10(max(p, 1e-300)) for p in pvals]
        labels = ["Original", "After paraphrase"]
        c = DATASET_COLORS[ds]

        x = np.arange(2)
        ax.bar(x[:1], neg_log_ps[:1], color=c, alpha=1.0, edgecolor="white", zorder=3)
        ax.bar(x[1:], neg_log_ps[1:], color=c, alpha=0.45, edgecolor="white", zorder=3, hatch="//")
        ax.axhline(SIG_LINE, color="red", linestyle="--", linewidth=1.5,
                   label="p=0.05 threshold", zorder=4)

        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=11)
        ax.set_title(DATASET_LABELS[ds], fontsize=11, fontweight="bold")
        ax.set_ylabel("-log₁₀(p)  H₁: mean detector p < 0.5", fontsize=9)
        ax.legend(fontsize=8)
        ax.grid(axis="y", alpha=0.3, zorder=0)

        for xi, (p, nlp, m) in enumerate(zip(pvals, neg_log_ps, means)):
            stars = sig_stars(p)
            label = f"{stars}\nmean={m:.3f}\np={p:.1e}" if nlp < 200 else f"{stars}\nmean={m:.3f}\np≈0"
            ax.text(xi, nlp + 0.3, label, ha="center", va="bottom", fontsize=9)

    fig.suptitle(
        "Paraphrase Attack: Detection Significance Before vs After\n"
        "H₁: mean detector p < 0.5 (watermark present)  |  GRPO model, implicit, test split",
        fontsize=12, fontweight="bold",
    )
    plt.tight_layout()
    path = os.path.join(out_dir, "paraphrase_robustness_pvalues.png")
    plt.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


# ── plot 3: signal remaining (distance from null=0.5) ─────────────────────────

def plot_signal_remaining(df: pd.DataFrame, out_dir: str) -> None:
    fig, ax = plt.subplots(figsize=(9, 5))

    x = np.arange(len(DATASETS))
    width = 0.32
    colors = [DATASET_COLORS[ds] for ds in DATASETS]

    orig_signals = [df[df["dataset"] == ds].iloc[0]["orig_signal"] for ds in DATASETS]
    para_signals = [df[df["dataset"] == ds].iloc[0]["para_signal"] for ds in DATASETS]

    ax.bar(x - width/2, orig_signals, width, color=colors, alpha=0.95,
           label="Original (0.5 − mean)", zorder=3, edgecolor="white")
    ax.bar(x + width/2, para_signals, width, color=colors, alpha=0.45,
           label="After paraphrase (0.5 − mean)", zorder=3, edgecolor="white", hatch="//")

    ax.axhline(0, color="red", linestyle="--", linewidth=1.5, alpha=0.8,
               label="0 = null (no watermark)")
    ax.axhline(0.45, color="green", linestyle=":", linewidth=1.2, alpha=0.6,
               label="≈0.45 (strongest detectable signal)")

    # Annotate original bars
    for i, (ds, osig, psig) in enumerate(zip(DATASETS, orig_signals, para_signals)):
        ax.text(x[i] - width/2, osig + (0.01 if osig >= 0 else -0.03),
                f"{osig:+.3f}", ha="center", va="bottom" if osig >= 0 else "top",
                fontsize=9, fontweight="bold", color=colors[i])
        ax.text(x[i] + width/2, psig + (0.01 if psig >= 0 else -0.03),
                f"{psig:+.3f}", ha="center", va="bottom" if psig >= 0 else "top",
                fontsize=9, color="gray")

    ax.set_xticks(x)
    ax.set_xticklabels([DATASET_LABELS[ds] for ds in DATASETS], fontsize=10)
    ax.set_ylabel("Signal distance from null\n(0.5 − mean detector p-value)\n"
                  "Positive = watermark present", fontsize=10)
    ax.set_title(
        "Watermark Signal Remaining After Paraphrase Attack\n"
        "All signals driven negative (above null) after paraphrase — watermark destroyed",
        fontsize=12, fontweight="bold",
    )
    ax.legend(fontsize=9, loc="upper right")
    ax.grid(axis="y", alpha=0.3, zorder=0)

    plt.tight_layout()
    path = os.path.join(out_dir, "paraphrase_robustness_signal.png")
    plt.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


def print_table(df: pd.DataFrame) -> None:
    print()
    print("=" * 90)
    print("Paraphrase Robustness (GRPO, implicit, test split, Qwen2.5-7B paraphraser)")
    print("Detection p-value: one-sided H0: mean >= 0.5  (p<0.05 → watermark detected)")
    print()
    print(f"  {'Dataset':<30} {'Orig mean':>10} {'Orig p_det':>12} {'Para mean':>10} {'Para p_det':>12} {'Signal Δ':>10}")
    print("  " + "-" * 84)
    for _, r in df.iterrows():
        ds_label = DATASET_LABELS.get(r["dataset"], r["dataset"])
        delta = r["para_signal"] - r["orig_signal"]
        orig_sig = sig_stars(r["orig_p_det"])
        para_sig = sig_stars(r["para_p_det"])
        print(f"  {ds_label:<30} {r['orig_mean']:>10.4f} {r['orig_p_det']:>10.4e} {orig_sig:>2} "
              f"{r['para_mean']:>10.4f} {r['para_p_det']:>10.4e} {para_sig:>2} {delta:>+9.3f}")
    print()
    print("  Signal Δ = (para_signal − orig_signal): negative = watermark destroyed")
    print("  All three datasets show watermark erased after paraphrase.")
    print("  Note: run used right-padding (minor quality impact); re-run with fixed script for paper.")
    print("=" * 90)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=None)
    ap.add_argument("--out-dir", default="outputs/utility_plots")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    csv_path = args.csv
    if csv_path is None:
        import glob
        candidates = sorted(glob.glob(
            "paraphrase_robustness_results/paraphrase_robustness_summary.csv"
        ))
        if candidates:
            csv_path = candidates[-1]
            print(f"Auto-found: {csv_path}")

    df = build_df(csv_path)
    print_table(df)
    plot_before_after_means(df, args.out_dir)
    plot_before_after_pvals(df, args.out_dir)
    plot_signal_remaining(df, args.out_dir)


if __name__ == "__main__":
    main()

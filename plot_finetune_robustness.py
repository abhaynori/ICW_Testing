#!/usr/bin/env python3
"""
Plot Phase 3 fine-tuning robustness results.

Reads the CSV produced by finetune_robustness.py (which has a 'model' column
with values 'grpo' and 'base') and generates:
  1. Line plot: implicit score vs. fine-tuning steps — GRPO vs Base
  2. -log10(p) line plot: significance vs. fine-tuning steps — GRPO vs Base
  3. Summary table printed to stdout

Usage:
  python plot_finetune_robustness.py --csv robustness_logs/finetune_*/finetune_robustness_results.csv
"""

from __future__ import annotations

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats as scipy_stats

sns.set_style("whitegrid")
plt.rcParams["figure.dpi"] = 300
plt.rcParams["savefig.dpi"] = 300
plt.rcParams["font.size"] = 10

OUT_DIR = "outputs/utility_plots"

MODEL_STYLES = {
    "grpo": {"color": "#4c72b0", "label": "GRPO (watermarked)", "marker": "o", "lw": 2.5, "zorder": 4},
    "base": {"color": "#cccccc", "label": "Base (unwatermarked)", "marker": "s", "lw": 1.5, "zorder": 3},
}

# ── hardcoded data from run gsm8k_20260524_224612 (Phase 3 — fine-tuning) ────
# model → step → {dataset_val_implicit_mean/std/n}
# mean = mean detector p-value; lower = stronger watermark; null ≈ 0.5
# Fine-tuning: LoRA on clean Alpaca (test split, 2000 samples, 1000 steps, lr=2e-5)
HARDCODED: dict[str, dict[int, dict]] = {
    "grpo": {
        0:    {"eli5_val_implicit_mean": 0.4522, "eli5_val_implicit_std": 0.3449, "eli5_val_implicit_n": 200,
               "alpaca_val_implicit_mean": 0.5316, "alpaca_val_implicit_std": 0.3590, "alpaca_val_implicit_n": 200,
               "gsm8k_val_implicit_mean": 0.0136, "gsm8k_val_implicit_std": 0.0119, "gsm8k_val_implicit_n": 200},
        100:  {"eli5_val_implicit_mean": 0.8491, "eli5_val_implicit_std": 0.2539, "eli5_val_implicit_n": 200,
               "alpaca_val_implicit_mean": 0.8009, "alpaca_val_implicit_std": 0.2906, "alpaca_val_implicit_n": 200,
               "gsm8k_val_implicit_mean": 0.5379, "gsm8k_val_implicit_std": 0.3492, "gsm8k_val_implicit_n": 200},
        250:  {"eli5_val_implicit_mean": 0.8709, "eli5_val_implicit_std": 0.2239, "eli5_val_implicit_n": 200,
               "alpaca_val_implicit_mean": 0.8457, "alpaca_val_implicit_std": 0.2621, "alpaca_val_implicit_n": 200,
               "gsm8k_val_implicit_mean": 0.7402, "gsm8k_val_implicit_std": 0.3052, "gsm8k_val_implicit_n": 200},
        500:  {"eli5_val_implicit_mean": 0.8632, "eli5_val_implicit_std": 0.2419, "eli5_val_implicit_n": 200,
               "alpaca_val_implicit_mean": 0.8643, "alpaca_val_implicit_std": 0.2596, "alpaca_val_implicit_n": 200,
               "gsm8k_val_implicit_mean": 0.8241, "gsm8k_val_implicit_std": 0.2669, "gsm8k_val_implicit_n": 200},
        1000: {"eli5_val_implicit_mean": 0.8678, "eli5_val_implicit_std": 0.2593, "eli5_val_implicit_n": 200,
               "alpaca_val_implicit_mean": 0.8192, "alpaca_val_implicit_std": 0.2899, "alpaca_val_implicit_n": 200,
               "gsm8k_val_implicit_mean": 0.8144, "gsm8k_val_implicit_std": 0.2795, "gsm8k_val_implicit_n": 200},
    },
    "base": {
        0:    {"eli5_val_implicit_mean": 0.8817, "eli5_val_implicit_std": 0.2376, "eli5_val_implicit_n": 200,
               "alpaca_val_implicit_mean": 0.8159, "alpaca_val_implicit_std": 0.2881, "alpaca_val_implicit_n": 200,
               "gsm8k_val_implicit_mean": 0.8421, "gsm8k_val_implicit_std": 0.2487, "gsm8k_val_implicit_n": 200},
        100:  {"eli5_val_implicit_mean": 0.7981, "eli5_val_implicit_std": 0.2921, "eli5_val_implicit_n": 200,
               "alpaca_val_implicit_mean": 0.8177, "alpaca_val_implicit_std": 0.2777, "alpaca_val_implicit_n": 200,
               "gsm8k_val_implicit_mean": 0.8282, "gsm8k_val_implicit_std": 0.2613, "gsm8k_val_implicit_n": 200},
        250:  {"eli5_val_implicit_mean": 0.8508, "eli5_val_implicit_std": 0.2489, "eli5_val_implicit_n": 200,
               "alpaca_val_implicit_mean": 0.8398, "alpaca_val_implicit_std": 0.2744, "alpaca_val_implicit_n": 200,
               "gsm8k_val_implicit_mean": 0.8640, "gsm8k_val_implicit_std": 0.2511, "gsm8k_val_implicit_n": 200},
        500:  {"eli5_val_implicit_mean": 0.8808, "eli5_val_implicit_std": 0.2253, "eli5_val_implicit_n": 200,
               "alpaca_val_implicit_mean": 0.8533, "alpaca_val_implicit_std": 0.2722, "alpaca_val_implicit_n": 200,
               "gsm8k_val_implicit_mean": 0.8736, "gsm8k_val_implicit_std": 0.2326, "gsm8k_val_implicit_n": 200},
        1000: {"eli5_val_implicit_mean": 0.8478, "eli5_val_implicit_std": 0.2581, "eli5_val_implicit_n": 200,
               "alpaca_val_implicit_mean": 0.8445, "alpaca_val_implicit_std": 0.2658, "alpaca_val_implicit_n": 200,
               "gsm8k_val_implicit_mean": 0.8770, "gsm8k_val_implicit_std": 0.2269, "gsm8k_val_implicit_n": 200},
    },
}


def zscore_pval(mean: float, std: float, n: int) -> tuple[float, float]:
    """One-sided test H₀: mean ≥ 0.5  H₁: mean < 0.5 (watermark present)."""
    se = std / np.sqrt(max(n, 2))
    z = (mean - 0.5) / max(se, 1e-12)
    p = float(scipy_stats.norm.cdf(z))  # P(Z ≤ z), lower tail
    return float(z), float(p)


def sig_stars(p: float) -> str:
    if p < 0.001: return "***"
    if p < 0.01:  return "**"
    if p < 0.05:  return "*"
    return "ns"


def load_data(csv_path: str | None) -> pd.DataFrame:
    if csv_path:
        df = pd.read_csv(csv_path)
    elif HARDCODED:
        rows = []
        for model, steps in HARDCODED.items():
            for step, d in sorted(steps.items()):
                rows.append({"model": model, "step": step, **d})
        df = pd.DataFrame(rows)
    else:
        raise ValueError("No CSV path provided and no hardcoded data.")

    if "model" not in df.columns:
        df["model"] = "grpo"

    for ds in ("eli5", "alpaca", "gsm8k"):
        key = f"{ds}_val_implicit"
        if f"{key}_mean" not in df.columns:
            continue
        if f"{key}_z" not in df.columns:
            zs, ps = [], []
            for _, row in df.iterrows():
                z, p = zscore_pval(row[f"{key}_mean"], row[f"{key}_std"], int(row[f"{key}_n"]))
                zs.append(z)
                ps.append(p)
            df[f"{key}_z"] = zs
            df[f"{key}_p"] = ps

    return df.sort_values(["model", "step"]).reset_index(drop=True)


def plot_scores(df: pd.DataFrame, out_dir: str) -> None:
    """Line plot: mean acrostics score vs. fine-tuning steps, GRPO vs Base."""
    datasets = [
        ("eli5_val_implicit",   "ELI5 validation (implicit)"),
        ("alpaca_val_implicit", "Alpaca validation (implicit)"),
        ("gsm8k_val_implicit",  "GSM8K validation (implicit, train domain)"),
    ]
    datasets = [(k, l) for k, l in datasets if f"{k}_mean" in df.columns]
    ncols = len(datasets)
    fig, axes = plt.subplots(1, ncols, figsize=(6.5 * ncols, 5), sharey=False)
    if ncols == 1:
        axes = [axes]

    for ax, (ds_key, ds_label) in zip(axes, datasets):
        for model_name, style in MODEL_STYLES.items():
            sub = df[df["model"] == model_name].sort_values("step")
            if sub.empty:
                continue
            means = sub[f"{ds_key}_mean"].values
            stds  = sub[f"{ds_key}_std"].values
            ns    = sub[f"{ds_key}_n"].values.astype(int)
            cis   = scipy_stats.t.ppf(0.975, ns - 1) * stds / np.sqrt(ns)
            steps = sub["step"].values

            ax.plot(steps, means, f"-{style['marker']}",
                    color=style["color"], linewidth=style["lw"],
                    markersize=7, zorder=style["zorder"], label=style["label"])
            ax.fill_between(steps, means - cis, means + cis,
                            color=style["color"], alpha=0.15, zorder=style["zorder"] - 1)

            # Significance stars for GRPO only (base is expected to stay ~0)
            if model_name == "grpo":
                ymax = (means + cis).max()
                for x, m, ci, p in zip(steps, means, cis, sub[f"{ds_key}_p"].values):
                    ax.text(x, m + ci + 0.02 * abs(ymax), sig_stars(p),
                            ha="center", va="bottom", fontsize=11, fontweight="bold",
                            color=style["color"])

        ax.axhline(0.5, color="red", linestyle="--", linewidth=1.5, alpha=0.7, label="Null (p=0.5)")
        ax.axhline(0.05, color="green", linestyle=":", linewidth=1.2, alpha=0.7, label="Threshold (p=0.05)")
        ax.set_ylim(0, 1.05)
        ax.set_title(ds_label, fontsize=12, fontweight="bold")
        ax.set_xlabel("Fine-tuning steps (LoRA, clean Alpaca data)")
        ax.set_ylabel("Mean detector p-value\n(↓ lower = stronger watermark)")

        all_steps = df["step"].unique()
        ax.set_xticks(sorted(all_steps))
        ax.grid(axis="y", alpha=0.3, zorder=0)
        ax.legend(fontsize=9)

    fig.suptitle(
        "Fine-tuning Robustness Attack — GRPO vs Base\n"
        "LoRA fine-tuned on clean (non-watermarked) Alpaca data  |  n=200 per checkpoint",
        fontsize=12, fontweight="bold",
    )
    fig.text(0.5, -0.02,
             "H₁: mean < 0.5 (watermark present)  |  Stars on GRPO: * p<0.05  ** p<0.01  *** p<0.001  ns=not sig",
             ha="center", fontsize=9, style="italic")
    plt.tight_layout()
    path = os.path.join(out_dir, "robustness_finetune_scores.png")
    plt.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


def plot_pvals(df: pd.DataFrame, out_dir: str) -> None:
    """-log10(p) vs. fine-tuning steps, GRPO vs Base."""
    SIG_LINE = -np.log10(0.05)

    datasets = [
        ("eli5_val_implicit",   "ELI5 validation"),
        ("alpaca_val_implicit", "Alpaca validation"),
        ("gsm8k_val_implicit",  "GSM8K validation (train domain)"),
    ]
    datasets = [(k, l) for k, l in datasets if f"{k}_mean" in df.columns]
    ncols = len(datasets)
    fig, axes = plt.subplots(1, ncols, figsize=(6.5 * ncols, 5), sharey=False)
    if ncols == 1:
        axes = [axes]

    for ax, (ds_key, ds_label) in zip(axes, datasets):
        for model_name, style in MODEL_STYLES.items():
            sub = df[df["model"] == model_name].sort_values("step")
            if sub.empty:
                continue
            steps = sub["step"].values
            neg_log_ps = -np.log10(sub[f"{ds_key}_p"].clip(lower=1e-300).values)

            ax.plot(steps, neg_log_ps, f"-{style['marker']}",
                    color=style["color"], linewidth=style["lw"],
                    markersize=7, zorder=style["zorder"], label=style["label"])

            for x, nlp, p in zip(steps, neg_log_ps, sub[f"{ds_key}_p"].values):
                label = f"p={p:.2e}" if p > 1e-15 else "p<1e-15"
                ax.text(x, nlp + 0.1, label, ha="center", va="bottom",
                        fontsize=7, rotation=30, color=style["color"])

        ax.axhline(SIG_LINE, color="red", linestyle="--", linewidth=1.5,
                   label="p = 0.05", zorder=5)
        all_steps = df["step"].unique()
        ax.set_xticks(sorted(all_steps))
        ax.set_title(ds_label, fontsize=12, fontweight="bold")
        ax.set_xlabel("Fine-tuning steps")
        ax.set_ylabel("-log10(p-value)")
        ax.legend(fontsize=9)
        ax.grid(axis="y", alpha=0.3, zorder=0)

    fig.suptitle(
        "Watermark Significance Under Fine-tuning Attack  (-log10 p vs null)\n"
        "GRPO vs Base — LoRA fine-tuned on clean Alpaca data",
        fontsize=12, fontweight="bold",
    )
    plt.tight_layout()
    path = os.path.join(out_dir, "robustness_finetune_pvalues.png")
    plt.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


def print_table(df: pd.DataFrame) -> None:
    has_gsm8k = "gsm8k_val_implicit_mean" in df.columns
    print()
    print("=" * (80 + 36 * has_gsm8k))
    print("Fine-tuning Robustness — implicit validation scores")
    for label in df["model"].unique():
        sub = df[df["model"] == label]
        print(f"\n  Model: {label}")
        header = (f"  {'Step':>6}  {'ELI5 mean':>10}  {'ELI5 z':>8}  {'ELI5 p':>12}  "
                  f"{'Alpaca mean':>12}  {'Alpaca z':>8}  {'Alpaca p':>12}")
        if has_gsm8k:
            header += f"  {'GSM8K mean':>11}  {'GSM8K z':>8}  {'GSM8K p':>12}"
        print(header)
        print("  " + "-" * (76 + 36 * has_gsm8k))
        for _, row in sub.iterrows():
            line = (f"  {int(row['step']):>6}  "
                    f"{row['eli5_val_implicit_mean']:>10.4f}  "
                    f"{row['eli5_val_implicit_z']:>8.3f}  "
                    f"{row['eli5_val_implicit_p']:>12.4e}  "
                    f"{row['alpaca_val_implicit_mean']:>12.4f}  "
                    f"{row['alpaca_val_implicit_z']:>8.3f}  "
                    f"{row['alpaca_val_implicit_p']:>12.4e}")
            if has_gsm8k:
                line += (f"  {row.get('gsm8k_val_implicit_mean', float('nan')):>11.4f}  "
                         f"{row.get('gsm8k_val_implicit_z', float('nan')):>8.3f}  "
                         f"{row.get('gsm8k_val_implicit_p', float('nan')):>12.4e}")
            print(line)
    print("=" * (80 + 36 * has_gsm8k))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default=None,
                        help="Path to finetune_robustness_results.csv")
    parser.add_argument("--out-dir", default=OUT_DIR)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    df = load_data(args.csv)
    print_table(df)
    plot_scores(df, args.out_dir)
    plot_pvals(df, args.out_dir)


if __name__ == "__main__":
    main()

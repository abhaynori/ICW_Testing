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

# ── hardcoded fallback ─────────────────────────────────────────────────────────
# Format: model → step → {eli5_val_implicit_mean, ..._std, ..._n,
#                          alpaca_val_implicit_mean, ..._std, ..._n}
HARDCODED: dict[str, dict[int, dict]] = {
    # "grpo": {
    #     0:    {"eli5_val_implicit_mean": ..., "eli5_val_implicit_std": ..., "eli5_val_implicit_n": 200,
    #            "alpaca_val_implicit_mean": ..., "alpaca_val_implicit_std": ..., "alpaca_val_implicit_n": 200},
    # },
    # "base": { ... },
}


def zscore_pval(mean: float, std: float, n: int) -> tuple[float, float]:
    z = mean / (std / np.sqrt(n))
    p = 2 * scipy_stats.norm.sf(abs(z))
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

    for ds in ("eli5", "alpaca"):
        key = f"{ds}_val_implicit"
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
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=False)

    datasets = [
        ("eli5_val_implicit",   "ELI5 validation (implicit)"),
        ("alpaca_val_implicit", "Alpaca validation (implicit)"),
    ]

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

        ax.axhline(0, color="gray", linestyle=":", alpha=0.5)
        ax.set_title(ds_label, fontsize=12, fontweight="bold")
        ax.set_xlabel("Fine-tuning steps (LoRA, clean Alpaca data)")
        ax.set_ylabel("Mean acrostics score (implicit)")

        all_steps = df["step"].unique()
        ax.set_xticks(sorted(all_steps))
        ax.grid(axis="y", alpha=0.3, zorder=0)
        ax.legend(fontsize=9)

    fig.suptitle(
        "Watermark Retention Under Fine-tuning Attack — GRPO vs Base\n"
        "LoRA fine-tuned on clean (non-watermarked) Alpaca data",
        fontsize=12, fontweight="bold",
    )
    fig.text(0.5, -0.02,
             "H0: mean score = 0  |  Stars on GRPO: * p<0.05  ** p<0.01  *** p<0.001  ns=not sig",
             ha="center", fontsize=9, style="italic")
    plt.tight_layout()
    path = os.path.join(out_dir, "robustness_finetune_scores.png")
    plt.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


def plot_pvals(df: pd.DataFrame, out_dir: str) -> None:
    """-log10(p) vs. fine-tuning steps, GRPO vs Base."""
    SIG_LINE = -np.log10(0.05)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=False)
    datasets = [
        ("eli5_val_implicit",   "ELI5 validation"),
        ("alpaca_val_implicit", "Alpaca validation"),
    ]

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
    print()
    print("=" * 80)
    print("Fine-tuning Robustness — implicit validation scores")
    for label in df["model"].unique():
        sub = df[df["model"] == label]
        print(f"\n  Model: {label}")
        print(f"  {'Step':>6}  {'ELI5 mean':>10}  {'ELI5 z':>8}  {'ELI5 p':>12}  "
              f"{'Alpaca mean':>12}  {'Alpaca z':>8}  {'Alpaca p':>12}")
        print("  " + "-" * 76)
        for _, row in sub.iterrows():
            print(f"  {int(row['step']):>6}  "
                  f"{row['eli5_val_implicit_mean']:>10.4f}  "
                  f"{row['eli5_val_implicit_z']:>8.3f}  "
                  f"{row['eli5_val_implicit_p']:>12.4e}  "
                  f"{row['alpaca_val_implicit_mean']:>12.4f}  "
                  f"{row['alpaca_val_implicit_z']:>8.3f}  "
                  f"{row['alpaca_val_implicit_p']:>12.4e}")
    print("=" * 80)


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

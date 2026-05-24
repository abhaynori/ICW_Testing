"""
Plot temperature robustness results for GRPO vs base.

Usage:
  python plot_robustness_temperature.py                        # use hardcoded data
  python plot_robustness_temperature.py --csv <master_csv>     # use parsed CSV
  python plot_robustness_temperature.py --logdir <robustness_logs/gsm8k_...>  # auto-find CSV
"""

from __future__ import annotations

import argparse
import glob
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


# ── hardcoded fallback (from earlier ELI5-trained run) ────────────────────────
TEMPS_ALL = ["t=0.0", "t=0.6", "t=0.7", "t=1.0", "t=1.5"]
TEMP_LABELS = {
    "t=0.0": "Greedy\n(t=0)",
    "t=0.6": "t=0.6",
    "t=0.7": "t=0.7",
    "t=1.0": "t=1.0",
    "t=1.5": "t=1.5",
}
METRICS = ["ELI5 implicit", "ELI5 explicit", "Alpaca implicit", "Alpaca explicit"]
METRIC_KEYS = [
    ("eli5",   "implicit"),
    ("eli5",   "explicit"),
    ("alpaca", "implicit"),
    ("alpaca", "explicit"),
]

HARDCODED_GRPO = {
    "t=0.0": [1.1878, 1.1654, 0.4284, 1.6921],
    "t=0.6": [0.8965, 1.2663, 0.6836, 1.6136],
    "t=0.7": [0.7172, 1.3447, 0.5043, 1.5912],
    "t=1.0": [0.5603, 1.3335, 0.4931, 1.4231],
    "t=1.5": [0.4706, 1.2775, 0.4258, 1.3223],
}
HARDCODED_BASE = {
    "t=0.0": [0.5155, 2.5998, 0.2241, 2.0955],
    "t=0.6": [0.4819, 2.7790, 0.3362, 2.3420],
    "t=0.7": [0.6275, 2.5549, 0.2914, 2.1067],
    "t=1.0": [0.5603, 2.5661, 0.1681, 1.8938],
    "t=1.5": [0.4819, 2.1067, 0.2129, 1.9274],
}


# ── data loading ───────────────────────────────────────────────────────────────

def load_from_csv(csv_path: str) -> tuple[dict, dict] | None:
    """
    Returns (grpo_means, base_means) dicts: temp_label → [eli5_imp, eli5_exp, alpaca_imp, alpaca_exp]
    or None if the CSV doesn't contain temperature data.
    """
    df = pd.read_csv(csv_path)
    temp_df = df[(df.get("phase", pd.Series(["temp"] * len(df))) == "temp") &
                 (df["profile"] == "natural") & (df["split"] == "validation")]
    if temp_df.empty:
        return None

    grpo_means: dict[str, list] = {}
    base_means: dict[str, list] = {}

    for temp_label in TEMPS_ALL:
        sub = temp_df[temp_df["temperature"] == temp_label]
        for store, model in [(grpo_means, "grpo"), (base_means, "base")]:
            row_vals = []
            for ds, mode in METRIC_KEYS:
                match = sub[(sub["model"] == model) & (sub["dataset"] == ds) & (sub["mode"] == mode)]
                row_vals.append(float(match["mean"].values[0]) if not match.empty else float("nan"))
            store[temp_label] = row_vals

    return grpo_means, base_means


# ── plots ──────────────────────────────────────────────────────────────────────

def plot_grouped_bars(grpo: dict, base: dict, out_dir: str) -> None:
    n_temps = len(TEMPS_ALL)
    width = 0.30
    group_gap = 0.8

    fig, ax = plt.subplots(figsize=(17, 6))

    for idx in range(len(METRICS)):
        group_start = idx * (n_temps + group_gap)
        for t_idx, t in enumerate(TEMPS_ALL):
            x_center = group_start + t_idx
            ax.bar(x_center - width / 2, grpo[t][idx], width,
                   label="GRPO" if idx == 0 and t_idx == 0 else "",
                   color="#4c72b0")
            ax.bar(x_center + width / 2, base[t][idx], width,
                   label="Base" if idx == 0 and t_idx == 0 else "",
                   color="#dd8452")

    group_centers = [i * (n_temps + group_gap) + (n_temps - 1) / 2 for i in range(len(METRICS))]
    ax.set_xticks(group_centers)
    ax.set_xticklabels(METRICS, fontsize=10)

    sub_positions, sub_labels = [], []
    for idx in range(len(METRICS)):
        group_start = idx * (n_temps + group_gap)
        for t_idx, t in enumerate(TEMPS_ALL):
            sub_positions.append(group_start + t_idx)
            sub_labels.append(TEMP_LABELS[t])
    ax2 = ax.twiny()
    ax2.set_xlim(ax.get_xlim())
    ax2.set_xticks(sub_positions)
    ax2.set_xticklabels(sub_labels, fontsize=7, rotation=30, ha="left")
    ax2.tick_params(top=False)
    ax2.spines["top"].set_visible(False)

    ax.set_ylabel("Mean acrostics score")
    ax.set_title("Temperature Robustness (validation, natural profile)\nGRPO vs Base")
    ax.legend(ncol=2, fontsize=9)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    path = os.path.join(out_dir, "robustness_temperature_validation.png")
    plt.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


def plot_pval_heatmap(df: pd.DataFrame, out_dir: str) -> None:
    implicit = df[(df["split"] == "validation") & (df["mode"] == "implicit")].copy()
    if implicit.empty:
        return
    implicit["row"] = implicit["model"].str.upper() + " " + implicit["temperature"]
    pivot = implicit.pivot_table(index="row", columns="dataset", values="p", aggfunc="first")
    order = [
        "BASE t=0.0", "BASE t=0.6", "BASE t=0.7", "BASE t=1.0", "BASE t=1.5",
        "GRPO t=0.0", "GRPO t=0.6", "GRPO t=0.7", "GRPO t=1.0", "GRPO t=1.5",
    ]
    pivot = pivot.reindex([r for r in order if r in pivot.index])

    fig, ax = plt.subplots(figsize=(8, max(3, 0.6 * len(pivot) + 1.5)))
    import seaborn as _sns
    _sns.heatmap(pivot, annot=True, fmt=".2e", cmap="viridis_r",
                 cbar_kws={"label": "p-value (vs null score=0)"}, ax=ax)
    ax.set_title("Temperature Robustness: p-values vs null (implicit validation)")
    ax.set_xlabel("Dataset")
    ax.set_ylabel("Model & temperature")
    plt.tight_layout()
    path = os.path.join(out_dir, "robustness_temperature_pvalues.png")
    plt.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


def plot_pval_bars(df: pd.DataFrame, out_dir: str) -> None:
    implicit = df[(df["split"] == "validation") & (df["mode"] == "implicit")].copy()
    if implicit.empty:
        return
    implicit["neg_log10_p"] = -np.log10(implicit["p"].clip(lower=1e-300))
    implicit["dataset_upper"] = implicit["dataset"].str.upper()
    implicit["model_upper"] = implicit["model"].str.upper()

    datasets = [d for d in ["ELI5", "ALPACA", "GSM8K"] if d in implicit["dataset_upper"].values]
    if not datasets:
        return
    fig, axes = plt.subplots(1, len(datasets), figsize=(6.5 * len(datasets), 5), sharey=False)
    if len(datasets) == 1:
        axes = [axes]

    temp_order = [t for t in TEMPS_ALL if t in implicit["temperature"].values]
    for ax, dataset in zip(axes, datasets):
        subset = implicit[implicit["dataset_upper"] == dataset].copy()
        present_temps = [t for t in temp_order if t in subset["temperature"].values]
        sns.barplot(data=subset, x="temperature", y="neg_log10_p", hue="model_upper",
                    order=present_temps, ax=ax)
        ax.axhline(-np.log10(0.05), color="red", linestyle="--", linewidth=1.5, label="p=0.05")
        ax.set_title(f"{dataset} (implicit validation)")
        ax.set_xlabel("Temperature")
        ax.set_ylabel("-log10(p-value)")
        ax.grid(True, axis="y", alpha=0.3)
    axes[0].legend(title="Model", fontsize=8)
    for ax in axes[1:]:
        if ax.get_legend():
            ax.get_legend().remove()
    fig.suptitle("Temperature Robustness: significance vs null (implicit)")
    plt.tight_layout()
    path = os.path.join(out_dir, "robustness_temperature_pvalues_bars.png")
    plt.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


# ── entrypoint ─────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=None,
                    help="Path to robustness_master.csv from parse_robustness_results.py")
    ap.add_argument("--logdir", default=None,
                    help="Robustness log dir — auto-find robustness_master.csv inside it")
    ap.add_argument("--out-dir", default="outputs/utility_plots")
    args = ap.parse_args()

    out_dir = args.out_dir
    os.makedirs(out_dir, exist_ok=True)

    csv_path = args.csv
    if csv_path is None and args.logdir:
        candidates = sorted(glob.glob(f"{args.logdir}/**/robustness_master.csv", recursive=True))
        if candidates:
            csv_path = candidates[-1]
            print(f"Auto-found CSV: {csv_path}")

    grpo, base = HARDCODED_GRPO, HARDCODED_BASE
    df_pval = pd.DataFrame()

    if csv_path and os.path.exists(csv_path):
        result = load_from_csv(csv_path)
        if result:
            grpo, base = result
            print(f"Using data from CSV: {csv_path}")
        df_pval = pd.read_csv(csv_path)
        df_pval = df_pval[df_pval.get("phase", pd.Series(["temp"] * len(df_pval))) == "temp"]
    else:
        print("No CSV found — using hardcoded data.")

    plot_grouped_bars(grpo, base, out_dir)
    if not df_pval.empty:
        plot_pval_heatmap(df_pval, out_dir)
        plot_pval_bars(df_pval, out_dir)
    else:
        print("Skipping p-value plots (no CSV data).")


if __name__ == "__main__":
    main()

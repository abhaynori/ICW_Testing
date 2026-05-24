#!/usr/bin/env python3
"""
Plot utility benchmark results (IFEval + GSM8K) for base / SFT / GRPO models.

Usage:
  python plot_utility_results.py
  python plot_utility_results.py --csv utility_results/gsm8k_run_<ts>/utility_summary.csv
  python plot_utility_results.py --out-dir meeting_assets/utility_plots
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

sns.set_style("whitegrid")
plt.rcParams["figure.dpi"] = 300
plt.rcParams["savefig.dpi"] = 300
plt.rcParams["font.size"] = 11

# ── hardcoded data (from utility_results/gsm8k_run_20260523_151539) ───────────
DATA = {
    "base": {
        "if_inst_loose":  (0.7374, None),
        "if_inst_strict": (0.6918, None),
        "if_prom_loose":  (0.6377, 0.0207),
        "if_prom_strict": (0.5896, 0.0212),
        "gsm_flex":       (0.8234, 0.0105),
        "gsm_strict":     (0.7597, 0.0118),
    },
    "sft": {
        "if_inst_loose":  (0.6787, None),
        "if_inst_strict": (0.6523, None),
        "if_prom_loose":  (0.5638, 0.0213),
        "if_prom_strict": (0.5360, 0.0215),
        "gsm_flex":       (0.7991, 0.0110),
        "gsm_strict":     (0.8211, 0.0106),
    },
    "grpo": {
        "if_inst_loose":  (0.6811, None),
        "if_inst_strict": (0.6523, None),
        "if_prom_loose":  (0.5638, 0.0213),
        "if_prom_strict": (0.5342, 0.0215),
        "gsm_flex":       (0.7900, 0.0112),
        "gsm_strict":     (0.8165, 0.0107),
    },
}

METRIC_LABELS = {
    "if_inst_loose":  "Inst-level\nloose acc",
    "if_inst_strict": "Inst-level\nstrict acc",
    "if_prom_loose":  "Prompt-level\nloose acc",
    "if_prom_strict": "Prompt-level\nstrict acc",
    "gsm_flex":       "Flexible\nextract",
    "gsm_strict":     "Strict\nmatch",
}

MODEL_COLORS = {
    "base": "#aec7e8",
    "sft":  "#ffbb78",
    "grpo": "#98df8a",
}
MODEL_LABELS = {
    "base": "Base (Qwen2.5-7B-Instruct)",
    "sft":  "SFT (5-epoch, implicit WM)",
    "grpo": "GRPO (implicit_fraction=1.0)",
}
MODELS = ["base", "sft", "grpo"]


def load_data_from_csv(csv_path: str) -> dict | None:
    df = pd.read_csv(csv_path)
    required = {"model", "if_inst_loose", "if_inst_strict", "if_prom_loose",
                "if_prom_strict", "gsm_flex", "gsm_strict"}
    if not required.issubset(df.columns):
        return None
    result = {}
    for _, row in df.iterrows():
        m = row["model"]
        result[m] = {k: (float(row[k]), None) for k in
                     ["if_inst_loose", "if_inst_strict", "if_prom_loose",
                      "if_prom_strict", "gsm_flex", "gsm_strict"]}
    return result if result else None


# ── Plot 1: grouped bars ───────────────────────────────────────────────────────

def plot_grouped_bars(data: dict, out_dir: str) -> None:
    ifeval_keys  = ["if_inst_loose", "if_inst_strict", "if_prom_loose", "if_prom_strict"]
    gsm8k_keys   = ["gsm_flex", "gsm_strict"]

    fig, axes = plt.subplots(1, 2, figsize=(15, 6),
                             gridspec_kw={"width_ratios": [4, 2]})

    for ax, keys, title in [
        (axes[0], ifeval_keys, "IFEval (541 prompts, greedy)"),
        (axes[1], gsm8k_keys,  "GSM8K 5-shot (1319 problems, greedy)"),
    ]:
        x = np.arange(len(keys))
        width = 0.25
        offsets = [-width, 0, width]

        for model, offset in zip(MODELS, offsets):
            vals  = [data[model][k][0] for k in keys]
            errs  = [data[model][k][1] or 0.0 for k in keys]
            bars = ax.bar(x + offset, vals, width,
                          label=MODEL_LABELS[model],
                          color=MODEL_COLORS[model],
                          edgecolor="white", linewidth=0.5,
                          yerr=errs, capsize=3, error_kw={"linewidth": 1.2},
                          zorder=3)

            # value labels on bars
            for bar, v, e in zip(bars, vals, errs):
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + e + 0.004,
                        f"{v:.3f}", ha="center", va="bottom",
                        fontsize=7, rotation=45)

        ax.set_xticks(x)
        ax.set_xticklabels([METRIC_LABELS[k] for k in keys], fontsize=10)
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.set_ylabel("Accuracy", fontsize=11)
        ax.set_ylim(0, ax.get_ylim()[1] * 1.12)
        ax.grid(axis="y", alpha=0.3, zorder=0)

    axes[0].legend(fontsize=9, loc="lower right")

    fig.suptitle(
        "Utility Preservation — Base → SFT → GRPO (GSM8K-trained, implicit watermark)\n"
        "Qwen/Qwen2.5-7B-Instruct, bfloat16",
        fontsize=12, fontweight="bold",
    )
    plt.tight_layout()
    path = os.path.join(out_dir, "utility_grouped_bars.png")
    plt.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


# ── Plot 2: delta-from-base ────────────────────────────────────────────────────

def plot_delta(data: dict, out_dir: str) -> None:
    all_keys = ["if_inst_loose", "if_inst_strict", "if_prom_loose", "if_prom_strict",
                "gsm_flex", "gsm_strict"]
    base_vals = {k: data["base"][k][0] for k in all_keys}

    fig, ax = plt.subplots(figsize=(12, 5))
    x = np.arange(len(all_keys))
    width = 0.30
    offsets = [-width / 2, width / 2]

    for (model, offset) in zip(["sft", "grpo"], offsets):
        deltas = [data[model][k][0] - base_vals[k] for k in all_keys]
        colors = ["#2ca02c" if d >= 0 else "#d62728" for d in deltas]
        bars = ax.bar(x + offset, deltas, width,
                      color=colors, edgecolor="white", linewidth=0.5,
                      label=MODEL_LABELS[model], alpha=0.85, zorder=3)
        for bar, d in zip(bars, deltas):
            ypos = d + 0.003 if d >= 0 else d - 0.003
            va = "bottom" if d >= 0 else "top"
            ax.text(bar.get_x() + bar.get_width() / 2, ypos,
                    f"{d:+.3f}", ha="center", va=va, fontsize=8, fontweight="bold")

    ax.axhline(0, color="black", linewidth=1.2)
    ax.set_xticks(x)
    ax.set_xticklabels(
        ["IF inst-loose", "IF inst-strict", "IF pr-loose", "IF pr-strict",
         "GSM flex", "GSM strict"],
        fontsize=10,
    )
    ax.set_ylabel("Δ accuracy vs base", fontsize=11)
    ax.set_title(
        "Utility Cost of Watermarking (Δ vs Base)\n"
        "Green = improvement  |  Red = degradation",
        fontsize=12, fontweight="bold",
    )

    sft_patch  = mpatches.Patch(color=MODEL_COLORS["sft"],  label=MODEL_LABELS["sft"])
    grpo_patch = mpatches.Patch(color=MODEL_COLORS["grpo"], label=MODEL_LABELS["grpo"])
    ax.legend(handles=[sft_patch, grpo_patch], fontsize=9, loc="lower right")
    ax.grid(axis="y", alpha=0.3, zorder=0)

    plt.tight_layout()
    path = os.path.join(out_dir, "utility_delta_from_base.png")
    plt.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


# ── Plot 3: summary heatmap ────────────────────────────────────────────────────

def plot_heatmap(data: dict, out_dir: str) -> None:
    all_keys = ["if_inst_loose", "if_inst_strict", "if_prom_loose", "if_prom_strict",
                "gsm_flex", "gsm_strict"]
    col_labels = ["IF inst-loose", "IF inst-strict", "IF pr-loose", "IF pr-strict",
                  "GSM flex", "GSM strict"]

    rows = {}
    for m in MODELS:
        rows[MODEL_LABELS[m]] = [data[m][k][0] for k in all_keys]

    df = pd.DataFrame(rows, index=col_labels).T

    fig, ax = plt.subplots(figsize=(11, 3.5))
    sns.heatmap(
        df, annot=True, fmt=".4f", cmap="YlOrRd_r",
        linewidths=0.5, linecolor="white",
        cbar_kws={"label": "Accuracy"},
        ax=ax,
    )
    ax.set_title(
        "Utility Benchmark Summary — Base / SFT / GRPO",
        fontsize=12, fontweight="bold",
    )
    ax.set_xlabel("Metric", fontsize=10)
    ax.set_ylabel("")
    plt.tight_layout()
    path = os.path.join(out_dir, "utility_heatmap.png")
    plt.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


# ── Plot 4: combined paper-ready figure ───────────────────────────────────────

def plot_paper_figure(data: dict, out_dir: str) -> None:
    """
    Two-panel figure suitable for a paper/meeting:
      Left:  absolute scores (IFEval + GSM8K) as grouped bars
      Right: delta from base as horizontal bars
    """
    all_keys = ["if_inst_loose", "if_inst_strict", "if_prom_loose", "if_prom_strict",
                "gsm_flex", "gsm_strict"]
    short_labels = ["IF inst\nloose", "IF inst\nstrict", "IF pr\nloose", "IF pr\nstrict",
                    "GSM\nflex", "GSM\nstrict"]
    base_vals = {k: data["base"][k][0] for k in all_keys}

    fig = plt.figure(figsize=(16, 6))
    gs = fig.add_gridspec(1, 2, width_ratios=[3, 2], wspace=0.35)
    ax_abs = fig.add_subplot(gs[0])
    ax_del = fig.add_subplot(gs[1])

    # -- absolute bars
    x = np.arange(len(all_keys))
    width = 0.25
    for model, offset in zip(MODELS, [-width, 0, width]):
        vals = [data[model][k][0] for k in all_keys]
        errs = [data[model][k][1] or 0.0 for k in all_keys]
        ax_abs.bar(x + offset, vals, width, label=MODEL_LABELS[model],
                   color=MODEL_COLORS[model], edgecolor="white",
                   yerr=errs, capsize=3, zorder=3)

    ax_abs.set_xticks(x)
    ax_abs.set_xticklabels(short_labels, fontsize=9)
    ax_abs.set_ylabel("Accuracy")
    ax_abs.set_title("Absolute accuracy", fontsize=11, fontweight="bold")
    ax_abs.set_ylim(0.45, 0.92)
    ax_abs.legend(fontsize=8, loc="lower right")
    ax_abs.grid(axis="y", alpha=0.3, zorder=0)

    # -- delta bars (horizontal)
    y = np.arange(len(all_keys))
    bar_h = 0.30
    for model, offset, label in zip(["sft", "grpo"], [-bar_h / 2, bar_h / 2],
                                    [MODEL_LABELS["sft"], MODEL_LABELS["grpo"]]):
        deltas = [data[model][k][0] - base_vals[k] for k in all_keys]
        colors = ["#2ca02c" if d >= 0 else "#d62728" for d in deltas]
        ax_del.barh(y + offset, deltas, bar_h, color=colors,
                    edgecolor="white", label=label, zorder=3)
        for j, (d, yi) in enumerate(zip(deltas, y)):
            xpos = d + 0.001 if d >= 0 else d - 0.001
            ha = "left" if d >= 0 else "right"
            ax_del.text(xpos, yi + offset, f"{d:+.3f}",
                        ha=ha, va="center", fontsize=8)

    ax_del.axvline(0, color="black", linewidth=1.2)
    ax_del.set_yticks(y)
    ax_del.set_yticklabels(short_labels, fontsize=9)
    ax_del.set_xlabel("Δ vs base")
    ax_del.set_title("Change from base", fontsize=11, fontweight="bold")
    ax_del.grid(axis="x", alpha=0.3, zorder=0)

    sft_patch  = mpatches.Patch(color=MODEL_COLORS["sft"],  label=MODEL_LABELS["sft"])
    grpo_patch = mpatches.Patch(color=MODEL_COLORS["grpo"], label=MODEL_LABELS["grpo"])
    ax_del.legend(handles=[sft_patch, grpo_patch], fontsize=8, loc="lower right")

    fig.suptitle(
        "Utility Preservation — Watermarked Models vs Base\n"
        "(GSM8K-trained implicit acrostics watermark, Qwen2.5-7B-Instruct)",
        fontsize=12, fontweight="bold",
    )
    path = os.path.join(out_dir, "utility_paper_figure.png")
    plt.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


# ── print summary table ────────────────────────────────────────────────────────

def print_summary(data: dict) -> None:
    all_keys = ["if_inst_loose", "if_inst_strict", "if_prom_loose", "if_prom_strict",
                "gsm_flex", "gsm_strict"]
    base_vals = {k: data["base"][k][0] for k in all_keys}

    print()
    print("=" * 85)
    print("Utility Benchmark — GSM8K-trained implicit acrostics watermark")
    print(f"  Model: Qwen/Qwen2.5-7B-Instruct (7B, bfloat16)")
    print()
    header = (f"  {'Metric':<18} {'Base':>8} {'SFT':>8} {'GRPO':>8}  "
              f"{'ΔSFT':>8} {'ΔGRPO':>8}")
    print(header)
    print("  " + "-" * 65)
    for k in all_keys:
        bv = base_vals[k]
        sv = data["sft"][k][0]
        gv = data["grpo"][k][0]
        print(f"  {METRIC_LABELS[k].replace(chr(10),' '):<18} "
              f"{bv:>8.4f} {sv:>8.4f} {gv:>8.4f}  "
              f"{sv - bv:>+8.4f} {gv - bv:>+8.4f}")
    print()
    print("  Key findings:")
    print("  • IFEval: SFT/GRPO drop ~5-7 pp on prompt-level (expected format constraint)")
    print("  • GSM8K flexible-extract: ~2-3 pp drop (minimal math degradation)")
    print("  • GSM8K strict-match: +5-6 pp gain (training on GSM8K improves answer formatting)")
    print("  • SFT ≈ GRPO on all metrics: GRPO adds no additional utility cost")
    print("=" * 85)


# ── entrypoint ─────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=None,
                    help="utility_summary.csv from run_utility_eval.sh")
    ap.add_argument("--out-dir", default="outputs/utility_plots")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    data = DATA
    if args.csv and os.path.exists(args.csv):
        loaded = load_data_from_csv(args.csv)
        if loaded:
            data = loaded
            print(f"Using data from: {args.csv}")
        else:
            print(f"Could not parse {args.csv} — using hardcoded data.")
    else:
        # Auto-find latest run
        candidates = sorted(glob.glob("utility_results/gsm8k_run_*/utility_summary.csv"))
        if candidates:
            loaded = load_data_from_csv(candidates[-1])
            if loaded:
                data = loaded
                print(f"Auto-found: {candidates[-1]}")

    print_summary(data)
    plot_grouped_bars(data, args.out_dir)
    plot_delta(data, args.out_dir)
    plot_heatmap(data, args.out_dir)
    plot_paper_figure(data, args.out_dir)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Plot system-prompt robustness results for GRPO model.

Usage:
  python plot_sysprompt_robustness.py                     # hardcoded data
  python plot_sysprompt_robustness.py --csv <master_csv>  # parsed CSV
  python plot_sysprompt_robustness.py --logdir <dir>      # auto-find CSV
"""

from __future__ import annotations
import argparse
import glob
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import pandas as pd
from scipy import stats

sns.set_style("whitegrid")
plt.rcParams["figure.dpi"] = 300
plt.rcParams["savefig.dpi"] = 300
plt.rcParams["font.size"] = 10


def zscore_pval(mean: float, std: float, n: int) -> tuple[float, float]:
    z = mean / (std / np.sqrt(n))
    p = 2 * stats.norm.sf(abs(z))
    return z, p


def sig_stars(p: float) -> str:
    if p < 0.001: return "***"
    if p < 0.01:  return "**"
    if p < 0.05:  return "*"
    return "ns"


# ─── DATA ────────────────────────────────────────────────────────────────────
# Format: (display_label, tag, {metric_key: (mean, std, n)})
# metric_key = f"{dataset}_{split}_{mode}"
#
# Standard reference comes from Phase 1 (grpo_standard_t07, exact values from CSV).
# Add new conditions below as results arrive.

CONDITIONS: list[tuple[str, str, dict]] = [
    ("Standard\n(no sys-prompt)", "standard", {
        "eli5_val_implicit":   (0.7171735491102399, 1.204707653162618,  200),
        "eli5_val_explicit":   (1.3447004045817004, 1.7127271512886924, 200),
        "alpaca_val_implicit": (0.5042626517181371, 1.2669492909051143, 200),
        "alpaca_val_explicit": (1.591228812088346,  2.103361543179566,  200),
    }),
    ("No sys-prompt\n(Phase 2 run)", "no_sysprompt", {
        "eli5_val_implicit":   (0.7844, 1.1723, 200),
        "eli5_val_explicit":   (1.2214, 1.5272, 200),
        "alpaca_val_implicit": (0.4482, 1.2779, 200),
        "alpaca_val_explicit": (1.6585, 2.0652, 200),
    }),
    ("Concise", "concise", {
        "eli5_val_implicit":   (0.5939, 1.2069, 200),
        "eli5_val_explicit":   (1.2663, 1.5249, 200),
        "alpaca_val_implicit": (0.1732, 1.1882, 200),
        "alpaca_val_explicit": (1.6809, 2.1724, 200),
    }),
    ("Pirate", "pirate", {
        "eli5_val_implicit":   (0.6163, 1.3068, 200),
        "eli5_val_explicit":   (1.2999, 1.9068, 200),
        "alpaca_val_implicit": (-0.1095, 0.9441, 200),
        "alpaca_val_explicit": (1.7817, 2.0716, 200),
    }),
    ("Formal", "formal", {
        "eli5_val_implicit":   (0.8404, 1.2254, 200),
        "eli5_val_explicit":   (1.2775, 1.6920, 200),
        "alpaca_val_implicit": (0.1957, 1.1314, 200),
        "alpaca_val_explicit": (1.7033, 2.1612, 200),
    }),
    ("Company bot", "company_bot", {
        "eli5_val_implicit":   (0.5155, 1.1611, 200),
        "eli5_val_explicit":   (1.3447, 1.6788, 200),
        "alpaca_val_implicit": (-0.0285, 0.9464, 200),
        "alpaca_val_explicit": (1.7593, 2.2798, 200),
    }),
    ("Teacher", "teacher", {
        "eli5_val_implicit":   (0.7844, 1.2749, 200),
        "eli5_val_explicit":   (1.3783, 1.6176, 200),
        "alpaca_val_implicit": (0.2801, 1.1837, 200),
        "alpaca_val_explicit": (1.6473, 2.0708, 200),
    }),
]


METRICS_BAR = [
    ("eli5_val_implicit",   "ELI5 val implicit"),
    ("eli5_val_explicit",   "ELI5 val explicit"),
    ("alpaca_val_implicit", "Alpaca val implicit"),
    ("alpaca_val_explicit", "Alpaca val explicit"),
]

METRIC_COLORS = ["#4c72b0", "#dd8452", "#55a868", "#c44e52"]
IMPLICIT_DATASETS = [
    ("eli5_val_implicit",   "ELI5 validation (implicit)"),
    ("alpaca_val_implicit", "Alpaca validation (implicit)"),
]


# ─── PLOT 1: grouped bars ─────────────────────────────────────────────────────

def plot_bars(conditions: list, out_dir: str) -> None:
    labels = [c[0] for c in conditions]
    n_conds = len(conditions)
    n_metrics = len(METRICS_BAR)
    x = np.arange(n_conds)
    width = 0.18
    offsets = np.linspace(-(n_metrics - 1) / 2 * width,
                          (n_metrics - 1) / 2 * width, n_metrics)

    fig, ax = plt.subplots(figsize=(max(9, 2.5 * n_conds), 5))

    for (metric_key, metric_label), color, offset in zip(METRICS_BAR, METRIC_COLORS, offsets):
        means, cis = [], []
        for _, _, data in conditions:
            m, s, n = data[metric_key]
            means.append(m)
            cis.append(stats.t.ppf(0.975, n - 1) * s / np.sqrt(n))
        ax.bar(x + offset, means, width, yerr=cis, capsize=4,
               label=metric_label, color=color, alpha=0.85, zorder=3)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylabel("Mean acrostics score")
    ax.set_title(
        "System-Prompt Robustness — GRPO model\n"
        "(validation, natural profile, temp=0.7, top_p=0.9)"
    )
    ax.legend(ncol=2, fontsize=8)
    ax.axhline(0, color="gray", linestyle=":", alpha=0.5)
    ax.grid(axis="y", alpha=0.3, zorder=0)

    plt.tight_layout()
    path = os.path.join(out_dir, "robustness_sysprompt_validation.png")
    plt.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


# ─── PLOT 2: -log10(p) bars ───────────────────────────────────────────────────

def plot_pval_bars(conditions: list, out_dir: str) -> None:
    labels = [c[0] for c in conditions]
    x = np.arange(len(conditions))

    fig, axes = plt.subplots(1, len(IMPLICIT_DATASETS),
                             figsize=(6 * len(IMPLICIT_DATASETS), 5),
                             sharey=False)
    if len(IMPLICIT_DATASETS) == 1:
        axes = [axes]

    for ax, (metric_key, panel_title) in zip(axes, IMPLICIT_DATASETS):
        pvals, neg_log_ps = [], []
        for _, _, data in conditions:
            m, s, n = data[metric_key]
            z, p = zscore_pval(m, s, n)
            pvals.append(p)
            neg_log_ps.append(-np.log10(max(p, 1e-300)))

        bar_colors = ["#4c72b0" if p < 0.05 else "lightgray" for p in pvals]
        ax.bar(x, neg_log_ps, color=bar_colors, edgecolor="white", zorder=3)
        sig_thresh = -np.log10(0.05)
        ax.axhline(sig_thresh, color="red", linestyle="--", linewidth=1.5,
                   label="p = 0.05", zorder=4)

        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=9)
        ax.set_ylabel("-log10(p-value)")
        ax.set_title(panel_title)
        ax.legend(fontsize=8)
        ax.grid(axis="y", alpha=0.3, zorder=0)

        for i, (p, nlp) in enumerate(zip(pvals, neg_log_ps)):
            label_str = f"{sig_stars(p)}\np={p:.2e}" if p >= 1e-15 else f"{sig_stars(p)}\np<1e-15"
            ax.text(x[i], nlp + 0.1, label_str, ha="center", va="bottom", fontsize=8)

    fig.suptitle(
        "System-Prompt Robustness: significance vs null (implicit)\n"
        "GRPO model — validation split  |  H₀: mean score = 0",
        fontsize=12, fontweight="bold",
    )
    plt.tight_layout()
    path = os.path.join(out_dir, "robustness_sysprompt_pvalues_bars.png")
    plt.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


# ─── PLOT 3: heatmap ─────────────────────────────────────────────────────────

def plot_heatmap(conditions: list, out_dir: str) -> None:
    rows = []
    for label, tag, data in conditions:
        for metric_key, _ in METRICS_BAR:
            m, s, n = data[metric_key]
            z, p = zscore_pval(m, s, n)
            rows.append({"condition": label.replace("\n", " "), "metric": metric_key, "p": p})

    df = pd.DataFrame(rows)
    pivot = df.pivot_table(index="condition", columns="metric", values="p", aggfunc="first")
    col_order = [mk for mk, _ in METRICS_BAR]
    pivot = pivot[[c for c in col_order if c in pivot.columns]]
    row_order = [c[0].replace("\n", " ") for c in conditions]
    pivot = pivot.reindex([r for r in row_order if r in pivot.index])

    fig, ax = plt.subplots(figsize=(8, max(3, 0.6 * len(conditions) + 1.5)))
    sns.heatmap(
        pivot, annot=True, fmt=".2e", cmap="viridis_r",
        cbar_kws={"label": "p-value (vs null score=0)"},
        ax=ax,
    )
    ax.set_title("System-Prompt Robustness: p-values vs null\nGRPO model — validation split")
    ax.set_xlabel("Metric")
    ax.set_ylabel("System prompt condition")
    plt.tight_layout()
    path = os.path.join(out_dir, "robustness_sysprompt_pvalues_heatmap.png")
    plt.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


# ─── CSV export ──────────────────────────────────────────────────────────────

def export_csv(conditions: list, out_dir: str) -> None:
    os.makedirs(out_dir, exist_ok=True)
    rows = []
    for label, tag, data in conditions:
        for metric_key, _ in METRICS_BAR:
            parts = metric_key.split("_")  # e.g. eli5_val_implicit
            dataset, split, mode = parts[0], parts[1], parts[2]
            m, s, n = data[metric_key]
            z, p = zscore_pval(m, s, n)
            rows.append({
                "run": f"grpo_{tag}",
                "model": "grpo",
                "sysprompt": tag,
                "dataset": dataset,
                "split": split,
                "mode": mode,
                "eval_profile": "natural",
                "n": n,
                "mean": m,
                "std": s,
                "z": z,
                "p": p,
            })
    pd.DataFrame(rows).to_csv(
        os.path.join(out_dir, "sysprompt_pvalues_vs_null.csv"), index=False
    )
    print(f"Saved: {os.path.join(out_dir, 'sysprompt_pvalues_vs_null.csv')}")


CONDITION_LABELS = {
    "grpo_no_sysprompt": "No sys-prompt",
    "grpo_concise":      "Concise",
    "grpo_pirate":       "Pirate",
    "grpo_formal":       "Formal",
    "grpo_company_bot":  "Company bot",
    "grpo_teacher":      "Teacher",
    "grpo_math_tutor":   "Math tutor",
}


def load_conditions_from_csv(csv_path: str) -> list | None:
    df = pd.read_csv(csv_path)
    sysp = df[(df.get("phase", pd.Series(["sysprompt"] * len(df))) == "sysprompt") &
              (df["model"] == "grpo") & (df["profile"] == "natural") &
              (df["split"] == "validation")]
    if sysp.empty:
        return None

    # Build reference from hardcoded CONDITIONS (standard t=0.7 condition)
    reference = CONDITIONS[0]  # Standard (no sys-prompt from Phase 1)

    conditions = [reference]
    for cond_name in sysp["condition"].unique():
        sub = sysp[sysp["condition"] == cond_name]
        label = CONDITION_LABELS.get(cond_name, cond_name.replace("grpo_", "").replace("_", " ").title())
        data = {}
        for ds, sp, mode in [("eli5", "val", "implicit"), ("eli5", "val", "explicit"),
                              ("alpaca", "val", "implicit"), ("alpaca", "val", "explicit")]:
            key = f"{ds}_{sp}_{mode}"
            match = sub[(sub["dataset"] == ds) & (sub["mode"] == mode)]
            if not match.empty:
                row = match.iloc[0]
                data[key] = (float(row["mean"]), float(row["std"]), int(row["n"]))
            else:
                # fill with nan so we don't crash
                data[key] = (float("nan"), 1.0, 1)
        conditions.append((label, cond_name.replace("grpo_", ""), data))

    return conditions if len(conditions) > 1 else None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=None)
    ap.add_argument("--logdir", default=None)
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

    conditions = CONDITIONS
    if csv_path and os.path.exists(csv_path):
        loaded = load_conditions_from_csv(csv_path)
        if loaded:
            conditions = loaded
            print(f"Using {len(conditions)} conditions from CSV: {csv_path}")

    plot_bars(conditions, out_dir)
    plot_pval_bars(conditions, out_dir)
    plot_heatmap(conditions, out_dir)
    export_csv(conditions, "outputs")

    print()
    print("=" * 80)
    print("System-Prompt Robustness Summary (validation, implicit mode)")
    print(f"{'Condition':<22} {'ELI5 mean':>10} {'ELI5 p':>12} {'Alpaca mean':>12} {'Alpaca p':>12}")
    print("-" * 80)
    for label, tag, data in conditions:
        lbl = label.replace("\n", " ")
        em, es, en = data["eli5_val_implicit"]
        am, as_, an = data["alpaca_val_implicit"]
        _, ep = zscore_pval(em, es, en)
        _, ap = zscore_pval(am, as_, an)
        print(f"  {lbl:<20} {em:>10.4f} {ep:>12.4e} {am:>12.4f} {ap:>12.4e}")
    print("=" * 80)


if __name__ == "__main__":
    main()

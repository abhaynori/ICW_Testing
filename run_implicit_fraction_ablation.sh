#!/bin/bash
# ============================================================================
# Implicit-fraction ablation sweep
#
# Starting from the existing GSM8K SFT checkpoint, run GRPO with
# implicit_fraction ∈ {0.0, 0.3, 0.5, 0.7, 1.0} and evaluate each.
#
# Answers the key reviewer question: "How much does GRPO (vs pure SFT) matter,
# and how does the implicit fraction drive the watermark signal?"
#
# Usage:
#   bash run_implicit_fraction_ablation.sh
# ============================================================================
set -euo pipefail

# ── Paths — reuse the GSM8K SFT checkpoint ───────────────────────────────────
BASE_MODEL="Qwen/Qwen2.5-7B-Instruct"
RUN_ROOT="rerun_acrostics_42_20260522_235846"
SFT_MODEL="${RUN_ROOT}/sft_models/sft_acrostics_full_20260523_101707/final_model"

METHOD=acrostics
SECRET=SECRET
SEED=42
TRAIN_DATASET=gsm8k
EVAL_DATASETS="gsm8k,eli5,alpaca"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
ABLATION_ROOT="ablation_implicit_fraction_${TIMESTAMP}"
mkdir -p "$ABLATION_ROOT"

echo "========================================================"
echo "Implicit-fraction ablation — ${ABLATION_ROOT}"
echo "SFT checkpoint: ${SFT_MODEL}"
echo "========================================================"

# ── First record SFT-only (fraction=N/A, no GRPO) ────────────────────────────
echo ""
echo ">>> Evaluating SFT checkpoint (no GRPO, baseline for ablation)"
python grpo_train.py \
    --eval-only "$SFT_MODEL" \
    --model full \
    --method "$METHOD" \
    --secret-sequence "$SECRET" \
    --train-dataset "$TRAIN_DATASET" \
    --samples 200 \
    --eval-datasets "$EVAL_DATASETS" \
    --eval-splits validation \
    --eval-samples 200 \
    --eval-profiles natural \
    --eval-modes implicit \
    --gen-batch-size 4 \
    --max-new-tokens 200 \
    --seed "$SEED" \
    --eval-output-dir "$ABLATION_ROOT/fraction_sft_only"

# ── Sweep over implicit fractions ────────────────────────────────────────────
for FRAC in 0.0 0.3 0.5 0.7 1.0; do
    FRAC_TAG=$(echo "$FRAC" | tr '.' '_')
    OUTDIR="$ABLATION_ROOT/fraction_${FRAC_TAG}"
    mkdir -p "$OUTDIR/grpo_model"

    echo ""
    echo "========================================================"
    echo ">>> implicit_fraction = ${FRAC}"
    echo "========================================================"

    # GRPO from SFT checkpoint
    python grpo_train.py \
        --model full \
        --method "$METHOD" \
        --secret-sequence "$SECRET" \
        --warm-start-model "$SFT_MODEL" \
        --use-lora \
        --samples 200 \
        --epochs 3 \
        --batch-size 4 \
        --learning-rate 1e-5 \
        --num-generations 4 \
        --implicit-fraction "$FRAC" \
        --beta 0.04 \
        --allow-implicit-reference \
        --max-new-tokens 200 \
        --gen-batch-size 4 \
        --eval-splits "" \
        --train-dataset "$TRAIN_DATASET" \
        --seed "$SEED" \
        --output-dir "$OUTDIR/grpo_model"

    GRPO_MODEL=$(find "$OUTDIR/grpo_model" -type d -path "*/acrostics_full_*/final_model" | sort | tail -n 1)
    if [ -z "$GRPO_MODEL" ]; then
        echo "WARNING: no GRPO model found for fraction=${FRAC}, skipping eval" >&2
        continue
    fi

    # Eval
    python grpo_train.py \
        --eval-only "$GRPO_MODEL" \
        --model full \
        --method "$METHOD" \
        --secret-sequence "$SECRET" \
        --train-dataset "$TRAIN_DATASET" \
        --samples 200 \
        --eval-datasets "$EVAL_DATASETS" \
        --eval-splits validation \
        --eval-samples 200 \
        --eval-profiles natural \
        --eval-modes implicit \
        --gen-batch-size 4 \
        --max-new-tokens 200 \
        --seed "$SEED" \
        --eval-output-dir "$OUTDIR/eval"
done

# ── Summarise all fractions into one CSV ─────────────────────────────────────
python - <<PYEOF
import json, glob, csv, os, math
from pathlib import Path

root = Path("${ABLATION_ROOT}")
rows = []

def load_eval_json(path):
    try:
        with open(path) as f:
            return json.load(f).get("summary", {})
    except Exception:
        return {}

# SFT-only
for jf in sorted(root.glob("fraction_sft_only/eval_*_validation_implicit_natural.json")):
    s = load_eval_json(jf)
    ds = s.get("dataset", jf.stem.split("_")[1])
    rows.append({"fraction": "sft_only", "dataset": ds,
                 "mean": s.get("mean_score", float("nan")),
                 "std":  s.get("std_score", float("nan")),
                 "n":    s.get("num_samples", 0)})

# GRPO fractions
for frac_tag in ["0_0", "0_3", "0_5", "0_7", "1_0"]:
    frac_label = frac_tag.replace("_", ".")
    for jf in sorted(root.glob(f"fraction_{frac_tag}/eval/eval_*_validation_implicit_natural.json")):
        s = load_eval_json(jf)
        ds = s.get("dataset", jf.stem.split("_")[1])
        rows.append({"fraction": frac_label, "dataset": ds,
                     "mean": s.get("mean_score", float("nan")),
                     "std":  s.get("std_score", float("nan")),
                     "n":    s.get("num_samples", 0)})

# Compute z and p
import scipy.stats as st
for r in rows:
    m, s, n = r["mean"], r["std"], r["n"]
    if s and n >= 2:
        z = m / (s / n**0.5)
        p = 2 * float(st.norm.sf(abs(z)))
    else:
        z = p = float("nan")
    r["z"] = z
    r["p"] = p

csv_path = root / "ablation_implicit_fraction.csv"
with open(csv_path, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["fraction","dataset","mean","std","n","z","p"])
    w.writeheader()
    w.writerows(rows)
print(f"Saved: {csv_path}")

# Print table
print()
print("=" * 75)
print("Implicit-fraction ablation (validation, natural, implicit)")
print(f"  {'Fraction':<12} {'Dataset':<10} {'Mean':>8} {'Z':>7} {'P':>10}")
print("  " + "-" * 50)
for r in rows:
    print(f"  {r['fraction']:<12} {r['dataset']:<10} {r['mean']:>8.4f} "
          f"{r['z']:>7.3f} {r['p']:>10.4e}")
print("=" * 75)
PYEOF

echo ""
echo "========================================================"
echo "Ablation complete. Results in: ${ABLATION_ROOT}/"
echo ""
echo "Plot with:"
echo "  python plot_implicit_fraction_ablation.py --csv ${ABLATION_ROOT}/ablation_implicit_fraction.csv"
echo "========================================================"

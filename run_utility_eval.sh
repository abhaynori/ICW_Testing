#!/bin/bash
# Utility benchmark evaluation for the GSM8K-trained implicit watermarking models.
# Runs IFEval and GSM8K accuracy via lm_eval on:
#   1. Base model (Qwen/Qwen2.5-7B-Instruct)
#   2. SFT model  (5-epoch, trained on GSM8K, --no-wm-instruction)
#   3. GRPO model (implicit_fraction=1.0, warm-started from SFT)
#
# Usage:
#   bash run_utility_eval.sh                    # all three models
#   bash run_utility_eval.sh --model base       # base only
#   bash run_utility_eval.sh --model sft        # sft only
#   bash run_utility_eval.sh --model grpo       # grpo only
#
# Requirements: pip install lm_eval
set -euo pipefail

# ── Model paths ──────────────────────────────────────────────────────────────
BASE_MODEL="Qwen/Qwen2.5-7B-Instruct"

# From rerun_acrostics_42_20260522_235846 (output_gsm8k_3.txt)
RUN_ROOT="rerun_acrostics_42_20260522_235846"
SFT_MODEL="${RUN_ROOT}/sft_models/sft_acrostics_full_20260523_101707/final_model"
GRPO_MODEL="${RUN_ROOT}/grpo_models/acrostics_full_20260523_121419/final_model"

# ── Output ───────────────────────────────────────────────────────────────────
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUT_DIR="utility_results/gsm8k_run_${TIMESTAMP}"
mkdir -p "$OUT_DIR"

echo "============================================================"
echo "Utility Eval — ${TIMESTAMP}"
echo "Output: ${OUT_DIR}/"
echo "============================================================"

# ── Which models to run ──────────────────────────────────────────────────────
RUN_BASE=true
RUN_SFT=true
RUN_GRPO=true

for arg in "$@"; do
    case "$arg" in
        --model) continue ;;
        base)  RUN_BASE=true;  RUN_SFT=false;  RUN_GRPO=false ;;
        sft)   RUN_BASE=false; RUN_SFT=true;   RUN_GRPO=false ;;
        grpo)  RUN_BASE=false; RUN_SFT=false;  RUN_GRPO=true  ;;
    esac
done

# ── Helper: run lm_eval for a given model ────────────────────────────────────
# $1 = label (base / sft / grpo)
# $2 = pretrained model id or path
# $3 = peft adapter path (or "" for no peft)
run_lm_eval() {
    local LABEL="$1"
    local PRETRAINED="$2"
    local PEFT="$3"

    local MODEL_ARGS="pretrained=${PRETRAINED},dtype=bfloat16"
    if [ -n "$PEFT" ]; then
        MODEL_ARGS="${MODEL_ARGS},peft=${PEFT}"
    fi

    echo ""
    echo "──────────────────────────────────────────────────────────"
    echo "Model: ${LABEL}"
    echo "  pretrained: ${PRETRAINED}"
    [ -n "$PEFT" ] && echo "  peft:       ${PEFT}"
    echo "──────────────────────────────────────────────────────────"

    # IFEval
    echo ""
    echo "[${LABEL}] Running IFEval..."
    lm_eval \
        --model hf \
        --model_args "${MODEL_ARGS}" \
        --tasks ifeval \
        --batch_size auto \
        --output_path "${OUT_DIR}/${LABEL}/ifeval" \
        --log_samples \
        2>&1 | tee "${OUT_DIR}/${LABEL}_ifeval.log"

    # GSM8K (5-shot, flexible-extract + strict-match)
    echo ""
    echo "[${LABEL}] Running GSM8K (5-shot)..."
    lm_eval \
        --model hf \
        --model_args "${MODEL_ARGS}" \
        --tasks gsm8k \
        --num_fewshot 5 \
        --batch_size auto \
        --output_path "${OUT_DIR}/${LABEL}/gsm8k" \
        --log_samples \
        2>&1 | tee "${OUT_DIR}/${LABEL}_gsm8k.log"

    echo ""
    echo "[${LABEL}] Done."
}

# ── Run ──────────────────────────────────────────────────────────────────────
mkdir -p "${OUT_DIR}/base" "${OUT_DIR}/sft" "${OUT_DIR}/grpo"

if [ "$RUN_BASE" = true ]; then
    run_lm_eval "base" "$BASE_MODEL" ""
fi

if [ "$RUN_SFT" = true ]; then
    # SFT checkpoint has LoRA adapters baked in, but base weights are separate.
    # grpo_train saves the adapter merged or unmerged depending on config.
    # Try with peft= first; if it fails, try pretrained= directly.
    run_lm_eval "sft" "$BASE_MODEL" "$SFT_MODEL"
fi

if [ "$RUN_GRPO" = true ]; then
    run_lm_eval "grpo" "$BASE_MODEL" "$GRPO_MODEL"
fi

# ── Parse and print summary table ────────────────────────────────────────────
echo ""
echo "============================================================"
echo "Parsing results..."
echo "============================================================"

python - <<'PYEOF'
import json, glob, sys
from pathlib import Path
import os

out_dir = sorted(glob.glob("utility_results/gsm8k_run_*"))[-1]
print(f"Results dir: {out_dir}\n")

rows = []
for label in ["base", "sft", "grpo"]:
    row = {"model": label}

    # IFEval
    ifeval_jsons = glob.glob(f"{out_dir}/{label}/ifeval/**/*.json", recursive=True)
    ifeval_jsons = [f for f in ifeval_jsons if "samples_" not in f]
    if ifeval_jsons:
        with open(sorted(ifeval_jsons)[-1]) as f:
            data = json.load(f)
        results = data.get("results", {}).get("ifeval", {})
        row["if_inst_loose"]   = results.get("inst_level_loose_acc,none", float("nan"))
        row["if_inst_strict"]  = results.get("inst_level_strict_acc,none", float("nan"))
        row["if_prom_loose"]   = results.get("prompt_level_loose_acc,none", float("nan"))
        row["if_prom_strict"]  = results.get("prompt_level_strict_acc,none", float("nan"))
    else:
        row.update({k: float("nan") for k in
                    ["if_inst_loose","if_inst_strict","if_prom_loose","if_prom_strict"]})

    # GSM8K
    gsm_jsons = glob.glob(f"{out_dir}/{label}/gsm8k/**/*.json", recursive=True)
    gsm_jsons = [f for f in gsm_jsons if "samples_" not in f]
    if gsm_jsons:
        with open(sorted(gsm_jsons)[-1]) as f:
            data = json.load(f)
        results = data.get("results", {}).get("gsm8k", {})
        row["gsm_flex"]   = results.get("exact_match,flexible-extract", float("nan"))
        row["gsm_strict"] = results.get("exact_match,strict-match", float("nan"))
    else:
        row.update({"gsm_flex": float("nan"), "gsm_strict": float("nan")})

    rows.append(row)

# Print table
header = f"{'Model':<8} | {'IF inst-loose':>14} | {'IF inst-strict':>14} | {'IF pr-loose':>12} | {'IF pr-strict':>12} | {'GSM flex':>9} | {'GSM strict':>10}"
print(header)
print("-" * len(header))
base_row = next((r for r in rows if r["model"] == "base"), {})
for row in rows:
    def fmt(v, base_v):
        if v != v:  # nan
            return "   n/a"
        delta = v - base_v if base_v == base_v else 0
        sign = "+" if delta >= 0 else ""
        return f"{v:.4f}({sign}{delta:+.4f})"

    print(f"{row['model']:<8} | "
          f"{fmt(row['if_inst_loose'],  base_row.get('if_inst_loose', float('nan'))):>14} | "
          f"{fmt(row['if_inst_strict'], base_row.get('if_inst_strict', float('nan'))):>14} | "
          f"{fmt(row['if_prom_loose'],  base_row.get('if_prom_loose', float('nan'))):>12} | "
          f"{fmt(row['if_prom_strict'], base_row.get('if_prom_strict', float('nan'))):>12} | "
          f"{fmt(row['gsm_flex'],       base_row.get('gsm_flex', float('nan'))):>9} | "
          f"{fmt(row['gsm_strict'],     base_row.get('gsm_strict', float('nan'))):>10}")

# Save CSV
import csv
csv_path = f"{out_dir}/utility_summary.csv"
fields = ["model","if_inst_loose","if_inst_strict","if_prom_loose","if_prom_strict","gsm_flex","gsm_strict"]
with open(csv_path, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=fields)
    w.writeheader()
    w.writerows(rows)
print(f"\nSaved: {csv_path}")
PYEOF

echo ""
echo "Done. Results in: ${OUT_DIR}/"

#!/bin/bash
set -euo pipefail
# ============================================================================
# Phase 3: Fine-tuning Robustness
#
# Fine-tunes the GRPO watermarked model on clean (non-watermarked) Alpaca
# data using LoRA, evaluating implicit watermark retention at:
#   steps = 0, 100, 250, 500, 1000
#
# Usage:
#   bash run_finetune_robustness.sh
#   bash run_finetune_robustness.sh --max-steps 500
# ============================================================================

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTDIR="robustness_logs/finetune_${TIMESTAMP}"
mkdir -p "$OUTDIR"

echo "============================================================"
echo "Phase 3: Fine-tuning Robustness  [${TIMESTAMP}]"
echo "Output: ${OUTDIR}/"
echo "============================================================"

python finetune_robustness.py \
    --grpo-model  grpo_models/acrostics_full_20260318_234510/final_model \
    --finetune-dataset alpaca \
    --finetune-samples 2000 \
    --max-steps   1000 \
    --eval-samples 200 \
    --gen-batch   4 \
    --max-new-tokens 512 \
    --min-new-tokens 256 \
    --temperature 0.7 \
    --top-p       0.9 \
    --lora-rank   16 \
    --learning-rate 2e-5 \
    --batch-size  4 \
    --output-dir  "$OUTDIR" \
    "$@" \
    2>&1 | tee "${OUTDIR}/finetune_robustness.log"

echo ""
echo "============================================================"
echo "Done!  Results in: ${OUTDIR}/"
echo ""
echo "To generate plots run:"
echo "  python plot_finetune_robustness.py --csv ${OUTDIR}/finetune_robustness_results.csv"
echo "============================================================"

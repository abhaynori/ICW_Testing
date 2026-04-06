#!/bin/bash
set -euo pipefail

# ============================================================================
# Focused Results Table Pipeline (No-Instruction Approach)
# ============================================================================
# Produces the mentor's table:
#   Row 1: Baseline (implicit eval on base model)
#   Row 2: Baseline + Sys-prompt (explicit eval on base model)
#   Row 3: Baseline + SFT (implicit eval on no-instruction SFT models)
#   Row 4: Baseline + SFT + GRPO (implicit eval on GRPO models)
#
# Key: SFT uses --no-wm-instruction, GRPO uses --implicit-fraction 1.0
# ============================================================================

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOGDIR="experiment_logs/${TIMESTAMP}_results_table"
mkdir -p "$LOGDIR"

# --- Configuration ---------------------------------------------------------
METHOD="acrostics"
MODEL="full"
BASE_MODEL="Qwen/Qwen2.5-7B-Instruct"

# SFT sweep (no-instruction)
SFT_SAMPLES_SWEEP=(500 1000 2000)
SFT_EPOCHS_SWEEP=(3 5 10)

# GRPO sweep (conservative to avoid mode collapse)
GRPO_SAMPLES_SWEEP=(100 200)
GRPO_EPOCHS_SWEEP=(3 5)
GRPO_KL_BETA=0.1           # higher than default to prevent collapse
GRPO_IMPLICIT_FRACTION=1.0

EVAL_SAMPLES=200
EVAL_SPLITS="train,validation"
EVAL_DATASETS="eli5,alpaca"
GEN_BATCH=4
MAX_NEW_TOKENS=512
MIN_NEW_TOKENS=256

SKIP_DATAGEN=false
for arg in "$@"; do
    [ "$arg" = "--skip-datagen" ] && SKIP_DATAGEN=true
done

echo "============================================================"
echo "Results Table Pipeline (No-Instruction) — ${TIMESTAMP}"
echo "Method:       ${METHOD}"
echo "SFT samples:  ${SFT_SAMPLES_SWEEP[*]}"
echo "SFT epochs:   ${SFT_EPOCHS_SWEEP[*]}"
echo "GRPO samples: ${GRPO_SAMPLES_SWEEP[*]}"
echo "GRPO epochs:  ${GRPO_EPOCHS_SWEEP[*]}"
echo "GRPO KL beta: ${GRPO_KL_BETA}"
echo "Eval samples: ${EVAL_SAMPLES}"
echo "Logs:         ${LOGDIR}/"
echo "============================================================"

# Helper: run eval on a model
eval_model() {
    local MODEL_PATH="$1"
    local LOG_FILE="$2"

    python grpo_train.py \
        --eval-only "$MODEL_PATH" \
        --model "$MODEL" \
        --method "$METHOD" \
        --eval-splits "$EVAL_SPLITS" \
        --eval-datasets "$EVAL_DATASETS" \
        --eval-samples "$EVAL_SAMPLES" \
        --gen-batch-size "$GEN_BATCH" \
        --max-new-tokens "$MAX_NEW_TOKENS" \
        --min-new-tokens "$MIN_NEW_TOKENS" \
        2>&1 | tee "$LOG_FILE"
}

# ============================================================================
# PHASE 1: Baseline + Baseline+Sys-prompt (Rows 1 & 2)
# ============================================================================

echo ""
echo "============================================================"
echo "PHASE 1: Evaluating base model (Rows 1 & 2)"
echo "============================================================"

eval_model "$BASE_MODEL" "${LOGDIR}/01_eval_baseline.log"

# ============================================================================
# PHASE 2: Generate SFT data (once, max samples needed)
# ============================================================================

MAX_SFT_SAMPLES=${SFT_SAMPLES_SWEEP[-1]}

if [ "$SKIP_DATAGEN" = false ]; then
    echo ""
    echo "============================================================"
    echo "PHASE 2: Generating SFT data (${MAX_SFT_SAMPLES} samples)"
    echo "============================================================"

    python generate_sft_data.py \
        --method "$METHOD" \
        --model "$MODEL" \
        --samples "$MAX_SFT_SAMPLES" \
        --n-candidates 16 \
        --min-score 1.5 \
        --max-new-tokens "$MAX_NEW_TOKENS" \
        --gen-batch-size "$GEN_BATCH" \
        --dataset eli5 --split train \
        2>&1 | tee "${LOGDIR}/02_datagen.log"
fi

SFT_DATA=$(ls -t sft_data/${METHOD}_${MODEL}_*.json 2>/dev/null | head -1)
if [ -z "$SFT_DATA" ]; then
    echo "ERROR: No SFT data file found. Exiting."
    exit 1
fi
echo ">>> Using SFT data: ${SFT_DATA}"

# ============================================================================
# PHASE 3: SFT sweep with --no-wm-instruction (Row 3)
# ============================================================================

echo ""
echo "============================================================"
echo "PHASE 3: SFT training sweep (no-instruction)"
echo "============================================================"

mkdir -p "${LOGDIR}/sft"

for N_SAMPLES in "${SFT_SAMPLES_SWEEP[@]}"; do
    for N_EPOCHS in "${SFT_EPOCHS_SWEEP[@]}"; do
        TAG="sft_s${N_SAMPLES}_e${N_EPOCHS}"
        echo ""
        echo ">>> SFT: samples=${N_SAMPLES}, epochs=${N_EPOCHS}"

        # --- Train (no watermark instruction) ---
        python sft_train.py \
            --method "$METHOD" \
            --model "$MODEL" \
            --sft-data "$SFT_DATA" \
            --samples "$N_SAMPLES" \
            --epochs "$N_EPOCHS" \
            --batch-size 4 \
            --learning-rate 2e-5 \
            --use-lora \
            --max-length 1024 \
            --no-wm-instruction \
            2>&1 | tee "${LOGDIR}/sft/${TAG}_train.log"

        # Find the model just created
        SFT_MODEL=$(ls -td sft_models/sft_${METHOD}_${MODEL}_*/final_model 2>/dev/null | head -1)
        if [ -z "$SFT_MODEL" ]; then
            echo "WARNING: SFT model not found for ${TAG}, skipping."
            continue
        fi
        echo ">>> Model: ${SFT_MODEL}"

        # --- Eval ---
        eval_model "$SFT_MODEL" "${LOGDIR}/sft/${TAG}_eval.log"
    done
done

# ============================================================================
# PHASE 4: GRPO sweep (Row 4)
# ============================================================================

echo ""
echo "============================================================"
echo "PHASE 4: GRPO training sweep (no-instruction SFT base)"
echo "============================================================"

mkdir -p "${LOGDIR}/grpo"

# Only use SFT models from THIS run (filter by today's timestamp prefix)
RUN_DATE=$(echo "$TIMESTAMP" | cut -c1-8)

for SFT_DIR in sft_models/sft_${METHOD}_${MODEL}_${RUN_DATE}*/final_model; do
    [ -d "$SFT_DIR" ] || continue
    SFT_TAG=$(basename "$(dirname "$SFT_DIR")")

    for G_SAMPLES in "${GRPO_SAMPLES_SWEEP[@]}"; do
        for G_EPOCHS in "${GRPO_EPOCHS_SWEEP[@]}"; do
            TAG="grpo_${SFT_TAG}_s${G_SAMPLES}_e${G_EPOCHS}"
            echo ""
            echo ">>> GRPO: base=${SFT_TAG}, samples=${G_SAMPLES}, epochs=${G_EPOCHS}"

            # --- Train ---
            python grpo_train.py \
                --model "$MODEL" \
                --method "$METHOD" \
                --warm-start-model "$SFT_DIR" \
                --use-lora \
                --samples "$G_SAMPLES" \
                --epochs "$G_EPOCHS" \
                --batch-size 4 \
                --learning-rate 1e-5 \
                --num-generations 4 \
                --max-new-tokens 200 \
                --implicit-fraction "$GRPO_IMPLICIT_FRACTION" \
                --beta "$GRPO_KL_BETA" \
                --allow-implicit-reference \
                --eval-splits "$EVAL_SPLITS" \
                --eval-datasets "$EVAL_DATASETS" \
                --eval-samples "$EVAL_SAMPLES" \
                --gen-batch-size "$GEN_BATCH" \
                2>&1 | tee "${LOGDIR}/grpo/${TAG}_train.log"

            # Find the GRPO model just created
            GRPO_MODEL=$(ls -td grpo_models/${METHOD}_${MODEL}_*/final_model 2>/dev/null | head -1)
            if [ -z "$GRPO_MODEL" ]; then
                echo "WARNING: GRPO model not found for ${TAG}, skipping eval."
                continue
            fi

            # --- Eval ---
            eval_model "$GRPO_MODEL" "${LOGDIR}/grpo/${TAG}_eval.log"
        done
    done
done

echo ""
echo "============================================================"
echo "Pipeline complete! Logs: ${LOGDIR}/"
echo "============================================================"

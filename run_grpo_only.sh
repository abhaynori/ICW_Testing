#!/bin/bash
set -euo pipefail

# ============================================================================
# Phase 4 ONLY: GRPO training sweep on completed SFT models
# Skips Phases 1-3 (already done) and jumps straight to GRPO.
# ============================================================================

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOGDIR="experiment_logs/${TIMESTAMP}_grpo_only"
mkdir -p "$LOGDIR/grpo"

# --- Configuration (must match run_experiments.sh) -------------------------
METHOD="acrostics"
MODEL="full"
BASE_MODEL="Qwen/Qwen2.5-7B-Instruct"

GRPO_SAMPLES_SWEEP=(50 100 200)
GRPO_EPOCHS_SWEEP=(1 3)

EVAL_SAMPLES=50
EVAL_SPLITS="train,validation"
EVAL_DATASETS="eli5,alpaca"
GEN_BATCH=4
MAX_NEW_TOKENS=512
MIN_NEW_TOKENS=256

# --- Only use today's SFT models (20260318_*) ------------------------------
SFT_MODELS=(
    sft_models/sft_acrostics_full_20260318_034701/final_model   # 100 samples, 1 epoch
    sft_models/sft_acrostics_full_20260318_041933/final_model   # 100 samples, 3 epochs
    sft_models/sft_acrostics_full_20260318_045229/final_model   # 100 samples, 5 epochs
    sft_models/sft_acrostics_full_20260318_052719/final_model   # 500 samples, 1 epoch
    sft_models/sft_acrostics_full_20260318_060141/final_model   # 500 samples, 3 epochs
    sft_models/sft_acrostics_full_20260318_063827/final_model   # 500 samples, 5 epochs
    sft_models/sft_acrostics_full_20260318_071807/final_model   # 1000 samples, 1 epoch
    sft_models/sft_acrostics_full_20260318_075250/final_model   # 1000 samples, 3 epochs
    sft_models/sft_acrostics_full_20260318_083339/final_model   # 1000 samples, 5 epochs
    sft_models/sft_acrostics_full_20260318_092018/final_model   # 2000 samples, 1 epoch
    sft_models/sft_acrostics_full_20260318_095824/final_model   # 2000 samples, 3 epochs
    sft_models/sft_acrostics_full_20260318_104933/final_model   # 2000 samples, 5 epochs
)

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

echo "============================================================"
echo "GRPO-Only Sweep — ${TIMESTAMP}"
echo "SFT models: ${#SFT_MODELS[@]}"
echo "GRPO samples: ${GRPO_SAMPLES_SWEEP[*]}"
echo "GRPO epochs:  ${GRPO_EPOCHS_SWEEP[*]}"
echo "Total GRPO runs: $(( ${#SFT_MODELS[@]} * ${#GRPO_SAMPLES_SWEEP[@]} * ${#GRPO_EPOCHS_SWEEP[@]} ))"
echo "Logs: ${LOGDIR}/"
echo "============================================================"

for SFT_DIR in "${SFT_MODELS[@]}"; do
    [ -d "$SFT_DIR" ] || { echo "WARNING: $SFT_DIR not found, skipping."; continue; }
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
                --implicit-fraction 0.4 \
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
echo "GRPO sweep complete!"
echo "Logs: ${LOGDIR}/grpo/"
echo "============================================================"

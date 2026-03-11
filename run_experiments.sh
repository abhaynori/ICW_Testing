#!/bin/bash
set -euo pipefail

# ============================================================================
# ICW Rejection-Sampling SFT Pipeline
# Run on Hyak cluster: bash run_experiments.sh
# ============================================================================

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOGDIR="experiment_logs/${TIMESTAMP}"
mkdir -p "$LOGDIR"

echo "============================================================"
echo "ICW Experiment Pipeline — ${TIMESTAMP}"
echo "Logs: ${LOGDIR}/"
echo "============================================================"

# ---------------------------------------------------------------------------
# PHASE 1: Generate rejection-sampled SFT data (with implicit fraction)
# ---------------------------------------------------------------------------

echo ""
echo ">>> PHASE 1: Generating acrostics RS-SFT data (implicit-fraction=0.5)..."
python generate_sft_data.py \
    --method acrostics \
    --model full \
    --samples 2000 \
    --n-candidates 16 \
    --min-score 1.5 \
    --max-new-tokens 512 \
    --gen-batch-size 4 \
    --implicit-fraction 0.5 \
    2>&1 | tee "${LOGDIR}/01_datagen_acrostics.log"

ACROSTICS_DATA=$(ls -t sft_data/acrostics_full_*.json 2>/dev/null | head -1)
if [ -z "$ACROSTICS_DATA" ]; then
    echo "ERROR: No acrostics data file found in sft_data/. Exiting."
    exit 1
fi
echo ">>> Acrostics data: ${ACROSTICS_DATA}"

# ---------------------------------------------------------------------------
# PHASE 2: SFT training (with data mixing + per-record implicit flag)
# ---------------------------------------------------------------------------

echo ""
echo ">>> PHASE 2: SFT training on acrostics data (with mixing)..."
python sft_train.py \
    --method acrostics \
    --model full \
    --sft-data "$ACROSTICS_DATA" \
    --mix-ratio 0.5 \
    --use-lora \
    --learning-rate 5e-6 \
    --epochs 2 \
    --batch-size 4 \
    --max-length 1024 \
    2>&1 | tee "${LOGDIR}/02_sft_acrostics.log"

ACROSTICS_MODEL=$(ls -td sft_models/sft_acrostics_full_*/final_model 2>/dev/null | head -1)
if [ -z "$ACROSTICS_MODEL" ]; then
    echo "ERROR: No acrostics model found. Exiting."
    exit 1
fi
echo ">>> Acrostics model: ${ACROSTICS_MODEL}"

# ---------------------------------------------------------------------------
# PHASE 3: Evaluation (with min-new-tokens to prevent short responses)
# ---------------------------------------------------------------------------

echo ""
echo ">>> PHASE 3a: Evaluating RS-SFT model..."
python grpo_train.py \
    --eval-only "$ACROSTICS_MODEL" \
    --model full \
    --method acrostics \
    --gen-batch-size 4 \
    --max-new-tokens 512 \
    --min-new-tokens 256 \
    2>&1 | tee "${LOGDIR}/03_eval_acrostics.log"

echo ""
echo ">>> PHASE 3b: Evaluating base Qwen model..."
python grpo_train.py \
    --eval-only Qwen/Qwen2.5-7B-Instruct \
    --model full \
    --method acrostics \
    --gen-batch-size 4 \
    --max-new-tokens 512 \
    --min-new-tokens 256 \
    2>&1 | tee "${LOGDIR}/04_eval_base_qwen.log"

# Old SFT baseline
OLD_SFT_ACROSTICS="sft_models/sft_acrostics_full_20260303_022655/final_model"
if [ -d "$OLD_SFT_ACROSTICS" ]; then
    echo ""
    echo ">>> PHASE 3c: Evaluating old SFT model (non-watermarked targets)..."
    python grpo_train.py \
        --eval-only "$OLD_SFT_ACROSTICS" \
        --model full \
        --method acrostics \
        --gen-batch-size 4 \
        --max-new-tokens 512 \
        --min-new-tokens 256 \
        2>&1 | tee "${LOGDIR}/05_eval_old_sft.log"
fi

# ---------------------------------------------------------------------------
# PHASE 4: Utility benchmarks (lm_eval)
# ---------------------------------------------------------------------------

UTILITY_TASKS="mmlu gsm8k ifeval"
BASE_MODEL="Qwen/Qwen2.5-7B-Instruct"

echo ""
echo "============================================================"
echo "PHASE 4: Utility benchmarks (lm_eval)"
echo "Tasks: ${UTILITY_TASKS}"
echo "============================================================"

if command -v lm_eval &>/dev/null; then
    LM_EVAL="lm_eval"
elif python -c "import lm_eval" &>/dev/null 2>&1; then
    LM_EVAL="python -m lm_eval"
else
    echo "WARNING: lm_eval not found. Skipping utility benchmarks."
    echo "Install with: pip install lm-eval"
    LM_EVAL=""
fi

if [ -n "$LM_EVAL" ]; then
    mkdir -p "${LOGDIR}/utility"

    for TASK in $UTILITY_TASKS; do
        echo ""
        echo ">>> Utility: ${TASK} — base model (${BASE_MODEL})..."
        $LM_EVAL \
            --model hf \
            --model_args "pretrained=${BASE_MODEL},trust_remote_code=True" \
            --tasks "$TASK" \
            --batch_size auto \
            --output_path "${LOGDIR}/utility/base_${TASK}.json" \
            2>&1 | tee "${LOGDIR}/utility/base_${TASK}.log"

        echo ""
        echo ">>> Utility: ${TASK} — acrostics RS-SFT model..."
        $LM_EVAL \
            --model hf \
            --model_args "pretrained=${ACROSTICS_MODEL},trust_remote_code=True" \
            --tasks "$TASK" \
            --batch_size auto \
            --output_path "${LOGDIR}/utility/acrostics_${TASK}.json" \
            2>&1 | tee "${LOGDIR}/utility/acrostics_${TASK}.log"
    done
fi

# ---------------------------------------------------------------------------
# DONE
# ---------------------------------------------------------------------------

echo ""
echo "============================================================"
echo "ALL EXPERIMENTS COMPLETE"
echo "============================================================"
echo "Logs:            ${LOGDIR}/"
echo "Acrostics data:  ${ACROSTICS_DATA}"
echo "Acrostics model: ${ACROSTICS_MODEL}"
echo "Utility results: ${LOGDIR}/utility/"
echo "============================================================"

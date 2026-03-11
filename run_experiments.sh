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
# PHASE 1: Generate rejection-sampled SFT data
# ---------------------------------------------------------------------------

echo ""
echo ">>> PHASE 1a: Generating initials RS-SFT data..."
python generate_sft_data.py \
    --method initials \
    --model full \
    --samples 2000 \
    --n-candidates 8 \
    --min-score 1.5 \
    --max-new-tokens 512 \
    --gen-batch-size 4 \
    --implicit-fraction 0.0 \
    2>&1 | tee "${LOGDIR}/01_datagen_initials.log"

INITIALS_DATA=$(ls -t sft_data/initials_full_*.json 2>/dev/null | head -1)
if [ -z "$INITIALS_DATA" ]; then
    echo "ERROR: No initials data file found in sft_data/. Exiting."
    exit 1
fi
echo ">>> Initials data: ${INITIALS_DATA}"

echo ""
echo ">>> PHASE 1b: Generating acrostics RS-SFT data (round 2, warm-start)..."
python generate_sft_data.py \
    --method acrostics \
    --model full \
    --warm-start-model sft_models/sft_acrostics_full_20260306_070506/final_model \
    --samples 2000 \
    --n-candidates 16 \
    --min-score 2.0 \
    --max-new-tokens 512 \
    --gen-batch-size 4 \
    --implicit-fraction 0.0 \
    2>&1 | tee "${LOGDIR}/02_datagen_acrostics_r2.log"

ACROSTICS_DATA=$(ls -t sft_data/acrostics_full_*.json 2>/dev/null | head -1)
if [ -z "$ACROSTICS_DATA" ]; then
    echo "ERROR: No acrostics data file found in sft_data/. Exiting."
    exit 1
fi
echo ">>> Acrostics data: ${ACROSTICS_DATA}"

# ---------------------------------------------------------------------------
# PHASE 2: SFT training (with 50% data mixing to prevent catastrophic forgetting)
# ---------------------------------------------------------------------------

echo ""
echo ">>> PHASE 2a: SFT training on initials data (with mixing)..."
python sft_train.py \
    --method initials \
    --model full \
    --sft-data "$INITIALS_DATA" \
    --mix-ratio 0.5 \
    --use-lora \
    --learning-rate 5e-6 \
    --epochs 2 \
    --batch-size 4 \
    --max-length 1024 \
    2>&1 | tee "${LOGDIR}/03_sft_initials.log"

INITIALS_MODEL=$(ls -td sft_models/sft_initials_full_*/final_model 2>/dev/null | head -1)
if [ -z "$INITIALS_MODEL" ]; then
    echo "ERROR: No initials model found. Exiting."
    exit 1
fi
echo ">>> Initials model: ${INITIALS_MODEL}"

echo ""
echo ">>> PHASE 2b: SFT training on acrostics data (with mixing)..."
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
    2>&1 | tee "${LOGDIR}/04_sft_acrostics_r2.log"

ACROSTICS_MODEL=$(ls -td sft_models/sft_acrostics_full_*/final_model 2>/dev/null | head -1)
if [ -z "$ACROSTICS_MODEL" ]; then
    echo "ERROR: No acrostics model found. Exiting."
    exit 1
fi
echo ">>> Acrostics model: ${ACROSTICS_MODEL}"

# ---------------------------------------------------------------------------
# PHASE 3: Evaluation (watermark detection)
# ---------------------------------------------------------------------------

echo ""
echo ">>> PHASE 3a: Evaluating initials RS-SFT model..."
python grpo_train.py \
    --eval-only "$INITIALS_MODEL" \
    --model full \
    --method initials \
    --gen-batch-size 4 \
    --max-new-tokens 512 \
    2>&1 | tee "${LOGDIR}/05_eval_initials.log"

echo ""
echo ">>> PHASE 3b: Evaluating acrostics RS-SFT model (round 2)..."
python grpo_train.py \
    --eval-only "$ACROSTICS_MODEL" \
    --model full \
    --method acrostics \
    --gen-batch-size 4 \
    --max-new-tokens 512 \
    2>&1 | tee "${LOGDIR}/06_eval_acrostics_r2.log"

# ---------------------------------------------------------------------------
# PHASE 4: Baseline comparisons
# ---------------------------------------------------------------------------

# 4a/4b: Base Qwen (no fine-tuning — pure instruction-following baseline)
echo ""
echo ">>> PHASE 4a: Evaluating base Qwen model — initials..."
python grpo_train.py \
    --eval-only Qwen/Qwen2.5-7B-Instruct \
    --model full \
    --method initials \
    --gen-batch-size 4 \
    --max-new-tokens 512 \
    2>&1 | tee "${LOGDIR}/07_eval_base_qwen_initials.log"

echo ""
echo ">>> PHASE 4b: Evaluating base Qwen model — acrostics..."
python grpo_train.py \
    --eval-only Qwen/Qwen2.5-7B-Instruct \
    --model full \
    --method acrostics \
    --gen-batch-size 4 \
    --max-new-tokens 512 \
    2>&1 | tee "${LOGDIR}/08_eval_base_qwen_acrostics.log"

# 4c: Old SFT model (trained on non-watermarked targets — shows SFT alone)
OLD_SFT_ACROSTICS="sft_models/sft_acrostics_full_20260303_022655/final_model"
if [ -d "$OLD_SFT_ACROSTICS" ]; then
    echo ""
    echo ">>> PHASE 4c: Evaluating old SFT model (non-watermarked targets) — acrostics..."
    python grpo_train.py \
        --eval-only "$OLD_SFT_ACROSTICS" \
        --model full \
        --method acrostics \
        --gen-batch-size 4 \
        --max-new-tokens 512 \
        2>&1 | tee "${LOGDIR}/09_eval_old_sft_acrostics.log"
else
    echo ">>> Skipping old SFT eval: ${OLD_SFT_ACROSTICS} not found"
fi

# ---------------------------------------------------------------------------
# PHASE 5: Utility benchmarks (lm_eval — MMLU, GSM8K, IFEval)
# Measures whether watermark training degrades general capability
# ---------------------------------------------------------------------------

UTILITY_TASKS="mmlu gsm8k ifeval"
BASE_MODEL="Qwen/Qwen2.5-7B-Instruct"

echo ""
echo "============================================================"
echo "PHASE 5: Utility benchmarks (lm_eval)"
echo "Tasks: ${UTILITY_TASKS}"
echo "============================================================"

# Check if lm_eval is available
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
        # --- Base model ---
        echo ""
        echo ">>> Utility: ${TASK} — base model (${BASE_MODEL})..."
        $LM_EVAL \
            --model hf \
            --model_args "pretrained=${BASE_MODEL},trust_remote_code=True" \
            --tasks "$TASK" \
            --batch_size auto \
            --output_path "${LOGDIR}/utility/base_${TASK}.json" \
            2>&1 | tee "${LOGDIR}/utility/base_${TASK}.log"

        # --- Initials RS-SFT model ---
        echo ""
        echo ">>> Utility: ${TASK} — initials RS-SFT model..."
        $LM_EVAL \
            --model hf \
            --model_args "pretrained=${INITIALS_MODEL},trust_remote_code=True" \
            --tasks "$TASK" \
            --batch_size auto \
            --output_path "${LOGDIR}/utility/initials_${TASK}.json" \
            2>&1 | tee "${LOGDIR}/utility/initials_${TASK}.log"

        # --- Acrostics RS-SFT model ---
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
echo "Initials data:   ${INITIALS_DATA}"
echo "Acrostics data:  ${ACROSTICS_DATA}"
echo "Initials model:  ${INITIALS_MODEL}"
echo "Acrostics model: ${ACROSTICS_MODEL}"
echo ""
echo "Utility results: ${LOGDIR}/utility/"
echo "============================================================"

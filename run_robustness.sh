#!/bin/bash
set -euo pipefail

# ============================================================================
# Watermark Robustness Evaluation Script
# ============================================================================
# Tests watermark retention under three attack axes:
#   1. Temperature / top-p variation  (temp 0=greedy, 0.6, 1.0, 1.5)
#   2. Adversarial system prompts     (stylistic + semantic)
#   3. (Phase 3 — separate script)   Fine-tuning on other data
#
# Run this on the GPU cluster after the main results table run.
# Usage:
#   bash run_robustness.sh               # full sweep
#   bash run_robustness.sh --phase 1     # temperature only
#   bash run_robustness.sh --phase 2     # system prompts only
# ============================================================================

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOGDIR="robustness_logs/${TIMESTAMP}"
mkdir -p "$LOGDIR"

# --- Best model checkpoints (from prior runs) ------------------------------
GRPO_MODEL="grpo_models/acrostics_full_20260318_234510/final_model"
SFT_MODEL="sft_models/sft_acrostics_full_20260318_232049/final_model"
BASE_MODEL="Qwen/Qwen2.5-7B-Instruct"

# --- Shared eval settings ---------------------------------------------------
METHOD="acrostics"
MODEL_STRATEGY="full"
EVAL_SAMPLES=200
EVAL_SPLITS="train,validation"
EVAL_DATASETS="eli5,alpaca"
GEN_BATCH=4
MAX_NEW_TOKENS=512
MIN_NEW_TOKENS=256

# --- Which phases to run ----------------------------------------------------
RUN_PHASE1=true
RUN_PHASE2=true
for arg in "$@"; do
    if [ "$arg" = "--phase" ]; then continue; fi
    case "$arg" in
        1) RUN_PHASE1=true;  RUN_PHASE2=false ;;
        2) RUN_PHASE1=false; RUN_PHASE2=true  ;;
    esac
done

echo "============================================================"
echo "Robustness Evaluation — ${TIMESTAMP}"
echo "GRPO model:   ${GRPO_MODEL}"
echo "SFT model:    ${SFT_MODEL}"
echo "Eval samples: ${EVAL_SAMPLES}"
echo "Logs:         ${LOGDIR}/"
echo "============================================================"

# Helper ─ run eval with configurable temperature and system prompt
eval_robustness() {
    local MODEL_PATH="$1"
    local LOG_TAG="$2"
    local TEMP="$3"
    local TOP_P="$4"
    local LOG_FILE="${LOGDIR}/${LOG_TAG}.log"
    shift 4          # remaining args forwarded verbatim (e.g. --base-system-prompt)
    local EXTRA=("$@")

    echo ""
    echo ">>> ${LOG_TAG}  (temp=${TEMP}, top_p=${TOP_P})"
    python grpo_train.py \
        --eval-only "$MODEL_PATH" \
        --model "$MODEL_STRATEGY" \
        --method "$METHOD" \
        --eval-splits "$EVAL_SPLITS" \
        --eval-datasets "$EVAL_DATASETS" \
        --eval-samples "$EVAL_SAMPLES" \
        --gen-batch-size "$GEN_BATCH" \
        --max-new-tokens "$MAX_NEW_TOKENS" \
        --min-new-tokens "$MIN_NEW_TOKENS" \
        --temperature "$TEMP" \
        --top-p "$TOP_P" \
        --eval-output-dir "${LOGDIR}/${LOG_TAG}" \
        "${EXTRA[@]}" \
        2>&1 | tee "$LOG_FILE"
}

# ============================================================================
# PHASE 1: Temperature / top-p robustness
# ============================================================================
if [ "$RUN_PHASE1" = true ]; then
    echo ""
    echo "============================================================"
    echo "PHASE 1: Temperature robustness"
    echo "============================================================"

    mkdir -p "${LOGDIR}/temp"

    # Temperatures to sweep (0 = greedy decoding via do_sample=False)
    declare -A TEMP_TOP_P=(
        ["greedy_t0"]="0:1.0"
        ["low_t06"]="0.6:0.9"
        ["standard_t07"]="0.7:0.9"
        ["high_t10"]="1.0:0.95"
        ["very_high_t15"]="1.5:0.95"
    )

    # Ordered list so logs appear sequentially
    TEMP_KEYS=("greedy_t0" "low_t06" "standard_t07" "high_t10" "very_high_t15")

    for KEY in "${TEMP_KEYS[@]}"; do
        PAIR="${TEMP_TOP_P[$KEY]}"
        TEMP="${PAIR%%:*}"
        TOP_P="${PAIR##*:}"

        # Test on best GRPO model
        eval_robustness "$GRPO_MODEL" "temp/grpo_${KEY}" "$TEMP" "$TOP_P"

        # Also test base model at same temp (to confirm watermark is model property, not chance)
        eval_robustness "$BASE_MODEL"  "temp/base_${KEY}"  "$TEMP" "$TOP_P"
    done
fi

# ============================================================================
# PHASE 2: System-prompt robustness
# ============================================================================
if [ "$RUN_PHASE2" = true ]; then
    echo ""
    echo "============================================================"
    echo "PHASE 2: System-prompt robustness (temp=0.7, top_p=0.9)"
    echo "============================================================"

    mkdir -p "${LOGDIR}/sysprompt"

    # --- Stylistic prompts ---
    eval_robustness "$GRPO_MODEL" "sysprompt/grpo_concise" "0.7" "0.9" \
        --base-system-prompt "Always answer concisely and briefly."

    eval_robustness "$GRPO_MODEL" "sysprompt/grpo_pirate" "0.7" "0.9" \
        --base-system-prompt "You are a helpful assistant. Always talk like a pirate."

    eval_robustness "$GRPO_MODEL" "sysprompt/grpo_formal" "0.7" "0.9" \
        --base-system-prompt "You are a formal academic assistant. Use precise, technical language."

    # --- Semantic / persona prompts ---
    eval_robustness "$GRPO_MODEL" "sysprompt/grpo_company_bot" "0.7" "0.9" \
        --base-system-prompt "You are a customer service chatbot for Acme Corp. Your goal is to help customers with their queries politely and efficiently."

    eval_robustness "$GRPO_MODEL" "sysprompt/grpo_teacher" "0.7" "0.9" \
        --base-system-prompt "You are a patient teacher helping students understand complex topics. Explain things step by step."

    eval_robustness "$GRPO_MODEL" "sysprompt/grpo_no_sysprompt" "0.7" "0.9" \
        --base-system-prompt ""

    # Compare: base model under same system prompts (shows watermark is not in prompt)
    eval_robustness "$BASE_MODEL" "sysprompt/base_concise" "0.7" "0.9" \
        --base-system-prompt "Always answer concisely and briefly."

    eval_robustness "$BASE_MODEL" "sysprompt/base_pirate" "0.7" "0.9" \
        --base-system-prompt "You are a helpful assistant. Always talk like a pirate."

    eval_robustness "$BASE_MODEL" "sysprompt/base_company_bot" "0.7" "0.9" \
        --base-system-prompt "You are a customer service chatbot for Acme Corp. Your goal is to help customers with their queries politely and efficiently."
fi

# ============================================================================
# Summary
# ============================================================================
echo ""
echo "============================================================"
echo "Robustness evaluation complete!"
echo "Results in: ${LOGDIR}/"
echo ""
echo "To summarize z-scores across conditions run:"
echo "  python summarize_robustness.py --logdir ${LOGDIR}"
echo "============================================================"

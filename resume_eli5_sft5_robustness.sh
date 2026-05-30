#!/bin/bash
set -euo pipefail

# ============================================================================
# Resume run_eli5_sft5_robustness.sh from where SSH dropped.
#
# Completed:
#   Phase 1 (temperature): all 10 evals done
#   Phase 2 (sysprompt):   grpo_no_sysprompt, grpo_concise, grpo_pirate,
#                           grpo_formal, grpo_company_bot, grpo_teacher done
#                           grpo_math_tutor INCOMPLETE (stopped mid-eval)
#
# Resuming from:
#   - grpo_math_tutor (re-run entirely — partial results not reliable)
#   - base_no_sysprompt, base_concise, base_pirate
#   - Phase 3: fine-tuning robustness
# ============================================================================

BASE_MODEL="Qwen/Qwen2.5-7B-Instruct"
RUN_ROOT="rerun_acrostics_eli5_sft5_42_20260526_161831"
GRPO_MODEL="${RUN_ROOT}/grpo_models/acrostics_full_20260527_113049/final_model"

METHOD="acrostics"
SECRET="SECRET"
TRAIN_DATASET="eli5"
EVAL_DATASETS="gsm8k,eli5,alpaca"
EVAL_SPLITS="validation"
EVAL_SAMPLES=200
GEN_BATCH=4
MAX_NEW_TOKENS=200

# Reuse the same log directory so results are co-located with completed phases
LOGDIR="robustness_logs/eli5_sft5_20260528_172516"
mkdir -p "$LOGDIR"

echo "============================================================"
echo "Resuming ELI5 SFT5 robustness from Phase 2 (sysprompt)"
echo "GRPO model: ${GRPO_MODEL}"
echo "Logs:       ${LOGDIR}/"
echo "============================================================"

eval_robustness() {
    local MODEL_PATH="$1"
    local LOG_TAG="$2"
    local TEMP="$3"
    local TOP_P="$4"
    shift 4
    local EXTRA=("$@")

    local LOG_FILE="${LOGDIR}/${LOG_TAG}.log"
    mkdir -p "$(dirname "${LOGDIR}/${LOG_TAG}")"

    echo ""
    echo ">>> ${LOG_TAG}  (temp=${TEMP}, top_p=${TOP_P})"
    python grpo_train.py \
        --eval-only        "$MODEL_PATH" \
        --model            full \
        --method           "$METHOD" \
        --secret-sequence  "$SECRET" \
        --train-dataset    "$TRAIN_DATASET" \
        --eval-datasets    "$EVAL_DATASETS" \
        --eval-splits      "$EVAL_SPLITS" \
        --eval-samples     "$EVAL_SAMPLES" \
        --eval-profiles    natural \
        --eval-modes       implicit,explicit \
        --gen-batch-size   "$GEN_BATCH" \
        --max-new-tokens   "$MAX_NEW_TOKENS" \
        --temperature      "$TEMP" \
        --top-p            "$TOP_P" \
        --eval-output-dir  "${LOGDIR}/${LOG_TAG}" \
        "${EXTRA[@]}" \
        2>&1 | tee "$LOG_FILE"
}

# ── Finish Phase 2: remaining sysprompt evals ─────────────────────────────────
echo ""
echo "============================================================"
echo "PHASE 2 (resume): Remaining sysprompt evals"
echo "============================================================"

# grpo_math_tutor was incomplete — re-run it
eval_robustness "$GRPO_MODEL" "sysprompt/grpo_math_tutor" "0.7" "0.9" \
    --base-system-prompt "You are a math tutor. Show all steps clearly when solving problems."

# Base model controls (not yet run)
eval_robustness "$BASE_MODEL" "sysprompt/base_no_sysprompt" "0.7" "0.9" \
    --base-system-prompt ""

eval_robustness "$BASE_MODEL" "sysprompt/base_concise" "0.7" "0.9" \
    --base-system-prompt "Always answer concisely and briefly."

eval_robustness "$BASE_MODEL" "sysprompt/base_pirate" "0.7" "0.9" \
    --base-system-prompt "You are a helpful assistant. Always talk like a pirate."

# ── Phase 3: Fine-tuning robustness ──────────────────────────────────────────
echo ""
echo "============================================================"
echo "PHASE 3: Fine-tuning robustness"
echo "============================================================"

FINETUNE_OUTDIR="${LOGDIR}/finetune"
mkdir -p "$FINETUNE_OUTDIR"

python finetune_robustness.py \
    --grpo-model        "$GRPO_MODEL" \
    --base-model        "$BASE_MODEL" \
    --finetune-dataset  alpaca \
    --finetune-samples  2000 \
    --max-steps         1000 \
    --eval-samples      200 \
    --gen-batch         4 \
    --max-new-tokens    512 \
    --min-new-tokens    256 \
    --temperature       0.7 \
    --top-p             0.9 \
    --lora-rank         16 \
    --learning-rate     2e-5 \
    --batch-size        4 \
    --output-dir        "$FINETUNE_OUTDIR" \
    --seed              42 \
    2>&1 | tee "${FINETUNE_OUTDIR}/finetune_robustness.log"

# ── Summarise all results ─────────────────────────────────────────────────────
echo ""
echo "============================================================"
echo "All phases complete! Summarising..."
echo "============================================================"
python summarize_robustness.py --logdir "$LOGDIR" 2>/dev/null || true
python parse_robustness_results.py --logdir "$LOGDIR" || true

echo ""
echo "Results in: ${LOGDIR}/"
echo ""
echo "Next steps — generate plots:"
echo "  python plot_robustness_temperature.py --logdir ${LOGDIR}"
echo "  python plot_sysprompt_robustness.py --logdir ${LOGDIR}"
echo "  python plot_finetune_robustness.py --csv ${LOGDIR}/finetune/finetune_robustness_results.csv"
echo "============================================================"

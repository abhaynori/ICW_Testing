#!/bin/bash
# ============================================================================
# Resume god_eli5_sft5.sh from step 4 (SFT eval)
# SFT training already completed; continuing from eval onward.
#
# SFT model: rerun_acrostics_eli5_sft5_42_20260526_161831/sft_models/sft_acrostics_full_20260527_003232/final_model
# ============================================================================
set -euo pipefail

export METHOD=acrostics
export MODEL=full
export SECRET=SECRET
export TRAIN_DATASET=eli5
export SEED=42
export GRPO_IMPLICIT_FRACTION=1.0
export RUN_ROOT="rerun_acrostics_eli5_sft5_42_20260526_161831"
export SFT_MODEL="${RUN_ROOT}/sft_models/sft_acrostics_full_20260527_003232/final_model"

echo "========================================================"
echo "Resuming ELI5 SFT-5 run: ${RUN_ROOT}"
echo "Continuing from Step 4 (SFT eval)"
echo "SFT model: ${SFT_MODEL}"
echo "========================================================"

# ── Step 4: Eval SFT ──────────────────────────────────────────────────────────
python grpo_train.py \
    --eval-only "$SFT_MODEL" \
    --model "$MODEL" \
    --method "$METHOD" \
    --secret-sequence "$SECRET" \
    --train-dataset "$TRAIN_DATASET" \
    --samples 200 \
    --eval-datasets gsm8k,eli5,alpaca \
    --eval-splits validation,test \
    --eval-samples 200 \
    --eval-profiles natural \
    --eval-modes implicit,explicit \
    --gen-batch-size 4 \
    --max-new-tokens 200 \
    --seed "$SEED" \
    --eval-output-dir "$RUN_ROOT/eval/sft"

# ── Step 5: GRPO training ─────────────────────────────────────────────────────
python grpo_train.py \
    --model "$MODEL" \
    --method "$METHOD" \
    --secret-sequence "$SECRET" \
    --warm-start-model "$SFT_MODEL" \
    --use-lora \
    --samples 200 \
    --epochs 3 \
    --batch-size 4 \
    --learning-rate 1e-5 \
    --num-generations 4 \
    --implicit-fraction "$GRPO_IMPLICIT_FRACTION" \
    --beta 0.04 \
    --allow-implicit-reference \
    --max-new-tokens 200 \
    --gen-batch-size 4 \
    --eval-splits "" \
    --train-dataset "$TRAIN_DATASET" \
    --seed "$SEED" \
    --output-dir "$RUN_ROOT/grpo_models"

export GRPO_MODEL=$(find "$RUN_ROOT/grpo_models" -type d -path "*/acrostics_full_*/final_model" | sort | tail -n 1)
if [ -z "$GRPO_MODEL" ]; then
    echo "ERROR: no GRPO model found" >&2; exit 1
fi
echo "GRPO model: $GRPO_MODEL"

# ── Step 6: Eval GRPO ─────────────────────────────────────────────────────────
python grpo_train.py \
    --eval-only "$GRPO_MODEL" \
    --model "$MODEL" \
    --method "$METHOD" \
    --secret-sequence "$SECRET" \
    --train-dataset "$TRAIN_DATASET" \
    --samples 200 \
    --eval-datasets gsm8k,eli5,alpaca \
    --eval-splits validation,test \
    --eval-samples 200 \
    --eval-profiles natural \
    --eval-modes implicit,explicit \
    --gen-batch-size 4 \
    --max-new-tokens 200 \
    --seed "$SEED" \
    --eval-output-dir "$RUN_ROOT/eval/grpo"

# ── Step 7: Text-attack robustness on GRPO ────────────────────────────────────
mapfile -t EVAL_JSONS < <(find "$RUN_ROOT/eval/grpo" -maxdepth 1 -type f -name "eval_*_test_*_natural.json" | sort)
for EVAL_JSON in "${EVAL_JSONS[@]}"; do
    python robustness_eval.py \
        --eval-json "$EVAL_JSON" \
        --method "$METHOD" \
        --secret-sequence "$SECRET" \
        --attacks format_cleanup,truncate_sentence_50,truncate_word_50,sentence_merge,sentence_split,word_dropout,compression
done

echo ""
echo "========================================================"
echo "ELI5 SFT-5 run complete: ${RUN_ROOT}"
echo ""
echo "Results:"
echo "  base:  $RUN_ROOT/eval/base/"
echo "  sft:   $RUN_ROOT/eval/sft/"
echo "  grpo:  $RUN_ROOT/eval/grpo/"
echo "========================================================"

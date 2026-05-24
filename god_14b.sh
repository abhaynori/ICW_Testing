#!/bin/bash
# ============================================================================
# 14B model pipeline — identical to god.sh but with Qwen2.5-14B-Instruct
#
# Purpose: show the watermark scales to a larger model and (likely) achieves
# stronger OOD generalisation due to better implicit-instruction following.
#
# Memory: 14B in bfloat16 ≈ 28 GB. Requires a single A100-80 GB or 2× A40.
# The script sets --model full which uses device_map=auto.
#
# Usage:
#   bash god_14b.sh
# ============================================================================
set -euo pipefail

export METHOD=acrostics
export MODEL=full
export BASE_MODEL="Qwen/Qwen2.5-14B-Instruct"
export TEACHER_MODEL="Qwen/Qwen2.5-14B-Instruct"
export SECRET=SECRET
export TRAIN_DATASET=gsm8k
export SEED=42
export GRPO_IMPLICIT_FRACTION=1.0
export RUN_ROOT="rerun_acrostics_14b_${SEED}_$(date +%Y%m%d_%H%M%S)"

mkdir -p "$RUN_ROOT"/{sft_data,sft_models,grpo_models,eval}

echo "========================================================"
echo "14B ICW run: ${RUN_ROOT}"
echo "Base model: ${BASE_MODEL}"
echo "========================================================"

# ── Step 1: Base eval ─────────────────────────────────────────────────────────
python grpo_train.py \
    --eval-only "$BASE_MODEL" \
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
    --gen-batch-size 2 \
    --max-new-tokens 200 \
    --seed "$SEED" \
    --eval-output-dir "$RUN_ROOT/eval/base"

# ── Step 2: Generate SFT data (teacher = 14B itself) ─────────────────────────
python generate_sft_data.py \
    --model "$MODEL" \
    --warm-start-model "$TEACHER_MODEL" \
    --method "$METHOD" \
    --secret-sequence "$SECRET" \
    --dataset "$TRAIN_DATASET" \
    --split train \
    --samples 500 \
    --n-candidates 32 \
    --top-k 2 \
    --min-score 1.0 \
    --max-new-tokens 200 \
    --temperature 0.7 \
    --top-p 0.9 \
    --implicit-fraction 0.0 \
    --gen-batch-size 2 \
    --strict-acrostics \
    --seed "$SEED" \
    --output-dir "$RUN_ROOT/sft_data"

export SFT_DATA=$(find "$RUN_ROOT/sft_data" -maxdepth 1 -type f -name "acrostics_full_*.json" | sort | tail -n 1)
if [ -z "$SFT_DATA" ]; then
    echo "ERROR: no SFT data file found" >&2; exit 1
fi
export SFT_SAMPLES=$(python -c 'import json,sys; print(len(json.load(open(sys.argv[1]))["records"]))' "$SFT_DATA")
echo "SFT records: $SFT_SAMPLES"

# ── Step 3: SFT training ──────────────────────────────────────────────────────
# Use a smaller LoRA rank to stay within memory
python sft_train.py \
    --model "$MODEL" \
    --method "$METHOD" \
    --train-dataset "$TRAIN_DATASET" \
    --sft-data "$SFT_DATA" \
    --samples "$SFT_SAMPLES" \
    --epochs 5 \
    --batch-size 1 \
    --gradient-accumulation-steps 4 \
    --learning-rate 2e-5 \
    --max-length 1024 \
    --use-lora \
    --lora-rank 16 \
    --no-wm-instruction \
    --secret-sequence "$SECRET" \
    --seed "$SEED" \
    --output-dir "$RUN_ROOT/sft_models"

export SFT_MODEL=$(find "$RUN_ROOT/sft_models" -type d -path "*/sft_acrostics_full_*/final_model" | sort | tail -n 1)
if [ -z "$SFT_MODEL" ]; then
    echo "ERROR: no SFT model found" >&2; exit 1
fi
echo "SFT model: $SFT_MODEL"

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
    --gen-batch-size 2 \
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
    --batch-size 2 \
    --gradient-accumulation-steps 2 \
    --learning-rate 1e-5 \
    --num-generations 4 \
    --implicit-fraction "$GRPO_IMPLICIT_FRACTION" \
    --beta 0.04 \
    --allow-implicit-reference \
    --max-new-tokens 200 \
    --gen-batch-size 2 \
    --eval-splits "" \
    --train-dataset "$TRAIN_DATASET" \
    --seed "$SEED" \
    --output-dir "$RUN_ROOT/grpo_models"

export GRPO_MODEL=$(find "$RUN_ROOT/grpo_models" -type d -path "*/acrostics_full_*/final_model" | sort | tail -n 1)
if [ -z "$GRPO_MODEL" ]; then
    echo "ERROR: no GRPO model found" >&2; exit 1
fi
echo "GRPO model: $GRPO_MODEL"

# ── Step 6: Final eval ────────────────────────────────────────────────────────
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
    --gen-batch-size 2 \
    --max-new-tokens 200 \
    --seed "$SEED" \
    --eval-output-dir "$RUN_ROOT/eval/grpo"

echo ""
echo "========================================================"
echo "14B run complete: ${RUN_ROOT}"
echo "  base: $RUN_ROOT/eval/base/"
echo "  sft:  $RUN_ROOT/eval/sft/"
echo "  grpo: $RUN_ROOT/eval/grpo/"
echo "========================================================"

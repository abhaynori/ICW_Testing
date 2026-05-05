#!/bin/bash
set -euo pipefail

export METHOD=acrostics
export MODEL=full
export TEACHER_MODEL=Qwen/Qwen2.5-14B-Instruct
export SECRET=SECRET
export TRAIN_DATASET=eli5
export SEED=42
export GRPO_IMPLICIT_FRACTION=1.0
export RUN_ROOT="rerun_acrostics_${SEED}_$(date +%Y%m%d_%H%M%S)"

mkdir -p "$RUN_ROOT"/{sft_data,sft_models,grpo_models,eval,robustness,temp,sysprompt}

python grpo_train.py \
    --eval-only Qwen/Qwen2.5-7B-Instruct \
    --model "$MODEL" \
    --method "$METHOD" \
    --secret-sequence "$SECRET" \
    --train-dataset "$TRAIN_DATASET" \
    --samples 200 \
    --eval-datasets eli5,alpaca \
    --eval-splits validation,test \
    --eval-samples 200 \
    --eval-profiles natural,controlled \
    --eval-modes implicit,explicit \
    --gen-batch-size 4 \
    --max-new-tokens 200 \
    --controlled-min-new-tokens 128 \
    --seed "$SEED" \
    --eval-output-dir "$RUN_ROOT/eval/base"

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
    --gen-batch-size 4 \
    --strict-acrostics \
    --seed "$SEED" \
    --output-dir "$RUN_ROOT/sft_data"

export SFT_DATA=$(find "$RUN_ROOT/sft_data" -maxdepth 1 -type f -name "acrostics_full_*.json" | sort | tail -n 1)
echo "$SFT_DATA"
if [ -z "$SFT_DATA" ]; then
    echo "ERROR: no SFT data file found under $RUN_ROOT/sft_data" >&2
    exit 1
fi

python sft_train.py \
    --model "$MODEL" \
    --method "$METHOD" \
    --train-dataset "$TRAIN_DATASET" \
    --sft-data "$SFT_DATA" \
    --samples 500 \
    --epochs 1 \
    --batch-size 2 \
    --learning-rate 2e-5 \
    --max-length 1024 \
    --use-lora \
    --secret-sequence "$SECRET" \
    --seed "$SEED" \
    --output-dir "$RUN_ROOT/sft_models"

export SFT_MODEL=$(find "$RUN_ROOT/sft_models" -type d -path "*/sft_acrostics_full_*/final_model" | sort | tail -n 1)
echo "$SFT_MODEL"
if [ -z "$SFT_MODEL" ]; then
    echo "ERROR: no SFT model found under $RUN_ROOT/sft_models" >&2
    exit 1
fi

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
echo "$GRPO_MODEL"
if [ -z "$GRPO_MODEL" ]; then
    echo "ERROR: no GRPO model found under $RUN_ROOT/grpo_models" >&2
    exit 1
fi

python grpo_train.py \
    --eval-only "$GRPO_MODEL" \
    --model "$MODEL" \
    --method "$METHOD" \
    --secret-sequence "$SECRET" \
    --train-dataset "$TRAIN_DATASET" \
    --samples 200 \
    --eval-datasets eli5,alpaca \
    --eval-splits validation,test \
    --eval-samples 200 \
    --eval-profiles natural,controlled \
    --eval-modes implicit,explicit \
    --gen-batch-size 4 \
    --max-new-tokens 200 \
    --controlled-min-new-tokens 128 \
    --seed "$SEED" \
    --eval-output-dir "$RUN_ROOT/eval/grpo"

mapfile -t EVAL_JSONS < <(find "$RUN_ROOT/eval/grpo" -maxdepth 1 -type f -name "eval_*_test_*_natural.json" | sort)
if [ "${#EVAL_JSONS[@]}" -eq 0 ]; then
    echo "ERROR: no natural test eval JSON files found under $RUN_ROOT/eval/grpo" >&2
    exit 1
fi

for EVAL_JSON in "${EVAL_JSONS[@]}"; do
    python robustness_eval.py \
      --eval-json "$EVAL_JSON" \
      --method "$METHOD" \
      --secret-sequence "$SECRET" \
      --attacks format_cleanup,truncate_sentence_50,truncate_word_50,sentence_merge,sentence_split,word_dropout,compression
done

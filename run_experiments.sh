#!/bin/bash
set -euo pipefail

# ============================================================================
# ICW Results Table Pipeline
# Produces: Baseline / Baseline+Sys-prompt / Baseline+SFT / Baseline+SFT+GRPO
# Sweeps:   SFT samples × SFT epochs, GRPO samples × GRPO epochs
#
# Usage:
#   bash run_experiments.sh              # full pipeline
#   bash run_experiments.sh --skip-datagen  # reuse existing SFT data
# ============================================================================

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOGDIR="experiment_logs/${TIMESTAMP}"
mkdir -p "$LOGDIR"

# --- Sweep configuration ---------------------------------------------------
METHOD="acrostics"
MODEL="full"
BASE_MODEL="Qwen/Qwen2.5-7B-Instruct"

SFT_SAMPLES_SWEEP=(100 500 1000 2000)
SFT_EPOCHS_SWEEP=(1 3 5)
GRPO_SAMPLES_SWEEP=(50 100 200)
GRPO_EPOCHS_SWEEP=(1 3)

EVAL_SAMPLES=50
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
echo "ICW Results Table Pipeline — ${TIMESTAMP}"
echo "Method:       ${METHOD}"
echo "SFT samples:  ${SFT_SAMPLES_SWEEP[*]}"
echo "SFT epochs:   ${SFT_EPOCHS_SWEEP[*]}"
echo "GRPO samples: ${GRPO_SAMPLES_SWEEP[*]}"
echo "GRPO epochs:  ${GRPO_EPOCHS_SWEEP[*]}"
echo "Logs:         ${LOGDIR}/"
echo "============================================================"

# Helper: run eval on a model and save to a log file
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
# PHASE 1: Baseline + Baseline+Sys-prompt  (Rows 1 & 2)
# ============================================================================
# A single eval-only run on the base model produces both:
#   - implicit scores → Baseline row
#   - explicit scores → Baseline+Sys-prompt row

echo ""
echo "============================================================"
echo "PHASE 1: Evaluating base model (Baseline + Baseline+Sys-prompt)"
echo "============================================================"

eval_model "$BASE_MODEL" "${LOGDIR}/01_eval_baseline.log"

# ============================================================================
# PHASE 2: Generate SFT data  (once, max samples needed)
# ============================================================================

MAX_SFT_SAMPLES=${SFT_SAMPLES_SWEEP[-1]}  # largest value in sweep

if [ "$SKIP_DATAGEN" = false ]; then
    echo ""
    echo "============================================================"
    echo "PHASE 2: Generating RS-SFT data (${MAX_SFT_SAMPLES} samples)"
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
    echo "ERROR: No SFT data file found in sft_data/. Exiting."
    exit 1
fi
echo ">>> Using SFT data: ${SFT_DATA}"

# ============================================================================
# PHASE 3: SFT sweep  (Row 3: Baseline+SFT)
# ============================================================================

echo ""
echo "============================================================"
echo "PHASE 3: SFT training sweep"
echo "============================================================"

mkdir -p "${LOGDIR}/sft"

for N_SAMPLES in "${SFT_SAMPLES_SWEEP[@]}"; do
    for N_EPOCHS in "${SFT_EPOCHS_SWEEP[@]}"; do
        TAG="sft_s${N_SAMPLES}_e${N_EPOCHS}"
        echo ""
        echo ">>> SFT: samples=${N_SAMPLES}, epochs=${N_EPOCHS}"

        # --- Train ---
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
            2>&1 | tee "${LOGDIR}/sft/${TAG}_train.log"

        # Find the model just created (most recent)
        SFT_MODEL=$(ls -td sft_models/sft_${METHOD}_${MODEL}_*/final_model 2>/dev/null | head -1)
        if [ -z "$SFT_MODEL" ]; then
            echo "WARNING: SFT model not found for ${TAG}, skipping eval."
            continue
        fi
        echo ">>> Model: ${SFT_MODEL}"

        # --- Eval ---
        eval_model "$SFT_MODEL" "${LOGDIR}/sft/${TAG}_eval.log"
    done
done

# ============================================================================
# PHASE 4: GRPO sweep on top of best SFT  (Row 4: Baseline+SFT+GRPO)
# ============================================================================
# For each SFT checkpoint, run GRPO with varying samples × epochs.

echo ""
echo "============================================================"
echo "PHASE 4: GRPO training sweep (warm-started from SFT)"
echo "============================================================"

mkdir -p "${LOGDIR}/grpo"

# Iterate over all SFT models produced in Phase 3
for SFT_DIR in sft_models/sft_${METHOD}_${MODEL}_*/final_model; do
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
                --implicit-fraction 0.4 \
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

# ============================================================================
# PHASE 5: Collect results into table
# ============================================================================

echo ""
echo "============================================================"
echo "PHASE 5: Collecting results"
echo "============================================================"

python - <<'PYTHON_SCRIPT'
import json, glob, os, re, sys

def extract_z_scores(log_path):
    """Parse eval log to extract z-scores per (dataset, split, mode)."""
    results = {}
    current_key = None
    with open(log_path) as f:
        for line in f:
            # Match lines like: --- eli5:validation [implicit] (instruction=no) ---
            m = re.search(r'--- (\w+):(\w+) \[(\w+)\]', line)
            if m:
                current_key = (m.group(1), m.group(2), m.group(3))
            # Match z-score line
            m2 = re.search(r'Mean score vs training baseline \(z\): ([\-\+]?\d+\.?\d*)', line)
            if m2 and current_key:
                results[current_key] = float(m2.group(1))
            # Match raw mean score
            m3 = re.search(r'Mean score: ([\-\+]?\d+\.?\d*)', line)
            if m3 and current_key:
                results[current_key + ("raw",)] = float(m3.group(1))
            # Match baseline
            m4 = re.search(r'Baseline computed: mean=([\d.]+), std=([\d.]+)', line)
            if m4:
                results[("baseline", "mean")] = float(m4.group(1))
                results[("baseline", "std")] = float(m4.group(2))
    return results

def fmt(v):
    if v is None:
        return "—"
    return f"{v:+.4f}"

logdir = max(glob.glob("experiment_logs/*/"), key=os.path.getmtime).rstrip("/")
print(f"\nResults from: {logdir}\n")

# Baseline
bl_log = os.path.join(logdir, "01_eval_baseline.log")
if os.path.exists(bl_log):
    bl = extract_z_scores(bl_log)
    bl_mean = bl.get(("baseline", "mean"))
    bl_std = bl.get(("baseline", "std"))

    print(f"Baseline normalization: mean={bl_mean}, std={bl_std}")
    print()
    print(f"{'Model':<45} {'ELI5 train':>12} {'ELI5 val':>12} {'Alpaca val':>12}")
    print("-" * 85)

    # Row 1: Baseline (implicit)
    print(f"{'Baseline':<45} "
          f"{fmt(bl.get(('eli5','train','implicit'))):>12} "
          f"{fmt(bl.get(('eli5','validation','implicit'))):>12} "
          f"{fmt(bl.get(('alpaca','validation','implicit'))):>12}")

    # Row 2: Baseline + Sys-prompt (explicit)
    print(f"{'Baseline + Sys-prompt':<45} "
          f"{fmt(bl.get(('eli5','train','explicit'))):>12} "
          f"{fmt(bl.get(('eli5','validation','explicit'))):>12} "
          f"{fmt(bl.get(('alpaca','validation','explicit'))):>12}")

# SFT rows
sft_logs = sorted(glob.glob(os.path.join(logdir, "sft", "*_eval.log")))
for log_path in sft_logs:
    tag = os.path.basename(log_path).replace("_eval.log", "")
    m = re.search(r's(\d+)_e(\d+)', tag)
    label = f"SFT (n={m.group(1)}, ep={m.group(2)})" if m else tag
    r = extract_z_scores(log_path)
    print(f"{label:<45} "
          f"{fmt(r.get(('eli5','train','implicit'))):>12} "
          f"{fmt(r.get(('eli5','validation','implicit'))):>12} "
          f"{fmt(r.get(('alpaca','validation','implicit'))):>12}")

# GRPO rows
grpo_logs = sorted(glob.glob(os.path.join(logdir, "grpo", "*_eval.log")))
for log_path in grpo_logs:
    tag = os.path.basename(log_path).replace("_eval.log", "")
    # Extract SFT and GRPO params from tag
    label = tag.replace("grpo_sft_acrostics_full_", "SFT+GRPO ")
    r = extract_z_scores(log_path)
    print(f"{label:<45} "
          f"{fmt(r.get(('eli5','train','implicit'))):>12} "
          f"{fmt(r.get(('eli5','validation','implicit'))):>12} "
          f"{fmt(r.get(('alpaca','validation','implicit'))):>12}")

print()
PYTHON_SCRIPT

# ============================================================================
# DONE
# ============================================================================

echo ""
echo "============================================================"
echo "ALL EXPERIMENTS COMPLETE"
echo "============================================================"
echo "Logs:       ${LOGDIR}/"
echo "SFT data:   ${SFT_DATA}"
echo "Results:    see table above"
echo "============================================================"

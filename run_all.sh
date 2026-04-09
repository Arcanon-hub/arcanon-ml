#!/bin/bash
set -e # Exit on any error

echo "--- Arcanon ML Master Handler ---"

# 1. Validation: Ensure HF_TOKEN is present
if [ -z "$HF_TOKEN" ]; then
    echo "[ERROR] HF_TOKEN is not set. Cannot stream data or push to hub."
    exit 1
fi

# 2. Run Data Check (Optional, but recommended)
if [ "$SKIP_CHECK" != "true" ]; then
    echo "Running Data Verification..."
    python data_check.py --n 5
fi

# 3. Phase 1: Domain-Adaptive MLM
if [ "$RUN_PHASE_1" = "true" ]; then
    echo "Starting Phase 1 (MLM)..."
    python train_mlm.py \
        --push_to_hub \
        --hub_model_id "$HF_REPO_MLM" \
        --report_to "${REPORT_TO:-none}" \
        --subset "default" \
        --fp16
fi

# 4. Phase 2: Classification (SBC)
# Requires labels.jsonl to be present in /app or a mounted volume
if [ "$RUN_PHASE_2" = "true" ]; then
    if [ ! -f "labels.jsonl" ]; then
        echo "[ERROR] labels.jsonl not found. Phase 2 cannot proceed."
        exit 1
    fi
    echo "Starting Phase 2 (Classification)..."
    python train_classifier.py \
        --train_file labels.jsonl \
        --push_to_hub \
        --hub_model_id "$HF_REPO_CLASSIFIER" \
        --report_to "${REPORT_TO:-none}" \
        --fp16
fi

# 5. Export to ONNX
if [ "$RUN_EXPORT" = "true" ]; then
    echo "Exporting to Quantized ONNX..."
    EXPORT_ARGS="--model_dir ./classifier_checkpoints --output_dir ./onnx_export"
    
    if [ "$PUSH_ONNX" = "true" ]; then
        EXPORT_ARGS="$EXPORT_ARGS --push_to_hub --hub_model_id $HF_REPO_ONNX"
    fi

    python export_to_onnx.py $EXPORT_ARGS
fi

echo "--- All Requested Tasks Completed ---"

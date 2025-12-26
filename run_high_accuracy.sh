#!/bin/bash
# run_high_accuracy.sh - Training for >60% Fragment Accuracy
# Optimized for 4GB GPU with maximum accuracy

set -e

DATA_DIR="/home/anmol_kamath/ML HACKATHON/DataRepo/data/puzzle_3x3"
WORK_DIR="/home/anmol_kamath/ML HACKATHON/puzzle-shufflenet"
OUTPUT_DIR="${WORK_DIR}/outputs"

mkdir -p "${OUTPUT_DIR}"

echo "=============================================="
echo "HIGH ACCURACY JIGSAW PUZZLE TRAINING (V4)"
echo "Target: >60% Fragment Accuracy"
echo "=============================================="
echo ""
echo "Configuration:"
echo "  - Training samples: 15000"
echo "  - Epochs: 25"
echo "  - Batch size: 8 x 4 = 32 effective"
echo "  - Model: V4 Lite (MobileNetV3 + 3-layer Transformer)"
echo "  - Feature dim: 192"
echo "  - Label smoothing: 0.1"
echo ""

# Training
echo "=============================================="
echo "STEP 1: Training (ETA: 30-40 minutes)"
echo "=============================================="

python3 "${WORK_DIR}/train_v4.py" \
    --image_dir "${DATA_DIR}/train" \
    --manifest "${DATA_DIR}/train.csv" \
    --subset 15000 \
    --val_split 0.1 \
    --model_type lite \
    --feature_dim 192 \
    --num_layers 3 \
    --num_heads 6 \
    --epochs 25 \
    --batch_size 8 \
    --accum_steps 4 \
    --lr 2e-4 \
    --weight_decay 0.05 \
    --label_smoothing 0.1 \
    --patience 8 \
    --num_workers 4 \
    --out "${OUTPUT_DIR}/best_model_v4.pth" \
    --debug

echo ""
echo "Training complete!"
echo ""

# Evaluation
echo "=============================================="
echo "STEP 2: Evaluating on Validation Set"
echo "=============================================="

python3 "${WORK_DIR}/evaluate_v4.py" \
    --image_dir "${DATA_DIR}/valid" \
    --manifest "${DATA_DIR}/valid.csv" \
    --weights "${OUTPUT_DIR}/best_model_v4.pth" \
    --show_errors

echo ""

# Predictions
echo "=============================================="
echo "STEP 3: Generating Test Predictions"
echo "=============================================="

python3 "${WORK_DIR}/predict_v4.py" \
    --image_dir "${DATA_DIR}/test" \
    --manifest "${DATA_DIR}/test.csv" \
    --weights "${OUTPUT_DIR}/best_model_v4.pth" \
    --out "${OUTPUT_DIR}/predictions_v4.csv" \
    --tta

echo ""
echo "=============================================="
echo "COMPLETE!"
echo "=============================================="
echo "Model: ${OUTPUT_DIR}/best_model_v4.pth"
echo "Predictions: ${OUTPUT_DIR}/predictions_v4.csv"
echo ""

#!/bin/bash

# Machine Intelligence Node - Model Evaluation Script
# Runs AI model evaluation with logging, performance tracking, and GPU acceleration.
#
# Author: Machine Intelligence Node Development Team

# Default evaluation parameters
MODEL_PATH="models/checkpoint.pth"
CONFIG_PATH="configs/eval_config.yaml"
LOG_DIR="logs"
DATASET_PATH="data/test_dataset"
BATCH_SIZE=32
DEVICE="cuda"

# Create log directory if it doesn't exist
mkdir -p $LOG_DIR

# Check if CUDA is available, fallback to CPU if necessary
if ! command -v nvidia-smi &> /dev/null; then
    echo "[INFO] CUDA not detected. Running on CPU."
    DEVICE="cpu"
fi

# Start evaluation
echo "[INFO] Starting model evaluation..."
python3 src/evaluation/evaluator.py \
    --config $CONFIG_PATH \
    --model_path $MODEL_PATH \
    --dataset $DATASET_PATH \
    --batch_size $BATCH_SIZE \
    --device $DEVICE | tee $LOG_DIR/evaluation.log

# Check for errors
if [ $? -ne 0 ]; then
    echo "[ERROR] Evaluation failed. Check logs for details."
    exit 1
fi

echo "[INFO] Evaluation complete. Results saved in $LOG_DIR/evaluation.log"

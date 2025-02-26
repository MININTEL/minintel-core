#!/bin/bash

# Machine Intelligence Node - Training Script
# Automates AI model training with logging, checkpointing, and GPU acceleration.
#
# Author: Machine Intelligence Node Development Team

# Default training parameters
MODEL_PATH="models/checkpoint.pth"
CONFIG_PATH="configs/train_config.yaml"
LOG_DIR="logs"
BATCH_SIZE=32
EPOCHS=50
DEVICE="cuda"

# Create log directory if it doesn't exist
mkdir -p $LOG_DIR

# Check if CUDA is available, fallback to CPU if necessary
if ! command -v nvidia-smi &> /dev/null; then
    echo "[INFO] CUDA not detected. Running on CPU."
    DEVICE="cpu"
fi

# Start training
echo "[INFO] Starting training..."
python3 src/training/trainer.py \
    --config $CONFIG_PATH \
    --model_path $MODEL_PATH \
    --batch_size $BATCH_SIZE \
    --epochs $EPOCHS \
    --device $DEVICE | tee $LOG_DIR/training.log

# Check for errors
if [ $? -ne 0 ]; then
    echo "[ERROR] Training failed. Check logs for details."
    exit 1
fi

echo "[INFO] Training complete. Model saved to $MODEL_PATH"

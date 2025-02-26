#!/bin/bash

# Machine Intelligence Node - Container Startup Script
# Initializes the AI model server inside a Docker container.

echo "[INFO] Starting Machine Intelligence Node container..."

# Check if Python environment is set up
if ! command -v python3 &> /dev/null; then
    echo "[ERROR] Python is not installed. Exiting."
    exit 1
fi

# Run the AI model server
echo "[INFO] Launching AI model..."
python3 src/main.py --config configs/default.yaml

# Keep the container alive
tail -f /dev/null

"""
Machine Intelligence Node - Example Inference Script

This script demonstrates how to run inference using a trained AI model 
from Machine Intelligence Node.

Author: Machine Intelligence Node Development Team
"""

import torch
import argparse
import logging
import numpy as np
from src.core.model import Model
from src.utils.config import load_config

# Set up logging
logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(message)s")

# Argument parser for inference parameters
parser = argparse.ArgumentParser(description="Example Inference Script for Machine Intelligence Node")
parser.add_argument("--config", type=str, default="configs/default.yaml", help="Path to model configuration file")
parser.add_argument("--model_path", type=str, default="models/checkpoint.pth", help="Path to trained model")
parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device for inference (cuda or cpu)")
args = parser.parse_args()

# Load model configuration
config = load_config(args.config)
device = args.device

# Load AI model
logging.info(f"Loading model from {args.model_path}...")
model = Model(config["model"])
model.load_state_dict(torch.load(args.model_path, map_location=device))
model.to(device)
model.eval()

# Define a sample input (random data)
input_shape = (1, config["model"]["input_size"])
sample_input = torch.randn(input_shape).to(device)

# Run inference
logging.info("Running inference...")
with torch.no_grad():
    output = model(sample_input)
    predicted_class = torch.argmax(output, dim=-1).cpu().numpy()

logging.info(f"Inference complete. Predicted output: {predicted_class}")

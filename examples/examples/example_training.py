"""
Machine Intelligence Node - Example Training Script

This script demonstrates how to train an AI model using Machine Intelligence Node.
It loads a dataset, configures training parameters, and logs performance.

Author: Machine Intelligence Node Development Team
"""

import torch
import argparse
import logging
from src.core.model import Model
from src.training.trainer import Trainer
from src.utils.config import load_config

# Set up logging
logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(message)s")

# Argument parser for training parameters
parser = argparse.ArgumentParser(description="Example Training Script for Machine Intelligence Node")
parser.add_argument("--config", type=str, default="configs/training_config.json", help="Path to training configuration file")
parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device for training (cuda or cpu)")
args = parser.parse_args()

# Load training configuration
config = load_config(args.config)
device = args.device

# Load AI model
logging.info("Initializing model...")
model = Model(config["model"])
model.to(device)

# Initialize trainer
trainer = Trainer(
    model=model,
    config=config["training"],
    device=device
)

# Start training
logging.info("Starting training process...")
trainer.train()

logging.info("Training complete. Model saved to {}".format(config["model"]["model_path"]))

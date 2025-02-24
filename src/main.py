"""
Machine Intelligence Node - Main Execution Pipeline

Handles AI model initialization, API integration, and inference execution.

Author: Machine Intelligence Node Development Team
"""

import argparse
import os
import torch
from fastapi import FastAPI
from threading import Thread
from src.core.model import Model
from src.api.rest_api import start_rest_api
from src.api.websocket_server import start_websocket_server
from src.utils.config import load_config
from src.utils.logger import Logger

# Initialize application logger
logger = Logger(log_file="logs/server.log")

# Load configuration
CONFIG = load_config("config.yaml")

# Initialize FastAPI application
app = FastAPI()

def initialize_model():
    """
    Loads and initializes the AI model for inference.
    """
    logger.info("Initializing AI model...")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = Model(CONFIG["model"]["path"], device=device)
    
    logger.info(f"Model loaded successfully on {device}.")
    return model

def start_services(model):
    """
    Starts the REST API and WebSocket server.
    
    Args:
        model (Model): Initialized AI model instance.
    """
    logger.info("Starting API services...")

    # Start REST API in a separate thread
    rest_api_thread = Thread(target=start_rest_api, args=(model,))
    rest_api_thread.daemon = True
    rest_api_thread.start()

    # Start WebSocket server
    start_websocket_server(model)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Start Machine Intelligence Node.")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config file")
    
    args = parser.parse_args()
    CONFIG = load_config(args.config)
    
    model = initialize_model()
    start_services(model)

    logger.info("Machine Intelligence Node is now running.")

"""
Machine Intelligence Node - Core Framework Initialization

This file marks the 'src' directory as a package, allowing for modular imports
across different components of the framework.

Author: Machine Intelligence Node Development Team
"""

__version__ = "0.1.0"
__author__ = "Machine Intelligence Node Development Team"

# Ensure critical submodules are accessible at package level
from .core import model, optimizer, scheduler
from .data import dataset_loader, preprocessor, augmentation
from .utils import config, logger, metrics
from .training import trainer, loss_functions, callbacks
from .inference import inference_engine, postprocessing
from .experiments import benchmark, testing_pipeline

# Define public API for the package
__all__ = [
    "model", "optimizer", "scheduler",
    "dataset_loader", "preprocessor", "augmentation",
    "config", "logger", "metrics",
    "trainer", "loss_functions", "callbacks",
    "inference_engine", "postprocessing",
    "benchmark", "testing_pipeline"
]

# Optional: Log initialization for debugging
import logging
logging.basicConfig(level=logging.INFO)
logging.info("Machine Intelligence Node package initialized.")

"""
Machine Intelligence Node - Training Module

Handles AI model training, loss computation, and optimization strategies.

Author: Machine Intelligence Node Development Team
"""

from .trainer import Trainer
from .loss_functions import LossFunctions
from .callbacks import TrainingCallbacks
from .hyperparameter_tuner import HyperparameterTuner

__all__ = [
    "Trainer",
    "LossFunctions",
    "TrainingCallbacks",
    "HyperparameterTuner"
]

def initialize_training():
    """
    Initializes and returns the training components for AI model development.

    Returns:
        dict: Dictionary containing initialized training components.
    """
    trainer = Trainer()
    loss_fn = LossFunctions()
    callbacks = TrainingCallbacks()
    tuner = HyperparameterTuner()

    return {
        "trainer": trainer,
        "loss_functions": loss_fn,
        "callbacks": callbacks,
        "hyperparameter_tuner": tuner
    }

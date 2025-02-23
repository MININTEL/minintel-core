"""
Machine Intelligence Node - Data Processing Module

This module initializes the full data pipeline, handling dataset loading,
preprocessing, augmentation, and train-validation splitting.

Author: Machine Intelligence Node Development Team
"""

import os
import json
import torch
from torch.utils.data import DataLoader, random_split
from .dataset_loader import DatasetLoader
from .preprocessor import DataPreprocessor
from .augmentation import DataAugmentation
from .validation_split import train_validation_split

# Define available exports for package-level imports
__all__ = [
    "DatasetLoader",
    "DataPreprocessor",
    "DataAugmentation",
    "train_validation_split",
    "initialize_data_pipeline"
]

def initialize_data_pipeline(config):
    """
    Initializes the dataset pipeline based on provided configurations.

    Args:
        config (dict): Configuration dictionary defining dataset paths, 
                       augmentation strategies, preprocessing settings, 
                       and batch loading parameters.

    Returns:
        DataLoader: Configured dataset loader instance.
    """

    # Load dataset
    dataset_loader = DatasetLoader(config["dataset_path"])
    
    # Initialize preprocessing and augmentation modules
    preprocessor = DataPreprocessor(config["preprocessing"])
    augmentation = DataAugmentation(config["augmentation"])
    
    # Apply preprocessing and augmentation if enabled
    if config["enable_preprocessing"]:
        dataset_loader.apply_preprocessing(preprocessor)
    
    if config["enable_augmentation"]:
        dataset_loader.apply_augmentation(augmentation)
    
    # Perform train-validation split
    train_dataset, val_dataset = train_validation_split(dataset_loader.get_dataset(), config["train_split_ratio"])
    
    # Enable lazy dataset loading if required (useful for large datasets)
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config["batch_size"], 
        shuffle=True, 
        num_workers=config.get("num_workers", 4),
        pin_memory=config.get("pin_memory", True)
    )

    val_loader = DataLoader(
        val_dataset, 
        batch_size=config["batch_size"], 
        shuffle=False, 
        num_workers=config.get("num_workers", 4),
        pin_memory=config.get("pin_memory", True)
    )

    return train_loader, val_loader

def load_config(config_path):
    """
    Loads JSON configuration for dataset preprocessing and augmentation.

    Args:
        config_path (str): Path to the configuration JSON file.

    Returns:
        dict: Configuration dictionary.
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with open(config_path, "r") as f:
        config = json.load(f)
    
    return config

def save_config(config, config_path):
    """
    Saves the current dataset processing configuration to a JSON file.

    Args:
        config (dict): Configuration dictionary.
        config_path (str): Path to save the JSON file.
    """
    with open(config_path, "w") as f:
        json.dump(config, f, indent=4)

# Example usage
if __name__ == "__main__":
    example_config = {
        "dataset_path": "data/train_dataset.json",
        "enable_preprocessing": True,
        "preprocessing": {
            "normalize": True,
            "tokenize": True,
            "remove_stopwords": False
        },
        "enable_augmentation": True,
        "augmentation": {
            "random_flip": True,
            "random_crop": False,
            "color_jitter": False
        },
        "train_split_ratio": 0.8,
        "batch_size": 32,
        "num_workers": 4,
        "pin_memory": True
    }

    save_config(example_config, "config.json")
    
    config = load_config("config.json")
    train_loader, val_loader = initialize_data_pipeline(config)
    
    print(f"Train Loader: {len(train_loader)} batches")
    print(f"Validation Loader: {len(val_loader)} batches")

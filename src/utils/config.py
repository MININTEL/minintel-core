"""
Machine Intelligence Node - Configuration Loader

Handles dynamic configuration parsing from YAML, JSON, 
and environment variables for AI training and inference.

Author: Machine Intelligence Node Development Team
"""

import os
import json
import yaml
from typing import Any, Dict, Optional

class ConfigLoader:
    """
    A flexible configuration loader supporting YAML, JSON, and environment variable overrides.
    """
    def __init__(self, config_path: str, default_config: Optional[Dict[str, Any]] = None):
        """
        Initializes the configuration loader.

        Args:
            config_path (str): Path to the configuration file.
            default_config (Dict[str, Any], optional): Default fallback values.
        """
        self.config_path = config_path
        self.default_config = default_config or {}
        self.config = self.load_config()

    def load_config(self) -> Dict[str, Any]:
        """
        Loads configuration from YAML or JSON file.

        Returns:
            Dict[str, Any]: Parsed configuration dictionary.
        """
        if not os.path.exists(self.config_path):
            print(f"Warning: Config file {self.config_path} not found. Using default values.")
            return self.default_config

        with open(self.config_path, "r") as file:
            if self.config_path.endswith(".yaml") or self.config_path.endswith(".yml"):
                return yaml.safe_load(file)
            elif self.config_path.endswith(".json"):
                return json.load(file)
            else:
                raise ValueError("Unsupported configuration format. Use YAML or JSON.")

    def get(self, key: str, default: Any = None) -> Any:
        """
        Retrieves a configuration value with support for nested keys.

        Args:
            key (str): Configuration key (supports dot notation).
            default (Any): Default value if the key is missing.

        Returns:
            Any: Configuration value or default.
        """
        keys = key.split(".")
        value = self.config
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return os.getenv(key.upper(), default)  # Fallback to environment variable
        return value

    def set(self, key: str, value: Any):
        """
        Updates a configuration value dynamically.

        Args:
            key (str): Configuration key.
            value (Any): New value to set.
        """
        keys = key.split(".")
        config = self.config
        for k in keys[:-1]:
            if k not in config or not isinstance(config[k], dict):
                config[k] = {}
            config = config[k]
        config[keys[-1]] = value

    def save_config(self, output_path: Optional[str] = None):
        """
        Saves the updated configuration to a file.

        Args:
            output_path (str, optional): Path to save the configuration file.
        """
        output_path = output_path or self.config_path
        with open(output_path, "w") as file:
            if output_path.endswith(".yaml") or output_path.endswith(".yml"):
                yaml.dump(self.config, file, default_flow_style=False)
            elif output_path.endswith(".json"):
                json.dump(self.config, file, indent=4)
            else:
                raise ValueError("Unsupported output format. Use YAML or JSON.")

# Example Usage
if __name__ == "__main__":
    example_config = {
        "training": {
            "batch_size": 32,
            "learning_rate": 5e-5,
            "epochs": 10
        },
        "inference": {
            "device": "cuda",
            "precision": "fp16"
        }
    }

    # Save default config
    config_path = "configs/default.yaml"
    with open(config_path, "w") as file:
        yaml.dump(example_config, file, default_flow_style=False)

    # Load config and retrieve values
    config_loader = ConfigLoader(config_path)
    batch_size = config_loader.get("training.batch_size", 16)
    precision = config_loader.get("inference.precision", "fp32")

    print(f"Batch Size: {batch_size}")
    print(f"Inference Precision: {precision}")

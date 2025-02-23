"""
Machine Intelligence Node - File Utilities

Provides safe and efficient file operations, including JSON/YAML serialization, 
concurrent file handling, and large-scale data storage support.

Author: Machine Intelligence Node Development Team
"""

import os
import json
import yaml
import shutil
import threading
from typing import Any, Dict, Optional

class FileHandler:
    """
    A flexible file handling class supporting JSON/YAML serialization, 
    directory management, and thread-safe read/write operations.
    """
    _lock = threading.Lock()  # Ensures thread-safe file operations

    @staticmethod
    def ensure_directory_exists(directory: str):
        """
        Ensures a directory exists, creating it if necessary.

        Args:
            directory (str): Path to the directory.
        """
        os.makedirs(directory, exist_ok=True)

    @staticmethod
    def read_json(file_path: str) -> Optional[Dict[str, Any]]:
        """
        Safely reads a JSON file and returns its contents.

        Args:
            file_path (str): Path to the JSON file.

        Returns:
            Dict[str, Any]: Parsed JSON data or None if file not found.
        """
        if not os.path.exists(file_path):
            return None
        
        with FileHandler._lock, open(file_path, "r", encoding="utf-8") as file:
            try:
                return json.load(file)
            except json.JSONDecodeError:
                return None  # Return None if JSON is malformed

    @staticmethod
    def write_json(file_path: str, data: Dict[str, Any]):
        """
        Writes data to a JSON file safely.

        Args:
            file_path (str): Path to the JSON file.
            data (Dict[str, Any]): Data to write.
        """
        with FileHandler._lock, open(file_path, "w", encoding="utf-8") as file:
            json.dump(data, file, indent=4)

    @staticmethod
    def read_yaml(file_path: str) -> Optional[Dict[str, Any]]:
        """
        Safely reads a YAML file and returns its contents.

        Args:
            file_path (str): Path to the YAML file.

        Returns:
            Dict[str, Any]: Parsed YAML data or None if file not found.
        """
        if not os.path.exists(file_path):
            return None
        
        with FileHandler._lock, open(file_path, "r", encoding="utf-8") as file:
            try:
                return yaml.safe_load(file)
            except yaml.YAMLError:
                return None  # Return None if YAML is malformed

    @staticmethod
    def write_yaml(file_path: str, data: Dict[str, Any]):
        """
        Writes data to a YAML file safely.

        Args:
            file_path (str): Path to the YAML file.
            data (Dict[str, Any]): Data to write.
        """
        with FileHandler._lock, open(file_path, "w", encoding="utf-8") as file:
            yaml.dump(data, file, default_flow_style=False)

    @staticmethod
    def copy_file(source: str, destination: str):
        """
        Copies a file from source to destination.

        Args:
            source (str): Source file path.
            destination (str): Destination file path.
        """
        shutil.copy2(source, destination)

    @staticmethod
    def delete_file(file_path: str):
        """
        Deletes a file safely.

        Args:
            file_path (str): Path to the file to delete.
        """
        if os.path.exists(file_path):
            os.remove(file_path)

    @staticmethod
    def list_files(directory: str, extension: Optional[str] = None) -> list:
        """
        Lists all files in a directory with an optional file extension filter.

        Args:
            directory (str): Directory path.
            extension (str, optional): Filter files by extension (e.g., '.json').

        Returns:
            list: List of file paths.
        """
        if not os.path.exists(directory):
            return []

        return [
            os.path.join(directory, f)
            for f in os.listdir(directory)
            if os.path.isfile(os.path.join(directory, f)) and (extension is None or f.endswith(extension))
        ]

# Example Usage
if __name__ == "__main__":
    # Ensure directory exists
    FileHandler.ensure_directory_exists("data")

    # JSON operations
    sample_data = {"model": "Transformer", "accuracy": 0.98}
    json_path = "data/model_config.json"
    FileHandler.write_json(json_path, sample_data)
    loaded_json = FileHandler.read_json(json_path)
    print(f"Loaded JSON: {loaded_json}")

    # YAML operations
    yaml_path = "data/config.yaml"
    FileHandler.write_yaml(yaml_path, sample_data)
    loaded_yaml = FileHandler.read_yaml(yaml_path)
    print(f"Loaded YAML: {loaded_yaml}")

    # File management
    FileHandler.copy_file(json_path, "data/model_config_backup.json")
    print(f"Files in data/: {FileHandler.list_files('data')}")

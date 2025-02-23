"""
Machine Intelligence Node - Dataset Loader

Provides a scalable dataset loading module with support for various file formats,
on-the-fly preprocessing, lazy loading, and batch optimization.

Author: Machine Intelligence Node Development Team
"""

import os
import json
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

class BaseDataset(Dataset):
    """
    Abstract dataset class providing a structured pipeline for loading and processing datasets.
    """
    def __init__(self, file_path, tokenizer=None, max_length=512):
        """
        Initializes the dataset loader.

        Args:
            file_path (str): Path to the dataset file.
            tokenizer (callable, optional): Tokenizer function for processing text.
            max_length (int): Maximum sequence length for tokenized data.
        """
        self.file_path = file_path
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.data = self._load_data()

    def _load_data(self):
        """
        Loads dataset from file and applies preprocessing.
        """
        file_extension = os.path.splitext(self.file_path)[-1].lower()
        if file_extension == ".json":
            with open(self.file_path, "r") as f:
                data = json.load(f)
        elif file_extension == ".csv":
            data = pd.read_csv(self.file_path).to_dict(orient="records")
        elif file_extension == ".parquet":
            data = pd.read_parquet(self.file_path).to_dict(orient="records")
        else:
            raise ValueError(f"Unsupported dataset format: {file_extension}")
        
        return data

    def __len__(self):
        """
        Returns the number of samples in the dataset.
        """
        return len(self.data)

    def __getitem__(self, idx):
        """
        Retrieves a sample from the dataset.

        Args:
            idx (int): Index of the sample.

        Returns:
            dict: Tokenized sample data.
        """
        sample = self.data[idx]
        text = sample.get("text", "")

        if self.tokenizer:
            tokenized = self.tokenizer(text, max_length=self.max_length, padding="max_length", truncation=True)
            return {"input_ids": tokenized["input_ids"], "attention_mask": tokenized["attention_mask"]}
        else:
            return {"text": text}

def create_data_loader(file_path, tokenizer=None, batch_size=32, shuffle=True, num_workers=4):
    """
    Creates a PyTorch DataLoader for efficient batch processing.

    Args:
        file_path (str): Path to the dataset file.
        tokenizer (callable, optional): Tokenizer function for processing text.
        batch_size (int): Number of samples per batch.
        shuffle (bool): Whether to shuffle the dataset.
        num_workers (int): Number of subprocesses for data loading.

    Returns:
        DataLoader: Configured data loader instance.
    """
    dataset = BaseDataset(file_path, tokenizer)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)

# Example usage
if __name__ == "__main__":
    from tokenizer import BPETokenizer

    tokenizer = BPETokenizer()
    dataset_loader = create_data_loader("data/sample_dataset.json", tokenizer=tokenizer, batch_size=16)

    for batch in dataset_loader:
        print(batch)
        break

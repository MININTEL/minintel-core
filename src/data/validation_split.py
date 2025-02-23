"""
Machine Intelligence Node - Dataset Splitter

Handles dataset partitioning into training, validation, and test sets 
with stratified sampling and reproducibility controls.

Author: Machine Intelligence Node Development Team
"""

import random
import numpy as np
import torch
from typing import Tuple
from torch.utils.data import Dataset, Subset

class DatasetSplitter:
    """
    A dataset splitting utility that supports random and stratified splits.
    """
    def __init__(self, train_ratio: float = 0.8, val_ratio: float = 0.1, test_ratio: float = 0.1, stratify: bool = False, seed: int = 42):
        """
        Initializes dataset splitting parameters.

        Args:
            train_ratio (float): Proportion of the dataset for training.
            val_ratio (float): Proportion of the dataset for validation.
            test_ratio (float): Proportion of the dataset for testing.
            stratify (bool): If True, performs stratified sampling.
            seed (int): Random seed for reproducibility.
        """
        assert train_ratio + val_ratio + test_ratio == 1.0, "Splits must sum to 1.0"
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.stratify = stratify
        self.seed = seed
        random.seed(self.seed)
        np.random.seed(self.seed)

    def split(self, dataset: Dataset, labels: np.ndarray = None) -> Tuple[Subset, Subset, Subset]:
        """
        Splits the dataset into train, validation, and test sets.

        Args:
            dataset (Dataset): The dataset to split.
            labels (np.ndarray, optional): Labels for stratified splitting.

        Returns:
            Tuple[Subset, Subset, Subset]: Train, validation, and test subsets.
        """
        dataset_size = len(dataset)
        indices = np.arange(dataset_size)

        if self.stratify and labels is not None:
            from sklearn.model_selection import train_test_split
            
            train_indices, temp_indices, _, temp_labels = train_test_split(
                indices, labels, stratify=labels, test_size=(self.val_ratio + self.test_ratio), random_state=self.seed
            )

            val_ratio_adjusted = self.val_ratio / (self.val_ratio + self.test_ratio)
            val_indices, test_indices = train_test_split(temp_indices, stratify=temp_labels, test_size=(1 - val_ratio_adjusted), random_state=self.seed)

        else:
            np.random.shuffle(indices)
            train_end = int(self.train_ratio * dataset_size)
            val_end = train_end + int(self.val_ratio * dataset_size)
            
            train_indices = indices[:train_end]
            val_indices = indices[train_end:val_end]
            test_indices = indices[val_end:]

        return Subset(dataset, train_indices), Subset(dataset, val_indices), Subset(dataset, test_indices)

# Example usage
if __name__ == "__main__":
    from torch.utils.data import TensorDataset
    
    # Simulating a dataset with labels
    data = torch.randn(1000, 10)
    labels = np.random.randint(0, 2, size=(1000,))  # Binary classification labels
    dataset = TensorDataset(data, torch.tensor(labels))

    splitter = DatasetSplitter(train_ratio=0.75, val_ratio=0.15, test_ratio=0.10, stratify=True)
    train_set, val_set, test_set = splitter.split(dataset, labels)

    print(f"Train size: {len(train_set)}, Validation size: {len(val_set)}, Test size: {len(test_set)}")

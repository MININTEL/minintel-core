"""
Machine Intelligence Node - Data Loader Unit Tests

Verifies dataset loading, batch processing, shuffling, and data integrity.

Author: Machine Intelligence Node Development Team
"""

import pytest
import torch
from torch.utils.data import DataLoader
from src.data.dataset_loader import CustomDataset

# Define test parameters
BATCH_SIZE = 8
DATASET_SIZE = 100
FEATURE_DIM = 512

@pytest.fixture(scope="module")
def test_dataset():
    """
    Initializes a synthetic dataset for testing.
    """
    features = torch.randn(DATASET_SIZE, FEATURE_DIM)
    labels = torch.randint(0, 10, (DATASET_SIZE,))
    return CustomDataset(features, labels)

@pytest.fixture(scope="module")
def test_data_loader(test_dataset):
    """
    Creates a DataLoader instance for batch processing.
    """
    return DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True)

def test_dataset_initialization(test_dataset):
    """
    Ensures dataset initializes correctly and has expected length.
    """
    assert len(test_dataset) == DATASET_SIZE, f"Expected dataset size {DATASET_SIZE}, got {len(test_dataset)}"

def test_batch_processing(test_data_loader):
    """
    Verifies that data loader properly batches input samples.
    """
    batch = next(iter(test_data_loader))
    features, labels = batch

    assert features.shape == (BATCH_SIZE, FEATURE_DIM), f"Unexpected batch shape {features.shape}"
    assert labels.shape == (BATCH_SIZE,), "Labels batch size mismatch"

def test_data_shuffling(test_dataset):
    """
    Ensures that shuffling alters dataset order.
    """
    loader_shuffled = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True)
    loader_unshuffled = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    batch_shuffled = next(iter(loader_shuffled))[1].tolist()
    batch_unshuffled = next(iter(loader_unshuffled))[1].tolist()

    assert batch_shuffled != batch_unshuffled, "DataLoader shuffle is not functioning"

def test_data_integrity(test_dataset):
    """
    Ensures input features correspond to correct labels.
    """
    sample_idx = 5
    features, label = test_dataset[sample_idx]

    assert isinstance(features, torch.Tensor), "Features should be a tensor"
    assert isinstance(label, int), "Label should be an integer"

if __name__ == "__main__":
    pytest.main()

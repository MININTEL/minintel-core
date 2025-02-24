"""
Machine Intelligence Node - Scheduler Unit Tests

Verifies learning rate scheduling, warmup steps, and decay logic.

Author: Machine Intelligence Node Development Team
"""

import pytest
import torch
from src.training.scheduler import LearningRateScheduler

# Define test parameters
INITIAL_LR = 0.01
DECAY_FACTOR = 0.1
WARMUP_STEPS = 5
TOTAL_STEPS = 50

@pytest.fixture(scope="module")
def test_optimizer():
    """
    Initializes a dummy optimizer for testing scheduler functionality.
    """
    model = torch.nn.Linear(512, 10)
    optimizer = torch.optim.Adam(model.parameters(), lr=INITIAL_LR)
    return optimizer

@pytest.fixture(scope="module")
def test_scheduler(test_optimizer):
    """
    Initializes the learning rate scheduler.
    """
    return LearningRateScheduler(test_optimizer, warmup_steps=WARMUP_STEPS, decay_factor=DECAY_FACTOR)

def test_scheduler_initialization(test_scheduler):
    """
    Ensures scheduler initializes correctly.
    """
    assert test_scheduler is not None, "Scheduler failed to initialize."

def test_learning_rate_warmup(test_scheduler, test_optimizer):
    """
    Checks if learning rate increases during warmup.
    """
    initial_lr = test_optimizer.param_groups[0]["lr"]

    for step in range(WARMUP_STEPS):
        test_scheduler.step(step)

    warmup_lr = test_optimizer.param_groups[0]["lr"]
    assert warmup_lr > initial_lr, f"Learning rate did not increase during warmup. Initial: {initial_lr}, Warmup: {warmup_lr}"

def test_learning_rate_decay(test_scheduler, test_optimizer):
    """
    Verifies if learning rate decreases after warmup.
    """
    for step in range(WARMUP_STEPS, TOTAL_STEPS):
        test_scheduler.step(step)

    final_lr = test_optimizer.param_groups[0]["lr"]
    expected_lr = INITIAL_LR * (DECAY_FACTOR ** ((TOTAL_STEPS - WARMUP_STEPS) // 10))  # Assuming decay step every 10 steps

    assert final_lr <= expected_lr, f"Learning rate did not decay properly. Expected: {expected_lr}, Got: {final_lr}"

def test_learning_rate_zero(test_scheduler, test_optimizer):
    """
    Ensures scheduler handles zero learning rate edge case.
    """
    for param_group in test_optimizer.param_groups:
        param_group["lr"] = 0.0

    test_scheduler.step(TOTAL_STEPS)
    assert test_optimizer.param_groups[0]["lr"] == 0.0, "Learning rate should remain zero but changed."

if __name__ == "__main__":
    pytest.main()

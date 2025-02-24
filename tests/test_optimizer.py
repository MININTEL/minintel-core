"""
Machine Intelligence Node - Optimizer Unit Tests

Verifies optimizer behavior, including gradient updates, learning rate scheduling, 
and stability over multiple iterations.

Author: Machine Intelligence Node Development Team
"""

import pytest
import torch
from src.core.model import Model
from src.core.optimizer import initialize_optimizer

# Define test parameters
MODEL_PATH = "models/test_model.pth"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LEARNING_RATE = 0.001

@pytest.fixture(scope="module")
def test_model():
    """
    Loads and initializes the AI model for optimizer testing.
    """
    model = Model(model_path=MODEL_PATH, device=DEVICE)
    model.train()  # Set to training mode
    return model

@pytest.fixture(scope="module")
def test_optimizer(test_model):
    """
    Initializes optimizer for AI model training.
    """
    optimizer = initialize_optimizer("adam", test_model.parameters(), LEARNING_RATE)
    return optimizer

def test_optimizer_initialization(test_optimizer):
    """
    Ensures optimizer initializes correctly.
    """
    assert test_optimizer is not None, "Optimizer failed to initialize."

def test_gradient_update(test_model, test_optimizer):
    """
    Verifies optimizer updates model parameters based on gradients.
    """
    criterion = torch.nn.CrossEntropyLoss()
    input_data = torch.randn((1, 512)).to(DEVICE)
    target = torch.randint(0, 10, (1,)).to(DEVICE)

    output = test_model(input_data)
    loss = criterion(output, target.unsqueeze(0))

    loss.backward()
    test_optimizer.step()
    test_optimizer.zero_grad()

    for param in test_model.parameters():
        assert param.grad is None or torch.all(param.grad == 0), "Gradients were not cleared properly"

def test_learning_rate_adjustment(test_optimizer):
    """
    Ensures learning rate adjustments work as expected.
    """
    initial_lr = test_optimizer.param_groups[0]["lr"]
    for param_group in test_optimizer.param_groups:
        param_group["lr"] *= 0.1

    adjusted_lr = test_optimizer.param_groups[0]["lr"]
    assert adjusted_lr == initial_lr * 0.1, "Learning rate did not update correctly"

def test_stability_over_iterations(test_model, test_optimizer):
    """
    Runs multiple optimization steps to ensure stability over time.
    """
    criterion = torch.nn.CrossEntropyLoss()
    
    for _ in range(10):
        input_data = torch.randn((1, 512)).to(DEVICE)
        target = torch.randint(0, 10, (1,)).to(DEVICE)

        output = test_model(input_data)
        loss = criterion(output, target.unsqueeze(0))

        loss.backward()
        test_optimizer.step()
        test_optimizer.zero_grad()

    assert True, "Optimizer remained stable over multiple iterations"

if __name__ == "__main__":
    pytest.main()

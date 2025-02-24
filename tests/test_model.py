"""
Machine Intelligence Node - AI Model Unit Tests

Verifies AI model behavior, including forward pass, gradient updates, 
and deterministic output consistency.

Author: Machine Intelligence Node Development Team
"""

import pytest
import torch
from src.core.model import Model

# Define test parameters
MODEL_PATH = "models/test_model.pth"
INPUT_SHAPE = (1, 512)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

@pytest.fixture(scope="module")
def test_model():
    """
    Loads and initializes the AI model for testing.
    """
    model = Model(model_path=MODEL_PATH, device=DEVICE)
    model.eval()  # Set to evaluation mode
    return model

def test_model_initialization(test_model):
    """
    Ensures model initializes correctly.
    """
    assert test_model is not None, "Model failed to initialize."

def test_forward_pass(test_model):
    """
    Validates AI model forward pass output shape.
    """
    input_data = torch.randn(INPUT_SHAPE).to(DEVICE)
    output = test_model(input_data)
    assert output.shape[-1] == 10, f"Expected output shape (batch, 10), got {output.shape}"

def test_gradient_update(test_model):
    """
    Ensures gradients propagate correctly during backpropagation.
    """
    optimizer = torch.optim.Adam(test_model.parameters(), lr=0.001)
    criterion = torch.nn.CrossEntropyLoss()

    input_data = torch.randn(INPUT_SHAPE).to(DEVICE)
    target = torch.randint(0, 10, (1,)).to(DEVICE)

    output = test_model(input_data)
    loss = criterion(output, target.unsqueeze(0))
    loss.backward()

    optimizer.step()
    optimizer.zero_grad()

    for param in test_model.parameters():
        assert param.grad is not None, "Gradient update failed"

def test_deterministic_outputs(test_model):
    """
    Ensures AI model produces deterministic outputs for the same input.
    """
    input_data = torch.randn(INPUT_SHAPE).to(DEVICE)

    output1 = test_model(input_data).detach().cpu().numpy()
    output2 = test_model(input_data).detach().cpu().numpy()

    assert (output1 == output2).all(), "Model outputs should be deterministic"

if __name__ == "__main__":
    pytest.main()

"""
Machine Intelligence Node - Inference Unit Tests

Verifies AI model inference execution, consistency, and latency.

Author: Machine Intelligence Node Development Team
"""

import pytest
import torch
import time
from src.core.model import Model

# Define test parameters
MODEL_PATH = "models/test_model.pth"
INPUT_SHAPE = (1, 512)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

@pytest.fixture(scope="module")
def test_model():
    """
    Loads and initializes the AI model for inference testing.
    """
    model = Model(model_path=MODEL_PATH, device=DEVICE)
    model.eval()  # Set to evaluation mode
    return model

def test_model_inference(test_model):
    """
    Ensures AI model successfully executes inference.
    """
    input_data = torch.randn(INPUT_SHAPE).to(DEVICE)
    output = test_model(input_data)

    assert output is not None, "Inference output is None"
    assert output.shape[-1] == 10, f"Expected output shape (batch, 10), got {output.shape}"

def test_inference_consistency(test_model):
    """
    Ensures AI model returns consistent results for identical inputs.
    """
    input_data = torch.randn(INPUT_SHAPE).to(DEVICE)

    output1 = test_model(input_data).detach().cpu().numpy()
    output2 = test_model(input_data).detach().cpu().numpy()

    assert (output1 == output2).all(), "Model inference outputs are inconsistent"

def test_inference_latency(test_model):
    """
    Measures AI model inference speed to ensure real-time performance.
    """
    input_data = torch.randn(INPUT_SHAPE).to(DEVICE)

    start_time = time.time()
    _ = test_model(input_data)
    end_time = time.time()

    latency = end_time - start_time
    assert latency < 0.1, f"Inference latency too high: {latency:.4f} sec"

def test_invalid_input(test_model):
    """
    Ensures AI model handles invalid input gracefully.
    """
    with pytest.raises(RuntimeError):
        invalid_input = torch.randint(0, 255, (1, 512)).to(DEVICE)  # Non-float input
        _ = test_model(invalid_input)

if __name__ == "__main__":
    pytest.main()

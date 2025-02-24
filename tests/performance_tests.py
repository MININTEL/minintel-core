"""
Machine Intelligence Node - AI Performance Benchmarking

Evaluates inference speed, memory usage, and multi-threaded execution.

Author: Machine Intelligence Node Development Team
"""

import pytest
import torch
import time
import psutil
from concurrent.futures import ThreadPoolExecutor
from src.core.model import Model

# Define test parameters
MODEL_PATH = "models/test_model.pth"
INPUT_SHAPE = (1, 512)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 32

@pytest.fixture(scope="module")
def test_model():
    """
    Loads and initializes the AI model for performance testing.
    """
    model = Model(model_path=MODEL_PATH, device=DEVICE)
    model.eval()
    return model

def test_inference_speed(test_model):
    """
    Measures AI model inference latency.
    """
    input_data = torch.randn(INPUT_SHAPE).to(DEVICE)

    start_time = time.time()
    _ = test_model(input_data)
    end_time = time.time()

    latency = (end_time - start_time) * 1000  # Convert to milliseconds
    assert latency < 50, f"Inference latency too high: {latency:.2f}ms"

def test_memory_usage(test_model):
    """
    Checks GPU/CPU memory usage during inference.
    """
    initial_mem = psutil.virtual_memory().used / (1024 * 1024)  # Convert to MB
    input_data = torch.randn(INPUT_SHAPE).to(DEVICE)

    _ = test_model(input_data)

    final_mem = psutil.virtual_memory().used / (1024 * 1024)
    mem_increase = final_mem - initial_mem

    assert mem_increase < 100, f"Memory usage too high: {mem_increase:.2f}MB"

def test_multi_threaded_inference(test_model):
    """
    Runs inference in parallel threads to measure scalability.
    """
    input_data = [torch.randn(INPUT_SHAPE).to(DEVICE) for _ in range(8)]

    def infer(data):
        return test_model(data)

    with ThreadPoolExecutor(max_workers=8) as executor:
        results = list(executor.map(infer, input_data))

    assert all(result is not None for result in results), "Parallel inference failed"

def test_batch_processing(test_model):
    """
    Evaluates model performance on large batch inputs.
    """
    input_data = torch.randn((BATCH_SIZE, INPUT_SHAPE[1])).to(DEVICE)

    start_time = time.time()
    _ = test_model(input_data)
    end_time = time.time()

    batch_latency = (end_time - start_time) * 1000
    assert batch_latency < 200, f"Batch inference too slow: {batch_latency:.2f}ms"

if __name__ == "__main__":
    pytest.main()

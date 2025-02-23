"""
Machine Intelligence Node - Inference Benchmarking Tool

Evaluates AI inference performance by measuring latency, throughput, 
and memory usage in real-world scenarios.

Author: Machine Intelligence Node Development Team
"""

import time
import numpy as np
import torch
import psutil
import concurrent.futures
from typing import Dict, List
from src.inference.inference_engine import InferenceEngine
from src.utils.profiler import Profiler

class InferenceBenchmark:
    """
    A benchmarking tool for evaluating AI inference performance, 
    including latency, throughput, and memory efficiency.
    """
    def __init__(self, model_path: str, backend: str = "torch", device: str = "cuda"):
        """
        Initializes the inference benchmarking tool.

        Args:
            model_path (str): Path to the AI model file.
            backend (str): Backend framework ('torch', 'onnx').
            device (str): Execution device ('cuda' or 'cpu').
        """
        self.engine = InferenceEngine(model_path, backend, device)
        self.profiler = Profiler()

    def measure_latency(self, sample_input: np.ndarray, runs: int = 50) -> float:
        """
        Measures the average latency per inference request.

        Args:
            sample_input (np.ndarray): Sample input for inference.
            runs (int): Number of inference runs for averaging.

        Returns:
            float: Average latency in milliseconds.
        """
        start_time = time.time()
        for _ in range(runs):
            self.engine.predict(sample_input)
        elapsed_time = time.time() - start_time
        return (elapsed_time / runs) * 1000  # Convert to milliseconds

    def measure_throughput(self, batch_inputs: List[np.ndarray], duration: int = 5) -> float:
        """
        Measures model inference throughput (requests per second).

        Args:
            batch_inputs (List[np.ndarray]): Batch of input data.
            duration (int): Benchmarking time window in seconds.

        Returns:
            float: Throughput in requests per second.
        """
        start_time = time.time()
        requests = 0

        while time.time() - start_time < duration:
            self.engine.batch_predict(batch_inputs, num_threads=4)
            requests += len(batch_inputs)

        return requests / duration

    def measure_memory_usage(self) -> Dict[str, float]:
        """
        Measures CPU and GPU memory utilization.

        Returns:
            Dict[str, float]: CPU and GPU memory usage statistics.
        """
        cpu_memory = psutil.virtual_memory().used / (1024 ** 3)  # Convert to GB
        gpu_memory = self.profiler.profile_gpu_memory().get("gpu_memory_usage", 0.0)

        return {
            "cpu_memory_usage": cpu_memory,
            "gpu_memory_usage": gpu_memory,
        }

    def run_benchmark(self, sample_input: np.ndarray, batch_inputs: List[np.ndarray]):
        """
        Executes the full inference benchmarking pipeline.

        Args:
            sample_input (np.ndarray): Sample input for latency measurement.
            batch_inputs (List[np.ndarray]): Batch of input data for throughput testing.
        """
        print("Running Inference Benchmark...")

        latency = self.measure_latency(sample_input)
        print(f"Avg Latency per request: {latency:.2f} ms")

        throughput = self.measure_throughput(batch_inputs)
        print(f"Throughput: {throughput:.2f} requests/sec")

        memory_usage = self.measure_memory_usage()
        print(f"Memory Usage (CPU): {memory_usage['cpu_memory_usage']:.2f} GB")
        print(f"Memory Usage (GPU): {memory_usage['gpu_memory_usage']:.2f} GB")

# Example Usage
if __name__ == "__main__":
    model_path = "models/machine_intelligence.onnx"
    benchmark = InferenceBenchmark(model_path, backend="onnx", device="cuda")

    # Generate synthetic input data
    sample_input = np.random.rand(1, 512).astype(np.float32)
    batch_inputs = [np.random.rand(1, 512).astype(np.float32) for _ in range(16)]

    benchmark.run_benchmark(sample_input, batch_inputs)

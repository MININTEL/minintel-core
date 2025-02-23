"""
Machine Intelligence Node - AI Profiler

Provides real-time profiling of CPU/GPU utilization, memory usage, and execution times 
for AI training and inference workloads.

Author: Machine Intelligence Node Development Team
"""

import time
import logging
import torch
import psutil
import json
from typing import Dict, Optional

try:
    import pynvml  # NVIDIA Management Library for GPU monitoring
    pynvml.nvmlInit()
    NVML_AVAILABLE = True
except ImportError:
    NVML_AVAILABLE = False

class Profiler:
    """
    A high-performance profiler for tracking AI model execution times, 
    memory usage, and CPU/GPU resource consumption.
    """
    def __init__(self, log_to_file: Optional[str] = None, json_output: bool = False):
        """
        Initializes the profiler.

        Args:
            log_to_file (str, optional): Path to save profiler logs (default: None).
            json_output (bool): Whether to log in JSON format for structured monitoring.
        """
        self.logger = logging.getLogger("Profiler")
        self.json_output = json_output

        if log_to_file:
            handler = logging.FileHandler(log_to_file)
            handler.setFormatter(logging.Formatter("%(asctime)s - %(message)s"))
            self.logger.addHandler(handler)

    def profile_cpu_memory(self) -> Dict[str, float]:
        """
        Profiles CPU usage and system memory consumption.

        Returns:
            Dict[str, float]: CPU usage (%) and memory utilization (GB).
        """
        return {
            "cpu_usage": psutil.cpu_percent(interval=1),
            "memory_usage": psutil.virtual_memory().used / (1024 ** 3),  # Convert bytes to GB
        }

    def profile_gpu_memory(self) -> Dict[str, float]:
        """
        Profiles GPU usage and memory consumption using NVIDIA NVML.

        Returns:
            Dict[str, float]: GPU utilization (%) and memory usage (GB), or empty dict if NVML unavailable.
        """
        if not NVML_AVAILABLE:
            return {}

        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        memory_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        gpu_utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)

        return {
            "gpu_usage": gpu_utilization.gpu,  # GPU utilization percentage
            "gpu_memory_usage": memory_info.used / (1024 ** 3),  # Convert bytes to GB
        }

    def profile_execution_time(self, function, *args, **kwargs) -> float:
        """
        Measures the execution time of a given function.

        Args:
            function: Function to profile.
            *args: Positional arguments for the function.
            **kwargs: Keyword arguments for the function.

        Returns:
            float: Execution time in seconds.
        """
        start_time = time.time()
        result = function(*args, **kwargs)
        elapsed_time = time.time() - start_time
        return elapsed_time, result

    def profile_torch_memory(self) -> Dict[str, float]:
        """
        Profiles PyTorch memory usage, including allocated and cached GPU memory.

        Returns:
            Dict[str, float]: GPU memory allocation and cache stats.
        """
        if not torch.cuda.is_available():
            return {}

        return {
            "torch_allocated_memory": torch.cuda.memory_allocated() / (1024 ** 3),  # Convert bytes to GB
            "torch_reserved_memory": torch.cuda.memory_reserved() / (1024 ** 3),  # Convert bytes to GB
        }

    def log_metrics(self, metrics: Dict[str, float]):
        """
        Logs profiling metrics in either text or JSON format.

        Args:
            metrics (Dict[str, float]): Dictionary of profiling results.
        """
        if self.json_output:
            self.logger.info(json.dumps(metrics))
        else:
            self.logger.info(" | ".join([f"{key}: {value:.2f}" for key, value in metrics.items()]))

# Example Usage
if __name__ == "__main__":
    profiler = Profiler(log_to_file="logs/profiler.log", json_output=True)

    # Profile CPU and GPU resources
    cpu_metrics = profiler.profile_cpu_memory()
    gpu_metrics = profiler.profile_gpu_memory()
    profiler.log_metrics({**cpu_metrics, **gpu_metrics})

    # Profile PyTorch memory usage
    torch_metrics = profiler.profile_torch_memory()
    profiler.log_metrics(torch_metrics)

    # Profile execution time of a sample function
    def sample_function():
        time.sleep(1)
        return "Function complete"

    exec_time, result = profiler.profile_execution_time(sample_function)
    profiler.log_metrics({"execution_time": exec_time})
    print(result)

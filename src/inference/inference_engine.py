"""
Machine Intelligence Node - Inference Engine

Handles real-time AI model inference, batch processing, and multi-threaded 
execution for optimized performance.

Author: Machine Intelligence Node Development Team
"""

import torch
import onnxruntime as ort
import numpy as np
import concurrent.futures
from typing import Union, List, Dict, Optional

class InferenceEngine:
    """
    A high-performance AI inference engine supporting multi-threaded execution, 
    dynamic model loading, and batch processing.
    """
    def __init__(self, model_path: str, backend: str = "torch", device: str = "cuda"):
        """
        Initializes the inference engine.

        Args:
            model_path (str): Path to the AI model file.
            backend (str): Backend framework ('torch', 'onnx', 'tf').
            device (str): Execution device ('cuda' or 'cpu').
        """
        self.backend = backend.lower()
        self.device = device if torch.cuda.is_available() else "cpu"
        self.model = self.load_model(model_path)

    def load_model(self, model_path: str):
        """
        Loads the AI model from a given path.

        Args:
            model_path (str): Path to the model file.

        Returns:
            Model object loaded in the specified backend.
        """
        if self.backend == "torch":
            model = torch.jit.load(model_path, map_location=self.device)
            model.eval()
            return model

        elif self.backend == "onnx":
            return ort.InferenceSession(model_path, providers=["CUDAExecutionProvider" if self.device == "cuda" else "CPUExecutionProvider"])

        else:
            raise ValueError("Unsupported backend. Use 'torch' or 'onnx'.")

    def predict(self, inputs: Union[np.ndarray, List[np.ndarray]]) -> np.ndarray:
        """
        Runs inference on a single input or batch of inputs.

        Args:
            inputs (Union[np.ndarray, List[np.ndarray]]): Input data for inference.

        Returns:
            np.ndarray: Model output predictions.
        """
        if isinstance(inputs, list):
            inputs = np.stack(inputs)

        if self.backend == "torch":
            inputs_tensor = torch.tensor(inputs, dtype=torch.float32, device=self.device)
            with torch.no_grad():
                output_tensor = self.model(inputs_tensor)
            return output_tensor.cpu().numpy()

        elif self.backend == "onnx":
            input_name = self.model.get_inputs()[0].name
            output = self.model.run(None, {input_name: inputs})
            return np.array(output)

    def batch_predict(self, batch_inputs: List[np.ndarray], num_threads: int = 4) -> List[np.ndarray]:
        """
        Runs batch inference using multi-threaded execution.

        Args:
            batch_inputs (List[np.ndarray]): List of input data batches.
            num_threads (int): Number of threads for parallel execution.

        Returns:
            List[np.ndarray]: List of model output predictions.
        """
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
            results = list(executor.map(self.predict, batch_inputs))
        return results

# Example Usage
if __name__ == "__main__":
    model_path = "models/machine_intelligence.onnx"
    engine = InferenceEngine(model_path, backend="onnx", device="cuda")

    sample_input = np.random.rand(1, 512).astype(np.float32)
    prediction = engine.predict(sample_input)
    print(f"Single Inference Output: {prediction}")

    batch_inputs = [np.random.rand(1, 512).astype(np.float32) for _ in range(4)]
    batch_output = engine.batch_predict(batch_inputs, num_threads=4)
    print(f"Batch Inference Output: {batch_output}")

"""
Machine Intelligence Node - Postprocessing Module

Handles AI model output normalization, thresholding, and transformation 
for structured inference results.

Author: Machine Intelligence Node Development Team
"""

import numpy as np
from typing import List, Union, Dict

class PostProcessor:
    """
    A flexible AI inference postprocessing pipeline supporting normalization, 
    confidence thresholding, and structured output transformation.
    """
    def __init__(self, threshold: float = 0.5, normalization: str = "softmax"):
        """
        Initializes the postprocessing pipeline.

        Args:
            threshold (float): Confidence threshold for filtering outputs.
            normalization (str): Normalization technique ('softmax' or 'minmax').
        """
        self.threshold = threshold
        self.normalization = normalization

    def normalize(self, outputs: np.ndarray) -> np.ndarray:
        """
        Applies normalization to raw inference outputs.

        Args:
            outputs (np.ndarray): Raw model outputs.

        Returns:
            np.ndarray: Normalized outputs.
        """
        if self.normalization == "softmax":
            exp_outputs = np.exp(outputs - np.max(outputs, axis=-1, keepdims=True))
            return exp_outputs / np.sum(exp_outputs, axis=-1, keepdims=True)

        elif self.normalization == "minmax":
            return (outputs - np.min(outputs, axis=-1, keepdims=True)) / (
                np.max(outputs, axis=-1, keepdims=True) - np.min(outputs, axis=-1, keepdims=True) + 1e-10
            )

        else:
            raise ValueError("Unsupported normalization method. Choose 'softmax' or 'minmax'.")

    def apply_threshold(self, outputs: np.ndarray) -> np.ndarray:
        """
        Filters outputs based on a confidence threshold.

        Args:
            outputs (np.ndarray): Normalized outputs.

        Returns:
            np.ndarray: Thresholded outputs with low-confidence predictions removed.
        """
        return (outputs > self.threshold).astype(float)

    def transform_output(self, outputs: np.ndarray, labels: List[str]) -> List[Dict[str, Union[str, float]]]:
        """
        Transforms raw model outputs into structured results.

        Args:
            outputs (np.ndarray): Postprocessed model predictions.
            labels (List[str]): Label names for classification tasks.

        Returns:
            List[Dict[str, Union[str, float]]]: Structured output format.
        """
        structured_results = []
        for output in outputs:
            label_idx = np.argmax(output)
            structured_results.append({
                "label": labels[label_idx],
                "confidence": float(output[label_idx])
            })
        return structured_results

    def process(self, raw_outputs: np.ndarray, labels: List[str] = None) -> Union[np.ndarray, List[Dict[str, Union[str, float]]]]:
        """
        Executes the full postprocessing pipeline.

        Args:
            raw_outputs (np.ndarray): Raw model outputs.
            labels (List[str], optional): Labels for structured transformation.

        Returns:
            Processed outputs in either raw or structured format.
        """
        normalized_outputs = self.normalize(raw_outputs)
        thresholded_outputs = self.apply_threshold(normalized_outputs)

        if labels:
            return self.transform_output(thresholded_outputs, labels)
        return thresholded_outputs

# Example Usage
if __name__ == "__main__":
    processor = PostProcessor(threshold=0.6, normalization="softmax")

    raw_predictions = np.array([[2.3, 1.5, 0.7], [0.1, 2.8, 1.4]])
    labels = ["cat", "dog", "rabbit"]

    processed_outputs = processor.process(raw_predictions, labels)
    print(f"Postprocessed Output: {processed_outputs}")

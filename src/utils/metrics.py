"""
Machine Intelligence Node - Optimized Metrics Tracker

Provides real-time tracking and evaluation of AI model performance, 
leveraging vectorized operations for high-efficiency calculations.

Author: Machine Intelligence Node Development Team
"""

import numpy as np
from typing import Dict, Union

class MetricTracker:
    """
    A high-performance metrics tracking system with support for 
    vectorized AI evaluation, including accuracy, precision, recall, and loss calculations.
    """
    def __init__(self):
        """
        Initializes the metrics tracker with storage optimized for batch processing.
        """
        self.metrics = {}

    def update(self, name: str, value: float, weight: float = 1.0):
        """
        Updates a metric with a new value, supporting rolling averages.

        Args:
            name (str): Metric name (e.g., 'accuracy', 'loss').
            value (float): New value to add to the metric.
            weight (float): Weight for averaging (default: 1.0).
        """
        if name not in self.metrics:
            self.metrics[name] = np.array([0.0, 0.0])  # [total, count]
        self.metrics[name] += np.array([value * weight, weight])

    def get(self, name: str) -> float:
        """
        Retrieves the current value of a metric.

        Args:
            name (str): Metric name.

        Returns:
            float: Computed metric value or 0 if not found.
        """
        if name in self.metrics and self.metrics[name][1] > 0:
            return self.metrics[name][0] / self.metrics[name][1]
        return 0.0

    def reset(self):
        """
        Resets all stored metrics.
        """
        self.metrics.clear()

    def compute_classification_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """
        Computes classification evaluation metrics: accuracy, precision, recall, and F1-score.

        Args:
            y_true (np.ndarray): Ground truth labels.
            y_pred (np.ndarray): Predicted labels.

        Returns:
            Dict[str, float]: Computed metric values.
        """
        true_positive = np.sum((y_pred == 1) & (y_true == 1))
        false_positive = np.sum((y_pred == 1) & (y_true == 0))
        false_negative = np.sum((y_pred == 0) & (y_true == 1))
        true_negative = np.sum((y_pred == 0) & (y_true == 0))

        accuracy = (true_positive + true_negative) / y_true.size if y_true.size > 0 else 0.0
        precision = true_positive / (true_positive + false_positive) if (true_positive + false_positive) > 0 else 0.0
        recall = true_positive / (true_positive + false_negative) if (true_positive + false_negative) > 0 else 0.0
        f1_score = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

        return {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1_score,
        }

    def compute_loss(self, predictions: np.ndarray, targets: np.ndarray, loss_type: str = "mse") -> float:
        """
        Computes loss values using vectorized operations.

        Args:
            predictions (np.ndarray): Model predictions.
            targets (np.ndarray): Ground truth values.
            loss_type (str): Type of loss function ('mse' or 'mae').

        Returns:
            float: Computed loss.
        """
        if loss_type == "mse":
            return np.mean((predictions - targets) ** 2)
        elif loss_type == "mae":
            return np.mean(np.abs(predictions - targets))
        else:
            raise ValueError("Unsupported loss type. Choose 'mse' or 'mae'.")

# Example Usage
if __name__ == "__main__":
    tracker = MetricTracker()

    # Simulating accuracy updates over multiple batches
    tracker.update("accuracy", 0.85)
    tracker.update("accuracy", 0.87)
    tracker.update("accuracy", 0.90)

    print(f"Current Accuracy: {tracker.get('accuracy'):.4f}")

    # Simulating classification metrics computation
    y_true = np.array([1, 0, 1, 1, 0, 1, 0, 1])
    y_pred = np.array([1, 0, 1, 0, 0, 1, 1, 1])
    
    metrics = tracker.compute_classification_metrics(y_true, y_pred)
    print(f"Classification Metrics: {metrics}")

    # Simulating loss computation
    predictions = np.array([0.1, 0.4, 0.8, 0.6, 0.2])
    targets = np.array([0, 0, 1, 1, 0])
    
    mse_loss = tracker.compute_loss(predictions, targets, "mse")
    print(f"MSE Loss: {mse_loss:.4f}")

"""
Machine Intelligence Node - Loss Functions

Provides predefined and custom loss functions for AI model training.

Author: Machine Intelligence Node Development Team
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Callable

class LossFunctions:
    """
    A flexible loss function module supporting standard and custom AI training losses.
    """
    def __init__(self, loss_name: str = "cross_entropy", custom_loss: Callable = None, reduction: str = "mean"):
        """
        Initializes the loss function.

        Args:
            loss_name (str): Name of the loss function ('cross_entropy', 'mse', 'mae', etc.).
            custom_loss (Callable, optional): Custom loss function provided by the user.
            reduction (str): Reduction method ('mean', 'sum', or 'none').
        """
        self.loss_name = loss_name.lower()
        self.reduction = reduction
        self.custom_loss = custom_loss
        self.loss_fn = self._initialize_loss_function()

    def _initialize_loss_function(self):
        """
        Selects the appropriate loss function based on user selection.

        Returns:
            Callable: Loss function instance.
        """
        loss_functions = {
            "cross_entropy": nn.CrossEntropyLoss(reduction=self.reduction),
            "mse": nn.MSELoss(reduction=self.reduction),
            "mae": nn.L1Loss(reduction=self.reduction),
            "huber": nn.HuberLoss(reduction=self.reduction),
            "kl_div": nn.KLDivLoss(reduction=self.reduction),
        }

        return self.custom_loss if self.custom_loss else loss_functions.get(self.loss_name, nn.CrossEntropyLoss())

    def compute_loss(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Computes the loss between predictions and ground truth.

        Args:
            predictions (torch.Tensor): Model predictions.
            targets (torch.Tensor): Ground truth labels.

        Returns:
            torch.Tensor: Computed loss value.
        """
        return self.loss_fn(predictions, targets)

    @staticmethod
    def combine_losses(losses: Dict[str, torch.Tensor], weights: Dict[str, float]) -> torch.Tensor:
        """
        Combines multiple losses using weighted sum.

        Args:
            losses (Dict[str, torch.Tensor]): Dictionary of computed losses.
            weights (Dict[str, float]): Dictionary of loss weights.

        Returns:
            torch.Tensor: Combined loss value.
        """
        total_loss = sum(weights[key] * losses[key] for key in losses if key in weights)
        return total_loss

# Example Usage
if __name__ == "__main__":
    loss_fn = LossFunctions(loss_name="mse")

    predictions = torch.randn(5, requires_grad=True)
    targets = torch.randn(5)

    loss = loss_fn.compute_loss(predictions, targets)
    print(f"Loss: {loss.item()}")

    # Example of multiple losses
    loss1 = loss_fn.compute_loss(predictions, targets)
    loss2 = LossFunctions(loss_name="mae").compute_loss(predictions, targets)
    
    combined_loss = LossFunctions.combine_losses(
        {"loss1": loss1, "loss2": loss2}, 
        {"loss1": 0.7, "loss2": 0.3}
    )
    
    print(f"Combined Loss: {combined_loss.item()}")

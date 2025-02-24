"""
Machine Intelligence Node - Training Callbacks

Provides modular training callbacks for monitoring, early stopping, 
learning rate scheduling, and checkpointing.

Author: Machine Intelligence Node Development Team
"""

import torch
import os
import logging
from typing import Optional

class EarlyStopping:
    """
    Implements early stopping to prevent overfitting during AI model training.
    """
    def __init__(self, patience: int = 5, min_delta: float = 0.001):
        """
        Initializes early stopping.

        Args:
            patience (int): Number of epochs to wait before stopping if no improvement.
            min_delta (float): Minimum change in monitored value to qualify as improvement.
        """
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = float("inf")
        self.counter = 0
        self.should_stop = False

    def __call__(self, current_loss: float):
        """
        Evaluates whether training should stop early.

        Args:
            current_loss (float): Current validation loss.

        Returns:
            bool: True if training should stop early, False otherwise.
        """
        if current_loss < self.best_loss - self.min_delta:
            self.best_loss = current_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True

        return self.should_stop

class LearningRateScheduler:
    """
    Dynamically adjusts learning rate during training.
    """
    def __init__(self, optimizer: torch.optim.Optimizer, factor: float = 0.1, patience: int = 3):
        """
        Initializes learning rate scheduler.

        Args:
            optimizer (torch.optim.Optimizer): Optimizer whose learning rate should be adjusted.
            factor (float): Factor by which learning rate should be reduced.
            patience (int): Number of epochs with no improvement before reducing LR.
        """
        self.optimizer = optimizer
        self.factor = factor
        self.patience = patience
        self.counter = 0
        self.best_loss = float("inf")

    def step(self, current_loss: float):
        """
        Adjusts learning rate based on validation loss.

        Args:
            current_loss (float): Current validation loss.
        """
        if current_loss < self.best_loss:
            self.best_loss = current_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self._reduce_lr()
                self.counter = 0

    def _reduce_lr(self):
        """
        Reduces learning rate by the predefined factor.
        """
        for param_group in self.optimizer.param_groups:
            param_group["lr"] *= self.factor
        logging.info(f"Learning rate reduced by factor {self.factor}.")

class CheckpointSaver:
    """
    Saves training checkpoints at predefined intervals.
    """
    def __init__(self, model: torch.nn.Module, optimizer: torch.optim.Optimizer, save_dir: str = "checkpoints", interval: int = 5):
        """
        Initializes checkpoint saver.

        Args:
            model (torch.nn.Module): Model being trained.
            optimizer (torch.optim.Optimizer): Optimizer used in training.
            save_dir (str): Directory to save checkpoints.
            interval (int): Number of epochs between saving checkpoints.
        """
        self.model = model
        self.optimizer = optimizer
        self.save_dir = save_dir
        self.interval = interval
        os.makedirs(save_dir, exist_ok=True)

    def save(self, epoch: int):
        """
        Saves a model checkpoint.

        Args:
            epoch (int): Current training epoch.
        """
        if epoch % self.interval == 0:
            checkpoint_path = os.path.join(self.save_dir, f"checkpoint_epoch_{epoch}.pth")
            torch.save({
                "epoch": epoch,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
            }, checkpoint_path)
            logging.info(f"Checkpoint saved: {checkpoint_path}")

# Example Usage
if __name__ == "__main__":
    class DummyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = torch.nn.Linear(512, 10)

        def forward(self, x):
            return self.fc(x)

    model = DummyModel()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Initialize Callbacks
    early_stopping = EarlyStopping(patience=5)
    lr_scheduler = LearningRateScheduler(optimizer, factor=0.5, patience=3)
    checkpoint_saver = CheckpointSaver(model, optimizer, save_dir="checkpoints", interval=5)

    # Simulated Training Loop
    for epoch in range(20):
        simulated_loss = 1.0 / (epoch + 1)  # Simulated decreasing loss

        # Early Stopping
        if early_stopping(simulated_loss):
            print("Early stopping triggered.")
            break

        # Learning Rate Adjustment
        lr_scheduler.step(simulated_loss)

        # Save Checkpoints
        checkpoint_saver.save(epoch)

"""
Machine Intelligence Node - Training Pipeline

Handles model training, optimization, gradient updates, and checkpointing.

Author: Machine Intelligence Node Development Team
"""

import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, Optional
from src.utils.logger import Logger

class Trainer:
    """
    A modular AI training pipeline supporting dynamic optimization, checkpointing, 
    and adaptive learning strategies.
    """
    def __init__(self, model: nn.Module, optimizer: str = "adam", lr: float = 1e-3, device: str = "cuda"):
        """
        Initializes the training pipeline.

        Args:
            model (nn.Module): AI model to be trained.
            optimizer (str): Optimization algorithm ('adam', 'sgd', etc.).
            lr (float): Learning rate for training.
            device (str): Training device ('cuda' or 'cpu').
        """
        self.device = device if torch.cuda.is_available() else "cpu"
        self.model = model.to(self.device)
        self.optimizer = self._initialize_optimizer(optimizer, lr)
        self.criterion = nn.CrossEntropyLoss()  # Default loss function (can be customized)
        self.logger = Logger(log_file="logs/training.log")

    def _initialize_optimizer(self, optimizer_name: str, lr: float):
        """
        Initializes the optimizer based on user selection.

        Args:
            optimizer_name (str): Name of the optimizer.
            lr (float): Learning rate.

        Returns:
            torch.optim.Optimizer: Optimizer instance.
        """
        optimizers = {
            "adam": optim.Adam(self.model.parameters(), lr=lr),
            "sgd": optim.SGD(self.model.parameters(), lr=lr, momentum=0.9),
            "rmsprop": optim.RMSprop(self.model.parameters(), lr=lr),
        }
        return optimizers.get(optimizer_name.lower(), optim.Adam(self.model.parameters(), lr=lr))

    def train(self, dataloader, epochs: int = 10, gradient_accumulation: int = 1):
        """
        Trains the AI model.

        Args:
            dataloader (DataLoader): Training dataset loader.
            epochs (int): Number of training epochs.
            gradient_accumulation (int): Steps before performing optimizer step.
        """
        self.model.train()
        for epoch in range(epochs):
            epoch_loss = 0.0

            for i, (inputs, labels) in enumerate(dataloader):
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels) / gradient_accumulation
                loss.backward()

                if (i + 1) % gradient_accumulation == 0:
                    self.optimizer.step()
                    self.optimizer.zero_grad()

                epoch_loss += loss.item()

            avg_loss = epoch_loss / len(dataloader)
            self.logger.info(f"Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f}")

    def save_checkpoint(self, checkpoint_path: str):
        """
        Saves a training checkpoint.

        Args:
            checkpoint_path (str): Path to save the model checkpoint.
        """
        torch.save({
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
        }, checkpoint_path)
        self.logger.info(f"Checkpoint saved at {checkpoint_path}")

    def load_checkpoint(self, checkpoint_path: str):
        """
        Loads a training checkpoint.

        Args:
            checkpoint_path (str): Path to the saved checkpoint.
        """
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.logger.info(f"Checkpoint loaded from {checkpoint_path}")

# Example Usage
if __name__ == "__main__":
    class DummyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(512, 10)

        def forward(self, x):
            return self.fc(x)

    model = DummyModel()
    trainer = Trainer(model, optimizer="adam", lr=0.001)

    # Example training process (replace with actual DataLoader)
    from torch.utils.data import DataLoader, TensorDataset
    import torch

    X = torch.randn(100, 512)
    y = torch.randint(0, 10, (100,))
    dataset = TensorDataset(X, y)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

    trainer.train(dataloader, epochs=5)
    trainer.save_checkpoint("checkpoints/model.pth")

"""
Machine Intelligence Node - Optimizer

Provides a configurable optimizer with learning rate scheduling, gradient clipping, 
and momentum-based weight updates for efficient deep learning training.

Author: Machine Intelligence Node Development Team
"""

import torch
import torch.optim as optim

class OptimizerFactory:
    """
    Factory class to create and configure optimizers with learning rate scheduling and gradient clipping.
    """
    def __init__(self, model, optimizer_type="AdamW", lr=1e-4, weight_decay=0.01, clip_grad_norm=1.0, scheduler_type="cosine", warmup_steps=1000):
        """
        Initializes the optimizer and scheduler.

        Args:
            model (torch.nn.Module): Model to optimize.
            optimizer_type (str): Type of optimizer to use ('AdamW', 'SGD', 'RMSprop').
            lr (float): Initial learning rate.
            weight_decay (float): Weight decay coefficient.
            clip_grad_norm (float): Max norm for gradient clipping.
            scheduler_type (str): Type of scheduler ('cosine', 'step', 'plateau').
            warmup_steps (int): Number of warmup steps for learning rate scheduling.
        """
        self.model = model
        self.optimizer = self._get_optimizer(optimizer_type, lr, weight_decay)
        self.scheduler = self._get_scheduler(scheduler_type, warmup_steps)

        self.clip_grad_norm = clip_grad_norm

    def _get_optimizer(self, optimizer_type, lr, weight_decay):
        """
        Selects and initializes the optimizer based on provided arguments.
        """
        if optimizer_type == "AdamW":
            return optim.AdamW(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        elif optimizer_type == "SGD":
            return optim.SGD(self.model.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay)
        elif optimizer_type == "RMSprop":
            return optim.RMSprop(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        else:
            raise ValueError(f"Unsupported optimizer type: {optimizer_type}")

    def _get_scheduler(self, scheduler_type, warmup_steps):
        """
        Initializes learning rate scheduler for adaptive training.
        """
        if scheduler_type == "cosine":
            return optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=warmup_steps)
        elif scheduler_type == "step":
            return optim.lr_scheduler.StepLR(self.optimizer, step_size=10, gamma=0.1)
        elif scheduler_type == "plateau":
            return optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', factor=0.5, patience=5)
        else:
            raise ValueError(f"Unsupported scheduler type: {scheduler_type}")

    def step(self):
        """
        Performs an optimizer step with gradient clipping.
        """
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_grad_norm)
        self.optimizer.step()

    def zero_grad(self):
        """
        Zeros out gradients to prevent accumulation.
        """
        self.optimizer.zero_grad()

    def update_scheduler(self, val_loss=None):
        """
        Updates learning rate scheduler based on performance.
        """
        if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
            self.scheduler.step(val_loss)
        else:
            self.scheduler.step()

# Example usage
if __name__ == "__main__":
    from model import TransformerModel

    vocab_size = 30000
    model = TransformerModel(vocab_size=vocab_size)

    optimizer_factory = OptimizerFactory(model, optimizer_type="AdamW", lr=5e-5, scheduler_type="cosine")
    optimizer = optimizer_factory.optimizer
    scheduler = optimizer_factory.scheduler

    print(f"Using optimizer: {optimizer}")
    print(f"Using scheduler: {scheduler}")

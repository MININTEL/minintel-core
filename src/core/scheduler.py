"""
Machine Intelligence Node - Learning Rate Scheduler

Implements modular learning rate scheduling with warmup strategies, 
cosine annealing, step decay, and ReduceLROnPlateau.

Author: Machine Intelligence Node Development Team
"""

import torch
import torch.optim as optim

class LRScheduler:
    """
    A modular learning rate scheduler supporting multiple strategies for adaptive training.
    """
    def __init__(self, optimizer, scheduler_type="cosine", warmup_steps=1000, total_steps=50000, step_size=10, gamma=0.1, factor=0.5, patience=5):
        """
        Initializes the learning rate scheduler.

        Args:
            optimizer (torch.optim.Optimizer): The optimizer whose learning rate will be scheduled.
            scheduler_type (str): Type of scheduler ('cosine', 'step', 'plateau').
            warmup_steps (int): Number of warmup steps before decay.
            total_steps (int): Total training steps for cosine scheduling.
            step_size (int): Step size for step-based learning rate decay.
            gamma (float): Multiplicative factor for step decay.
            factor (float): Factor by which to reduce LR on plateau.
            patience (int): Number of epochs with no improvement before reducing LR.
        """
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.current_step = 0
        self.scheduler_type = scheduler_type

        if scheduler_type == "cosine":
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps)
        elif scheduler_type == "step":
            self.scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
        elif scheduler_type == "plateau":
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=factor, patience=patience)
        else:
            raise ValueError(f"Unsupported scheduler type: {scheduler_type}")

    def step(self, val_loss=None):
        """
        Performs a scheduler step, adjusting the learning rate if needed.
        """
        if self.scheduler_type == "plateau":
            self.scheduler.step(val_loss)
        else:
            self.scheduler.step()
        self.current_step += 1

    def get_lr(self):
        """
        Returns the current learning rate.
        """
        return self.optimizer.param_groups[0]['lr']

    def warmup_step(self):
        """
        Gradually increases the learning rate during the warmup phase.
        """
        if self.current_step < self.warmup_steps:
            lr = (self.current_step / self.warmup_steps) * self.optimizer.defaults["lr"]
            for param_group in self.optimizer.param_groups:
                param_group["lr"] = lr

# Example usage
if __name__ == "__main__":
    from optimizer import OptimizerFactory
    from model import TransformerModel

    vocab_size = 30000
    model = TransformerModel(vocab_size=vocab_size)

    optimizer_factory = OptimizerFactory(model, optimizer_type="AdamW", lr=5e-5)
    optimizer = optimizer_factory.optimizer

    scheduler = LRScheduler(optimizer, scheduler_type="cosine", warmup_steps=500, total_steps=10000)

    print(f"Initial learning rate: {scheduler.get_lr()}")
    for step in range(10):
        scheduler.warmup_step()
        scheduler.step()
        print(f"Step {step + 1}: Learning rate = {scheduler.get_lr()}")

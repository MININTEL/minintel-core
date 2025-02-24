"""
Machine Intelligence Node - Hyperparameter Tuning Module

Handles structured and adaptive hyperparameter tuning for AI model optimization.

Author: Machine Intelligence Node Development Team
"""

import random
import itertools
import torch
from typing import Dict, Any, List, Callable
from concurrent.futures import ThreadPoolExecutor
from src.training.trainer import Trainer
from src.utils.logger import Logger

class HyperparameterTuner:
    """
    A flexible hyperparameter tuning system supporting Grid Search, Random Search,
    and future Bayesian Optimization.
    """
    def __init__(self, model_fn: Callable, train_dataloader, search_space: Dict[str, List[Any]], method: str = "random", num_trials: int = 10, device: str = "cuda"):
        """
        Initializes the hyperparameter tuning module.

        Args:
            model_fn (Callable): Function to initialize a model.
            train_dataloader: Training dataset loader.
            search_space (Dict[str, List[Any]]): Dictionary defining hyperparameter ranges.
            method (str): Search method ('grid' or 'random').
            num_trials (int): Number of trials (only for random search).
            device (str): Training device ('cuda' or 'cpu').
        """
        self.model_fn = model_fn
        self.train_dataloader = train_dataloader
        self.search_space = search_space
        self.method = method.lower()
        self.num_trials = num_trials
        self.device = device if torch.cuda.is_available() else "cpu"
        self.logger = Logger(log_file="logs/hyperparameter_tuning.log")

    def _grid_search(self) -> List[Dict[str, Any]]:
        """
        Generates hyperparameter combinations using Grid Search.

        Returns:
            List[Dict[str, Any]]: All possible hyperparameter combinations.
        """
        keys, values = zip(*self.search_space.items())
        return [dict(zip(keys, v)) for v in itertools.product(*values)]

    def _random_search(self) -> List[Dict[str, Any]]:
        """
        Generates random hyperparameter combinations.

        Returns:
            List[Dict[str, Any]]: Randomly sampled hyperparameter combinations.
        """
        return [{k: random.choice(v) for k, v in self.search_space.items()} for _ in range(self.num_trials)]

    def run_tuning(self, epochs: int = 5, batch_size: int = 32):
        """
        Executes hyperparameter tuning across different configurations.

        Args:
            epochs (int): Number of epochs for training.
            batch_size (int): Batch size for training.
        """
        self.logger.info(f"Starting hyperparameter tuning using {self.method} search.")

        if self.method == "grid":
            param_combinations = self._grid_search()
        elif self.method == "random":
            param_combinations = self._random_search()
        else:
            raise ValueError("Unsupported tuning method. Use 'grid' or 'random'.")

        results = []
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = [executor.submit(self._train_model, params, epochs, batch_size) for params in param_combinations]
            for future in futures:
                results.append(future.result())

        best_config = min(results, key=lambda x: x["loss"])
        self.logger.info(f"Best hyperparameter set found: {best_config}")

    def _train_model(self, params: Dict[str, Any], epochs: int, batch_size: int) -> Dict[str, Any]:
        """
        Trains the model with a given hyperparameter configuration.

        Args:
            params (Dict[str, Any]): Hyperparameter configuration.
            epochs (int): Number of epochs.
            batch_size (int): Batch size.

        Returns:
            Dict[str, Any]: Training results, including loss and best configuration.
        """
        model = self.model_fn(**params).to(self.device)
        trainer = Trainer(model, optimizer=params.get("optimizer", "adam"), lr=params.get("lr", 0.001))
        
        trainer.train(self.train_dataloader, epochs=epochs)
        
        final_loss = trainer.logger.get_last_loss()
        self.logger.info(f"Finished training with {params} - Loss: {final_loss:.4f}")

        return {"params": params, "loss": final_loss}

# Example Usage
if __name__ == "__main__":
    class DummyModel(torch.nn.Module):
        def __init__(self, hidden_size: int = 128, dropout: float = 0.2):
            super().__init__()
            self.fc = torch.nn.Sequential(
                torch.nn.Linear(512, hidden_size),
                torch.nn.ReLU(),
                torch.nn.Dropout(dropout),
                torch.nn.Linear(hidden_size, 10),
            )

        def forward(self, x):
            return self.fc(x)

    # Simulated dataset
    from torch.utils.data import DataLoader, TensorDataset

    X = torch.randn(100, 512)
    y = torch.randint(0, 10, (100,))
    dataset = TensorDataset(X, y)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

    # Define search space
    search_space = {
        "hidden_size": [64, 128, 256],
        "dropout": [0.1, 0.2, 0.3],
        "optimizer": ["adam", "sgd"],
        "lr": [0.001, 0.01]
    }

    tuner = HyperparameterTuner(DummyModel, dataloader, search_space, method="random", num_trials=5)
    tuner.run_tuning(epochs=3)

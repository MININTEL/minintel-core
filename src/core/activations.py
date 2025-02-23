"""
Machine Intelligence Node - Advanced Activation Functions

Implements standard and advanced activation functions for improved 
gradient flow and training stability in Transformer models.

Author: Machine Intelligence Node Development Team
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class Swish(nn.Module):
    """
    Swish activation function: x * sigmoid(x).
    Swish has been shown to improve training efficiency in deep networks.
    """
    def forward(self, x):
        return x * torch.sigmoid(x)

class Mish(nn.Module):
    """
    Mish activation function: x * tanh(softplus(x)).
    Mish allows smoother gradient flow compared to ReLU and Swish.
    """
    def forward(self, x):
        return x * torch.tanh(F.softplus(x))

class ParametricReLU(nn.Module):
    """
    Parametric ReLU (PReLU) allows learned negative slopes for flexible activation.
    """
    def __init__(self, num_parameters=1, init=0.25):
        super(ParametricReLU, self).__init__()
        self.alpha = nn.Parameter(torch.full((num_parameters,), init))

    def forward(self, x):
        return torch.max(x, torch.zeros_like(x)) + self.alpha * torch.min(x, torch.zeros_like(x))

class SoftExponential(nn.Module):
    """
    Soft Exponential activation: Introduces trainable non-linearity.

    When α > 0: Similar to an exponential function.
    When α = 0: Reduces to identity function.
    When α < 0: Behaves like a logarithmic function.
    """
    def __init__(self, alpha=0.1):
        super(SoftExponential, self).__init__()
        self.alpha = nn.Parameter(torch.tensor(alpha))

    def forward(self, x):
        if self.alpha == 0:
            return x
        elif self.alpha > 0:
            return (torch.exp(self.alpha * x) - 1) / self.alpha + self.alpha
        else:
            return -torch.log(1 - self.alpha * (x + self.alpha)) / self.alpha

def activation_factory(activation_name):
    """
    Factory function to select activation dynamically.

    Args:
        activation_name (str): Name of the activation function.

    Returns:
        nn.Module: Instantiated activation function.
    """
    activations = {
        "relu": nn.ReLU(),
        "gelu": nn.GELU(),
        "swish": Swish(),
        "mish": Mish(),
        "prelu": ParametricReLU(),
        "soft_exp": SoftExponential(),
    }
    
    if activation_name.lower() in activations:
        return activations[activation_name.lower()]
    else:
        raise ValueError(f"Unsupported activation function: {activation_name}")

# Example Usage
if __name__ == "__main__":
    x = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0])

    swish = Swish()
    mish = Mish()
    prelu = ParametricReLU()
    soft_exp = SoftExponential()

    print(f"Swish: {swish(x)}")
    print(f"Mish: {mish(x)}")
    print(f"Parametric ReLU: {prelu(x)}")
    print(f"Soft Exponential: {soft_exp(x)}")

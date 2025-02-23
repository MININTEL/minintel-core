"""
Machine Intelligence Node - Inference Module

Handles real-time model inference, batch predictions, and post-processing 
for optimized AI deployment.

Author: Machine Intelligence Node Development Team
"""

from .inference_engine import InferenceEngine
from .postprocessing import PostProcessor

__all__ = [
    "InferenceEngine",
    "PostProcessor"
]

def initialize_inference():
    """
    Initializes and returns the inference components for AI model deployment.

    Returns:
        dict: Dictionary containing initialized inference components.
    """
    engine = InferenceEngine()
    post_processor = PostProcessor()

    return {
        "engine": engine,
        "post_processor": post_processor
    }

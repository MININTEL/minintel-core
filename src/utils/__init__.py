"""
Machine Intelligence Node - Utilities Module

Provides logging, configuration handling, metric tracking, 
and performance monitoring utilities for AI training and inference.

Author: Machine Intelligence Node Development Team
"""

from .config import ConfigLoader
from .logger import Logger
from .metrics import MetricTracker
from .profiler import Profiler
from .file_utils import FileHandler

__all__ = [
    "ConfigLoader",
    "Logger",
    "MetricTracker",
    "Profiler",
    "FileHandler",
]

def initialize_utilities():
    """
    Initializes essential utility components for Machine Intelligence Node.
    
    Returns:
        dict: A dictionary containing initialized utility instances.
    """
    logger = Logger()
    config = ConfigLoader()
    metrics = MetricTracker()
    profiler = Profiler()
    file_handler = FileHandler()

    return {
        "logger": logger,
        "config": config,
        "metrics": metrics,
        "profiler": profiler,
        "file_handler": file_handler,
    }

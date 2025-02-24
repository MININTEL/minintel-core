"""
Machine Intelligence Node - Test Suite Initialization

Handles structured imports and test discovery for unit, integration, and performance tests.

Author: Machine Intelligence Node Development Team
"""

import pytest

# Import all test modules
from .test_model import *
from .test_optimizer import *
from .test_data_loader import *
from .test_inference import *

__all__ = [
    "test_model",
    "test_optimizer",
    "test_data_loader",
    "test_inference"
]

# Fixture to initialize test environment (if needed)
@pytest.fixture(scope="session", autouse=True)
def setup_test_environment():
    """
    Initializes any necessary test setup before running tests.
    """
    print("\n[TEST SUITE] Initializing test environment...\n")

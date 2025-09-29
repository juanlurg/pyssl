# ABOUTME: SSL framework package initialization
# ABOUTME: Main imports for the semi-supervised learning framework

"""PySSL: Semi-Supervised Learning Framework

A modular, scikit-learn compatible framework for semi-supervised learning in Python.
"""

from .main import SelfTrainingClassifier
from . import strategies

__version__ = "0.1.0"
__author__ = "PySSL Contributors"

__all__ = [
    "SelfTrainingClassifier",
    "strategies",
]
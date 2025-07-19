"""CCTV video analysis tool for detecting humans in specified regions."""

from .detector import HumanDetector
from .cli import main

__version__ = "0.1.0"
__all__ = ["HumanDetector", "main"]

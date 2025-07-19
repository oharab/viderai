"""CCTV video analysis tool for detecting humans in specified regions."""

from .detector import HumanDetector
from .cli import main
from .region_selector import InteractiveRegionSelector, select_region_interactively

__version__ = "0.1.0"
__all__ = ["HumanDetector", "main", "InteractiveRegionSelector", "select_region_interactively"]

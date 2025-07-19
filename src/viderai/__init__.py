"""CCTV video analysis tool for detecting humans in specified regions."""

from .detector import HumanDetector
from .cli import main
from .region_selector import InteractiveRegionSelector, select_region_interactively

try:
    from ._version import __version__
except ImportError:
    # Fallback for development installations
    try:
        from importlib.metadata import version, PackageNotFoundError
        try:
            __version__ = version("viderai")
        except PackageNotFoundError:
            __version__ = "dev"
    except ImportError:
        __version__ = "dev"

__all__ = ["HumanDetector", "main", "InteractiveRegionSelector", "select_region_interactively", "__version__"]

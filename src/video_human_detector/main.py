"""Entry point script for PyInstaller packaging."""

import sys
import os

# Add the current directory to Python path to enable proper imports
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

# Add the parent directory to handle package imports
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

if __name__ == "__main__":
    try:
        # Try relative import first
        from .cli import main
    except ImportError:
        try:
            # Try direct import
            from cli import main
        except ImportError:
            # Try package import
            from video_human_detector.cli import main
    
    main()
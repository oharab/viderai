"""Entry point script for PyInstaller packaging."""

import sys
import os

# Add the current directory to Python path to enable proper imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

if __name__ == "__main__":
    from cli import main
    main()
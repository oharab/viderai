#!/usr/bin/env python3
"""
Build script for creating Windows executable using PyInstaller.

This script sets up the environment and builds a standalone Windows executable
that can run without requiring a Python installation.
"""

import subprocess
import sys
import shutil
from pathlib import Path


def run_command(cmd, check=True):
    """Run a command and print its output."""
    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, check=check, capture_output=True, text=True)
    if result.stdout:
        print(result.stdout)
    if result.stderr:
        print(result.stderr, file=sys.stderr)
    return result


def main():
    """Main build process."""
    print("Building Windows executable for video-human-detector...")
    
    # Ensure we're in the project root
    project_root = Path(__file__).parent
    print(f"Project root: {project_root}")
    
    # Check if build dependencies are installed
    try:
        import PyInstaller
        print(f"PyInstaller version: {PyInstaller.__version__}")
    except ImportError:
        print("PyInstaller not found. Installing build dependencies...")
        run_command([sys.executable, "-m", "uv", "sync", "--extra", "build"])
    
    # Clean previous builds
    dist_dir = project_root / "dist"
    build_dir = project_root / "build"
    
    if dist_dir.exists():
        print("Cleaning previous dist directory...")
        shutil.rmtree(dist_dir)
    
    if build_dir.exists():
        print("Cleaning previous build directory...")
        shutil.rmtree(build_dir)
    
    # Ensure YOLO model file exists
    model_file = project_root / "yolov8n.pt"
    if not model_file.exists():
        print("YOLO model file not found. This will be downloaded on first run.")
    
    # Build the executable
    spec_file = project_root / "video-human-detector.spec"
    
    print("Building executable with PyInstaller...")
    run_command([
        sys.executable, "-m", "PyInstaller",
        "--clean",
        str(spec_file)
    ])
    
    # Check if build was successful
    exe_path = dist_dir / "video-human-detector.exe"
    if exe_path.exists():
        print(f"\n✓ Build successful!")
        print(f"Executable created at: {exe_path}")
        print(f"File size: {exe_path.stat().st_size / (1024*1024):.1f} MB")
        
        print("\nTo test the executable:")
        print(f"  {exe_path} --help")
        
        print("\nTo distribute:")
        print(f"  Copy the entire 'dist' directory to the target Windows machine")
        print(f"  The executable is self-contained and doesn't require Python")
        
    else:
        print("✗ Build failed - executable not found")
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
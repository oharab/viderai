# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Viderai is an AI-powered video analysis tool for detecting humans in specified rectangular regions using YOLO models. The project uses uv for dependency management and follows a defensive security approach - it's designed for legitimate security analysis and surveillance applications only.

## Essential Commands

### Development Setup
```bash
# Install dependencies (including dev dependencies for testing)
uv sync --extra dev

# Run the CLI tool with manual region specification
uv run viderai VIDEO_PATH --center-x X --center-y Y --width W --height H

# Run with interactive region selection
uv run viderai VIDEO_PATH --interactive

# Run with frame capture for extended detections (NEW FEATURE)
uv run viderai VIDEO_PATH --interactive --capture-frames --min-duration 10.0 --output-dir surveillance_frames

# Run with verbose logging
uv run viderai VIDEO_PATH --center-x X --center-y Y --width W --height H --verbose

# Run in quiet mode (for scripting)
uv run viderai VIDEO_PATH --center-x X --center-y Y --width W --height H --quiet --output-format json
```

### Testing
```bash
# Run all tests
uv run pytest

# Run specific test file
uv run pytest tests/test_detector.py

# Run with coverage
uv run pytest --cov=viderai
```

### Windows Packaging

To create a standalone Windows executable that doesn't require Python installation:

```bash
# Install build dependencies
uv sync --extra build

# Create Windows executable using PyInstaller
uv run pyinstaller viderai.spec

# The executable will be created in dist/
# Copy the entire dist/ directory to Windows machines
```

**Alternative build script:**
```bash
# Use the automated build script
python build_windows.py
```

**Package Contents:**
- `dist/viderai.exe` - Standalone Windows executable (single file)

**Distribution:**
- Single file: Copy `viderai.exe` to target Windows machines
- No Python installation required on target machines
- Executable size: ~100-200MB (includes PyTorch and OpenCV dependencies)

**Fixed Issues:**
- ✅ Spec file now correctly generates `.exe` extension
- ✅ Fixed module import issues with robust fallback imports
- ✅ Supports both single-file and directory distribution modes

## Architecture

### Core Components

**DetectorBase (Abstract)** → **YOLODetector (Implementation)**
- Extensible architecture allowing for future detection models (OpenPose, etc.)
- YOLODetector uses Ultralytics YOLO models with configurable confidence thresholds
- DetectorBase.detect_humans() returns List[Detection] for any frame

**HumanDetector (Main Analysis Class)**
- Takes any DetectorBase implementation
- analyze_video() method processes entire videos frame-by-frame
- Returns List[TimeRange] indicating when humans were present in specified regions
- Supports progress callbacks and comprehensive logging

**Region System**
- Regions defined by center point (x, y) + dimensions (width, height)
- Auto-calculates bounding box coordinates (x1, y1, x2, y2)
- Detection.overlaps_with_region() determines if human detection intersects with monitored area
- **InteractiveRegionSelector**: Visual region selection using OpenCV with keyboard controls

**Interactive Region Selection**
- **Side Panel Interface**: Video frame with unobstructed view + instruction panel
- **Subtle Visual Design**: Thin, translucent region overlay (70% opacity) with corner markers
- **Movement**: Arrow keys to move region around frame
- **Resize Both**: Z/X keys to resize both width and height simultaneously
- **Width Only**: M/N keys to make region narrower/wider independently
- **Height Only**: H/J keys to make region shorter/taller independently
- **Confirm/Cancel**: Enter to confirm, Esc to cancel selection
- **Real-time Feedback**: Live region coordinates, size, and area display
- **Color-coded Instructions**: Clear visual hierarchy in side panel
- Supports starting with predefined region or default center placement

**Frame Capture System**
- Automatically saves annotated frames when humans detected for extended periods
- Configurable duration threshold before triggering frame capture
- Selects middle frame from detection range for best representation
- Saved frames include subtle region overlay with extended canvas for metadata
- Extended canvas design prevents metadata from obscuring video content
- Supports custom output directories and automatic filename generation

### Data Flow
1. Video frames → DetectorBase.detect_humans() → List[Detection]
2. Detections filtered by Region.overlaps_with_region()
3. Continuous human presence tracked across frames
4. Frame capture triggered for extended detections (if enabled)
5. Output as TimeRange objects with start/end timestamps and optional captured frame paths

### CLI vs Library Usage
- **CLI**: Uses Click framework with progress bars (tqdm) and configurable logging
- **Library**: Direct programmatic access to HumanDetector with optional progress callbacks
- Both support frame skipping for performance optimization

### Logging Strategy
- Default: Silent operation with progress bar only
- `--verbose/-v`: Detailed logging including detection events and timing
- `--quiet/-q`: Suppresses all output except results
- Uses Python logging module with separate loggers for different components

## Key Configuration Options

- **Frame Skip**: Process every Nth frame for performance (CLI: `--frame-skip`, API: `frame_skip` parameter)
- **Confidence Threshold**: YOLO detection confidence (CLI: `--confidence`, API: YOLODetector constructor)
- **YOLO Model**: Different model sizes available (nano/small/medium/large via `--model` or YOLODetector constructor)
- **Output Format**: Human-readable or JSON (CLI: `--output-format`)
- **Interactive Mode**: Visual region selection (CLI: `--interactive/-i`)
- **Frame Capture**: Save frames for extended detections (CLI: `--capture-frames`, API: `capture_frames` parameter)
- **Capture Duration**: Minimum duration before frame capture (CLI: `--min-duration`, API: `min_duration_for_capture` parameter)
- **Output Directory**: Custom directory for captured frames (CLI: `--output-dir`, API: `output_dir` parameter)

## Extension Points

To add new detection models:
1. Inherit from DetectorBase
2. Implement detect_humans(frame) → List[Detection]
3. Pass to HumanDetector constructor

The Region and TimeRange classes are designed to work with any detection backend.
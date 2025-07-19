# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a CCTV video analysis tool for detecting humans in specified rectangular regions using YOLO models. The project uses uv for dependency management and follows a defensive security approach - it's designed for legitimate security analysis and surveillance applications only.

## Essential Commands

### Development Setup
```bash
# Install dependencies (including dev dependencies for testing)
uv sync --extra dev

# Run the CLI tool with manual region specification
uv run video-human-detector VIDEO_PATH --center-x X --center-y Y --width W --height H

# Run with interactive region selection (NEW FEATURE)
uv run video-human-detector VIDEO_PATH --interactive

# Run with verbose logging
uv run video-human-detector VIDEO_PATH --center-x X --center-y Y --width W --height H --verbose

# Run in quiet mode (for scripting)
uv run video-human-detector VIDEO_PATH --center-x X --center-y Y --width W --height H --quiet --output-format json
```

### Testing
```bash
# Run all tests
uv run pytest

# Run specific test file
uv run pytest tests/test_detector.py

# Run with coverage
uv run pytest --cov=video_human_detector
```

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
- Displays first frame of video with overlay showing current region
- Controls: Arrow keys (move), Z/X keys (resize), Enter (confirm), Esc (cancel)
- Supports starting with predefined region or default center placement
- Real-time visual feedback with region coordinates and size display

### Data Flow
1. Video frames → DetectorBase.detect_humans() → List[Detection]
2. Detections filtered by Region.overlaps_with_region()
3. Continuous human presence tracked across frames
4. Output as TimeRange objects with start/end timestamps

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

## Extension Points

To add new detection models:
1. Inherit from DetectorBase
2. Implement detect_humans(frame) → List[Detection]
3. Pass to HumanDetector constructor

The Region and TimeRange classes are designed to work with any detection backend.
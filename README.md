# Viderai 🎥

**AI-Powered Video Analysis for Human Detection**

Viderai is a modern, efficient tool for detecting human presence in specified regions of video files. Designed for security professionals, researchers, and surveillance applications, it combines the power of YOLO object detection with an intuitive interface for precise region monitoring.

[![Python 3.13+](https://img.shields.io/badge/python-3.13+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![uv](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json)](https://github.com/astral-sh/uv)

## ✨ Features

- **🎯 Precise Region Monitoring** - Define rectangular regions of interest with pixel-perfect accuracy
- **🖱️ Interactive Region Selection** - Visual point-and-click interface for easy region setup
- **⚡ Real-Time Performance** - Optimized YOLO detection with configurable frame skipping
- **📸 Automated Frame Capture** - Save annotated frames when humans are detected for extended periods
- **⚙️ Flexible Configuration** - Adjustable confidence thresholds, models, and output formats
- **🌐 Cross-Platform** - Works on Windows, macOS, and Linux (source install)
- **📦 Standalone Executables** - Available for Windows and macOS
- **🔒 Security-Focused** - Designed for legitimate surveillance and security applications

## 🚀 Quick Start

### Installation

Viderai uses [uv](https://github.com/astral-sh/uv) for fast, reliable dependency management:

```bash
# Clone the repository
git clone https://github.com/yourusername/viderai.git
cd viderai

# Install dependencies
uv sync

# Run Viderai
uv run viderai --help
```

### Basic Usage

```bash
# Interactive region selection (recommended for first-time use)
uv run viderai video.mp4 --interactive

# Manual region specification
uv run viderai video.mp4 --center-x 320 --center-y 240 --width 200 --height 150

# With frame capture for extended detections
uv run viderai video.mp4 --interactive --capture-frames --min-duration 5.0
```

## 📚 Documentation

### Command Line Interface

```bash
uv run viderai [OPTIONS] VIDEO_PATH

Options:
  --center-x INTEGER       X coordinate of region center
  --center-y INTEGER       Y coordinate of region center  
  --width INTEGER          Width of region in pixels
  --height INTEGER         Height of region in pixels
  -i, --interactive        Interactive region selection mode
  --frame-skip INTEGER     Process every Nth frame (default: 1)
  --confidence FLOAT       Detection confidence threshold (default: 0.5)
  --model TEXT            YOLO model to use (default: yolov8n.pt)
  --output-format [human|json]  Output format
  --capture-frames        Save frames for extended detections
  --min-duration FLOAT    Minimum duration before capturing frames
  --output-dir TEXT       Directory for captured frames
  -v, --verbose           Enable verbose logging
  -q, --quiet             Suppress progress bar
  --help                  Show help message
```

### Programmatic Usage

```python
from viderai import HumanDetector
from viderai.detector import Region, YOLODetector

# Define region of interest
region = Region(center_x=320, center_y=240, width=200, height=150)

# Initialize detector
detector = HumanDetector(YOLODetector(confidence_threshold=0.6))

# Analyze video
time_ranges = detector.analyze_video(
    video_path="security_footage.mp4",
    region=region,
    frame_skip=5,
    capture_frames=True,
    min_duration_for_capture=3.0
)

# Process results
for tr in time_ranges:
    print(f"Human detected: {tr.start_time:.2f}s - {tr.end_time:.2f}s")
    if tr.captured_frame_path:
        print(f"Frame saved: {tr.captured_frame_path}")
```

## 🖱️ Interactive Region Selection

Viderai features an intuitive visual interface for selecting monitoring regions:

- **Visual Overlay**: See exactly what area you're monitoring
- **Keyboard Controls**: 
  - Arrow keys: Move region
  - Z/X: Resize both dimensions
  - M/N: Adjust width only  
  - H/J: Adjust height only
  - Enter: Confirm selection
  - Esc: Cancel

## 📊 Output Formats

### Human-Readable (Default)
```
Human detected in region during 2 time range(s):
  1. 45.30s - 48.80s (duration: 3.50s) [Frame saved: capture_frame_001.jpg]
  2. 127.60s - 132.10s (duration: 4.50s) [Frame saved: capture_frame_002.jpg]
```

### JSON Format
```json
[
  {
    "start_time": 45.3,
    "end_time": 48.8,
    "duration": 3.5,
    "captured_frame_path": "captures/capture_frame_001.jpg"
  }
]
```

## 🏗️ Architecture

Viderai is built with extensibility in mind:

- **DetectorBase**: Abstract interface for detection backends
- **YOLODetector**: Primary implementation using Ultralytics YOLO
- **HumanDetector**: Main analysis engine with progress tracking
- **InteractiveRegionSelector**: Visual region selection interface
- **Frame Capture System**: Automated screenshot functionality

The modular design allows for easy integration of additional detection models (MediaPipe, OpenPose, etc.) while maintaining consistent APIs.

## ⚙️ Advanced Configuration

### Performance Optimization
```bash
# Process every 5th frame for speed
uv run viderai video.mp4 --interactive --frame-skip 5

# Use lighter model for faster processing  
uv run viderai video.mp4 --interactive --model yolov8n.pt

# Higher confidence for fewer false positives
uv run viderai video.mp4 --interactive --confidence 0.7
```

### Frame Capture Settings
```bash
# Capture frames only for detections longer than 10 seconds
uv run viderai video.mp4 --interactive --capture-frames --min-duration 10.0

# Custom output directory
uv run viderai video.mp4 --interactive --capture-frames --output-dir surveillance_frames
```

## 📦 Platform Support

### Windows & macOS
Standalone executables are automatically built and available in GitHub releases:
- **Windows**: Download `viderai-windows.zip` 
- **macOS**: Download `viderai-macos.tar.gz`

Both are self-contained and require no Python installation.

### Linux
Use the source installation method:

```bash
# Install uv (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Clone and install
git clone https://github.com/oharab/viderai.git
cd viderai
uv sync
uv run viderai --help
```

**Note**: Linux standalone binaries exceed GitHub's size limits due to PyTorch dependencies.

## 🧪 Testing

```bash
# Install development dependencies
uv sync --extra dev

# Run all tests
uv run pytest

# Run with coverage
uv run pytest --cov=viderai

# Run specific test file
uv run pytest tests/test_detector.py
```

## 🤝 Contributing

We welcome contributions! Here's how to get started:

1. **Fork the repository**
2. **Create a feature branch**: `git checkout -b feature/amazing-feature`
3. **Install development dependencies**: `uv sync --extra dev`
4. **Make your changes** and add tests
5. **Run tests**: `uv run pytest`
6. **Commit your changes**: `git commit -m 'Add amazing feature'`
7. **Push to the branch**: `git push origin feature/amazing-feature`
8. **Open a Pull Request**

### Development Setup

```bash
# Clone and setup development environment
git clone https://github.com/yourusername/viderai.git
cd viderai
uv sync --extra dev

# Install pre-commit hooks (optional)
uv run pre-commit install
```

## 🔒 Security & Ethics

Viderai is designed exclusively for **legitimate security and surveillance applications**. Please ensure compliance with:

- Local privacy laws and regulations
- Organizational security policies  
- Ethical surveillance guidelines
- Data protection requirements

**This tool should only be used for authorized security monitoring and research purposes.**

## 📋 Requirements

- **Python**: 3.13+ 
- **Operating System**: Windows, macOS, Linux
- **Dependencies**: OpenCV, Ultralytics YOLO, Click, NumPy
- **Hardware**: GPU recommended for real-time processing

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [Ultralytics YOLO](https://github.com/ultralytics/ultralytics) for object detection
- [OpenCV](https://opencv.org/) for computer vision capabilities
- [uv](https://github.com/astral-sh/uv) for fast Python package management
- [Click](https://click.palletsprojects.com/) for the command-line interface

## 💬 Support

- **Issues**: [GitHub Issues](https://github.com/yourusername/viderai/issues)
- **Discussions**: [GitHub Discussions](https://github.com/yourusername/viderai/discussions)
- **Documentation**: See [CLAUDE.md](CLAUDE.md) for development details

---

**Made with ❤️ for the security and research community**
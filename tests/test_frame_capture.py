"""Tests for frame capture functionality."""

import os
import tempfile
from unittest.mock import Mock, patch
import numpy as np
import pytest

from video_human_detector.detector import HumanDetector, YOLODetector, Region, TimeRange


@pytest.fixture
def temp_output_dir():
    """Create a temporary directory for test outputs."""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield temp_dir


def test_timerange_duration_property():
    """Test TimeRange duration property calculation."""
    time_range = TimeRange(start_time=10.0, end_time=15.5)
    assert time_range.duration == 5.5
    
    time_range_with_frame = TimeRange(start_time=0.0, end_time=3.2, captured_frame_path="test.jpg")
    assert time_range_with_frame.duration == 3.2
    assert time_range_with_frame.captured_frame_path == "test.jpg"


def test_annotate_frame():
    """Test frame annotation functionality."""
    detector = HumanDetector()
    
    # Create a test frame
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    region = Region(center_x=320, center_y=240, width=100, height=100)
    
    # Annotate the frame
    annotated = detector._annotate_frame(frame, region, timestamp=10.5, duration=7.2)
    
    # Check that we got a frame back with same dimensions
    assert annotated.shape == frame.shape
    assert annotated.dtype == frame.dtype
    
    # Check that the frame was modified (not identical to original)
    assert not np.array_equal(frame, annotated)


def test_save_captured_frame(temp_output_dir):
    """Test frame saving functionality."""
    detector = HumanDetector()
    
    # Create test frame data
    frame1 = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    frame2 = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    frame3 = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    
    frame_data = [
        (frame1, 10.0, 300),
        (frame2, 12.5, 375),  # This should be selected (middle)
        (frame3, 15.0, 450)
    ]
    
    region = Region(center_x=320, center_y=240, width=100, height=100)
    
    # Save frame
    output_path = detector._save_captured_frame(
        frame_data=frame_data,
        video_path="/path/to/test_video.mp4",
        start_time=10.0,
        duration=5.0,
        output_dir=temp_output_dir,
        region=region
    )
    
    # Check that file was created
    assert output_path is not None
    assert os.path.exists(output_path)
    assert output_path.startswith(temp_output_dir)
    assert "test_video_t10.0s_d5.0s_f375.jpg" in output_path


def test_save_captured_frame_empty_data():
    """Test frame saving with empty frame data."""
    detector = HumanDetector()
    
    result = detector._save_captured_frame(
        frame_data=[],
        video_path="/path/to/test.mp4",
        start_time=10.0,
        duration=5.0,
        output_dir="test_dir",
        region=Region(100, 100, 50, 50)
    )
    
    assert result is None


@patch('cv2.VideoCapture')
def test_analyze_video_with_frame_capture(mock_video_capture, temp_output_dir):
    """Test video analysis with frame capture enabled."""
    # Mock video capture
    mock_cap = Mock()
    mock_cap.isOpened.return_value = True
    mock_cap.get.side_effect = lambda prop: {
        5: 30.0,  # CAP_PROP_FPS
        7: 900    # CAP_PROP_FRAME_COUNT (30 seconds of video)
    }.get(prop, 0)
    
    # Create test frames
    test_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    
    # Mock frame reading - simulate detection for frames 0-300 (10 seconds)
    frame_reads = [(True, test_frame.copy()) for _ in range(900)] + [(False, None)]
    mock_cap.read.side_effect = frame_reads
    mock_video_capture.return_value = mock_cap
    
    # Mock YOLO detector to always detect humans
    mock_detector = Mock()
    mock_detector.detect_humans.return_value = [
        Mock(overlaps_with_region=Mock(return_value=True))
    ]
    
    detector = HumanDetector(mock_detector)
    region = Region(center_x=320, center_y=240, width=100, height=100)
    
    # Analyze with frame capture
    time_ranges = detector.analyze_video(
        video_path="test_video.mp4",
        region=region,
        frame_skip=30,  # Process every 30th frame
        capture_frames=True,
        min_duration_for_capture=5.0,
        output_dir=temp_output_dir
    )
    
    # Should have one time range covering the whole video
    assert len(time_ranges) == 1
    time_range = time_ranges[0]
    
    # Check duration is longer than minimum
    assert time_range.duration >= 5.0
    
    # Check that frame was captured
    assert time_range.captured_frame_path is not None
    assert os.path.exists(time_range.captured_frame_path)


@patch('cv2.VideoCapture')
def test_analyze_video_short_duration_no_capture(mock_video_capture):
    """Test that short duration detections don't trigger frame capture."""
    # Mock video capture for short detection
    mock_cap = Mock()
    mock_cap.isOpened.return_value = True
    mock_cap.get.side_effect = lambda prop: {
        5: 30.0,  # CAP_PROP_FPS
        7: 150    # CAP_PROP_FRAME_COUNT (5 seconds)
    }.get(prop, 0)
    
    test_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    
    # Simulate detection for only first 3 seconds (90 frames)
    frame_reads = []
    for i in range(150):
        frame_reads.append((True, test_frame.copy()))
    frame_reads.append((False, None))
    
    mock_cap.read.side_effect = frame_reads
    mock_video_capture.return_value = mock_cap
    
    # Mock detector to detect human for first 90 frames only
    mock_detector = Mock()
    call_count = 0
    
    def mock_detect(frame):
        nonlocal call_count
        call_count += 1
        # Detect human for first 3 calls (3 seconds at 1fps sampling)
        if call_count <= 3:
            return [Mock(overlaps_with_region=Mock(return_value=True))]
        return []
    
    mock_detector.detect_humans.side_effect = mock_detect
    
    detector = HumanDetector(mock_detector)
    region = Region(center_x=320, center_y=240, width=100, height=100)
    
    # Analyze with frame capture (min duration = 5s, but detection only lasts 3s)
    time_ranges = detector.analyze_video(
        video_path="test_video.mp4",
        region=region,
        frame_skip=30,  # Sample at 1fps
        capture_frames=True,
        min_duration_for_capture=5.0,
        output_dir="test_captures"
    )
    
    # Should have one time range but no captured frame (duration < 5s)
    assert len(time_ranges) == 1
    time_range = time_ranges[0]
    assert time_range.duration < 5.0
    assert time_range.captured_frame_path is None
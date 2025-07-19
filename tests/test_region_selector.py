"""Tests for interactive region selection functionality."""

import numpy as np
import pytest
from unittest.mock import Mock, patch
from video_human_detector.region_selector import InteractiveRegionSelector
from video_human_detector.detector import Region


@pytest.fixture
def sample_frame():
    """Create a sample video frame for testing."""
    return np.zeros((480, 640, 3), dtype=np.uint8)


def test_interactive_region_selector_initialization(sample_frame):
    """Test InteractiveRegionSelector initialization."""
    # Test with default region
    selector = InteractiveRegionSelector(sample_frame)
    assert selector.region.center_x == 320  # 640 // 2
    assert selector.region.center_y == 240  # 480 // 2
    assert selector.region.width == 160     # min(200, 640 // 4)
    assert selector.region.height == 120    # min(150, 480 // 4)
    
    # Test with custom initial region
    initial_region = Region(100, 100, 80, 60)
    selector = InteractiveRegionSelector(sample_frame, initial_region)
    assert selector.region.center_x == 100
    assert selector.region.center_y == 100
    assert selector.region.width == 80
    assert selector.region.height == 60


def test_clamp_region(sample_frame):
    """Test region clamping to frame boundaries."""
    selector = InteractiveRegionSelector(sample_frame)
    
    # Test clamping region that's too large
    selector.region.width = 1000
    selector.region.height = 1000
    selector._clamp_region()
    assert selector.region.width == 640  # Frame width
    assert selector.region.height == 480  # Frame height
    
    # Test clamping region that's too small
    selector.region.width = 5
    selector.region.height = 5
    selector._clamp_region()
    assert selector.region.width == 20   # min_size
    assert selector.region.height == 20  # min_size
    
    # Test clamping position outside frame
    selector.region.center_x = -100
    selector.region.center_y = -100
    selector._clamp_region()
    assert selector.region.center_x >= selector.region.width // 2
    assert selector.region.center_y >= selector.region.height // 2


def test_draw_region(sample_frame):
    """Test region drawing functionality."""
    selector = InteractiveRegionSelector(sample_frame)
    display_frame = selector._draw_region()
    
    # Check that we got a frame back with the same dimensions
    assert display_frame.shape == sample_frame.shape
    assert display_frame.dtype == sample_frame.dtype


@patch('cv2.VideoCapture')
def test_select_region_interactively_video_error(mock_video_capture):
    """Test error handling for video file issues."""
    from video_human_detector.region_selector import select_region_interactively
    
    # Test video file can't be opened
    mock_cap = Mock()
    mock_cap.isOpened.return_value = False
    mock_video_capture.return_value = mock_cap
    
    with pytest.raises(ValueError, match="Could not open video file"):
        select_region_interactively("nonexistent.mp4")


@patch('cv2.VideoCapture')
def test_select_region_interactively_frame_error(mock_video_capture):
    """Test error handling for frame reading issues."""
    from video_human_detector.region_selector import select_region_interactively
    
    # Test video file opens but can't read first frame
    mock_cap = Mock()
    mock_cap.isOpened.return_value = True
    mock_cap.read.return_value = (False, None)
    mock_video_capture.return_value = mock_cap
    
    with pytest.raises(ValueError, match="Could not read first frame"):
        select_region_interactively("video.mp4")


def test_region_selector_movement(sample_frame):
    """Test that region movement works correctly."""
    selector = InteractiveRegionSelector(sample_frame)
    original_x = selector.region.center_x
    original_y = selector.region.center_y
    
    # Simulate moving right
    selector.region.center_x += selector.move_step
    selector._clamp_region()
    assert selector.region.center_x == original_x + selector.move_step
    
    # Simulate moving down
    selector.region.center_y += selector.move_step
    selector._clamp_region()
    assert selector.region.center_y == original_y + selector.move_step


def test_region_selector_resizing(sample_frame):
    """Test that region resizing works correctly."""
    selector = InteractiveRegionSelector(sample_frame)
    original_width = selector.region.width
    original_height = selector.region.height
    
    # Simulate making larger
    selector.region.width += selector.resize_step
    selector.region.height += selector.resize_step
    selector._clamp_region()
    assert selector.region.width == original_width + selector.resize_step
    assert selector.region.height == original_height + selector.resize_step
    
    # Simulate making smaller
    selector.region.width -= selector.resize_step * 2
    selector.region.height -= selector.resize_step * 2
    selector._clamp_region()
    assert selector.region.width == original_width - selector.resize_step
    assert selector.region.height == original_height - selector.resize_step


def test_region_selector_width_only_adjustment(sample_frame):
    """Test that width-only adjustment works correctly."""
    selector = InteractiveRegionSelector(sample_frame)
    original_width = selector.region.width
    original_height = selector.region.height
    
    # Simulate making width wider
    selector.region.width += selector.resize_step
    selector._clamp_region()
    assert selector.region.width == original_width + selector.resize_step
    assert selector.region.height == original_height  # Height unchanged
    
    # Simulate making width narrower
    selector.region.width -= selector.resize_step * 2
    selector._clamp_region()
    assert selector.region.width == original_width - selector.resize_step
    assert selector.region.height == original_height  # Height unchanged


def test_region_selector_height_only_adjustment(sample_frame):
    """Test that height-only adjustment works correctly."""
    selector = InteractiveRegionSelector(sample_frame)
    original_width = selector.region.width
    original_height = selector.region.height
    
    # Simulate making height taller
    selector.region.height += selector.resize_step
    selector._clamp_region()
    assert selector.region.width == original_width   # Width unchanged
    assert selector.region.height == original_height + selector.resize_step
    
    # Simulate making height shorter
    selector.region.height -= selector.resize_step * 2
    selector._clamp_region()
    assert selector.region.width == original_width   # Width unchanged
    assert selector.region.height == original_height - selector.resize_step


def test_region_selector_independent_dimensions(sample_frame):
    """Test that width and height can be adjusted independently."""
    selector = InteractiveRegionSelector(sample_frame)
    original_width = selector.region.width
    original_height = selector.region.height
    
    # Make width wider but height shorter
    selector.region.width += selector.resize_step * 2
    selector.region.height -= selector.resize_step
    selector._clamp_region()
    
    assert selector.region.width == original_width + (selector.resize_step * 2)
    assert selector.region.height == original_height - selector.resize_step
    
    # Make width narrower but height taller
    selector.region.width -= selector.resize_step * 3
    selector.region.height += selector.resize_step * 2
    selector._clamp_region()
    
    assert selector.region.width == original_width - selector.resize_step
    assert selector.region.height == original_height + selector.resize_step
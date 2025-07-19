"""Tests for human detection functionality."""

import pytest
import numpy as np
from video_human_detector.detector import Region, Detection, HumanDetector, YOLODetector


def test_region_properties():
    """Test Region coordinate calculations."""
    region = Region(center_x=100, center_y=50, width=80, height=60)
    
    assert region.x1 == 60  # 100 - 80//2
    assert region.y1 == 20  # 50 - 60//2
    assert region.x2 == 140  # 100 + 80//2
    assert region.y2 == 80   # 50 + 60//2


def test_detection_overlap():
    """Test detection overlap with region."""
    region = Region(center_x=100, center_y=100, width=50, height=50)
    
    # Detection inside region
    detection_inside = Detection(x1=80, y1=80, x2=120, y2=120, confidence=0.8)
    assert detection_inside.overlaps_with_region(region)
    
    # Detection outside region
    detection_outside = Detection(x1=200, y1=200, x2=220, y2=220, confidence=0.8)
    assert not detection_outside.overlaps_with_region(region)
    
    # Detection partially overlapping
    detection_partial = Detection(x1=120, y1=120, x2=140, y2=140, confidence=0.8)
    assert detection_partial.overlaps_with_region(region)


def test_human_detector_initialization():
    """Test HumanDetector initialization."""
    detector = HumanDetector()
    assert detector.detector is not None
    assert isinstance(detector.detector, YOLODetector)


def test_yolo_detector_initialization():
    """Test YOLODetector initialization."""
    detector = YOLODetector(confidence_threshold=0.7)
    assert detector.confidence_threshold == 0.7
    assert detector.person_class_id == 0
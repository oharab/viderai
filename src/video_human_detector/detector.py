"""Human detection functionality using YOLO models."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Tuple, Optional
import cv2
import numpy as np
from ultralytics import YOLO


@dataclass
class Region:
    """Represents a rectangular region of interest."""
    center_x: int
    center_y: int
    width: int
    height: int
    
    @property
    def x1(self) -> int:
        return self.center_x - self.width // 2
    
    @property
    def y1(self) -> int:
        return self.center_y - self.height // 2
    
    @property
    def x2(self) -> int:
        return self.center_x + self.width // 2
    
    @property
    def y2(self) -> int:
        return self.center_y + self.height // 2


@dataclass
class Detection:
    """Represents a human detection with bounding box and confidence."""
    x1: int
    y1: int
    x2: int
    y2: int
    confidence: float
    
    def overlaps_with_region(self, region: Region) -> bool:
        """Check if detection overlaps with specified region."""
        return not (self.x2 < region.x1 or self.x1 > region.x2 or 
                   self.y2 < region.y1 or self.y1 > region.y2)


@dataclass
class TimeRange:
    """Represents a time range when humans were detected."""
    start_time: float
    end_time: float


class DetectorBase(ABC):
    """Abstract base class for human detectors."""
    
    @abstractmethod
    def detect_humans(self, frame: np.ndarray) -> List[Detection]:
        """Detect humans in a frame and return detections."""
        pass


class YOLODetector(DetectorBase):
    """YOLO-based human detector."""
    
    def __init__(self, model_name: str = "yolov8n.pt", confidence_threshold: float = 0.5):
        """Initialize YOLO detector.
        
        Args:
            model_name: YOLO model to use (yolov8n.pt, yolov8s.pt, etc.)
            confidence_threshold: Minimum confidence for detections
        """
        self.model = YOLO(model_name)
        self.confidence_threshold = confidence_threshold
        self.person_class_id = 0  # COCO class ID for person
    
    def detect_humans(self, frame: np.ndarray) -> List[Detection]:
        """Detect humans in frame using YOLO."""
        results = self.model(frame, verbose=False)
        detections = []
        
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    # Check if detection is a person with sufficient confidence
                    if (box.cls == self.person_class_id and 
                        box.conf >= self.confidence_threshold):
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        confidence = box.conf.cpu().numpy().item()
                        detections.append(Detection(
                            int(x1), int(y1), int(x2), int(y2), confidence
                        ))
        
        return detections


class HumanDetector:
    """Main human detection class for video analysis."""
    
    def __init__(self, detector: Optional[DetectorBase] = None):
        """Initialize human detector.
        
        Args:
            detector: Detector implementation to use. Defaults to YOLODetector.
        """
        self.detector = detector or YOLODetector()
    
    def analyze_video(
        self, 
        video_path: str, 
        region: Region, 
        frame_skip: int = 1
    ) -> List[TimeRange]:
        """Analyze video for human presence in specified region.
        
        Args:
            video_path: Path to video file
            region: Region of interest to monitor
            frame_skip: Process every Nth frame (1 = every frame)
            
        Returns:
            List of time ranges when humans were detected in region
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video file: {video_path}")
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        time_ranges = []
        current_range_start = None
        frame_number = 0
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Skip frames based on frame_skip parameter
                if frame_number % frame_skip != 0:
                    frame_number += 1
                    continue
                
                current_time = frame_number / fps
                
                # Detect humans in current frame
                detections = self.detector.detect_humans(frame)
                
                # Check if any detection overlaps with region
                human_in_region = any(
                    detection.overlaps_with_region(region) 
                    for detection in detections
                )
                
                if human_in_region:
                    if current_range_start is None:
                        current_range_start = current_time
                else:
                    if current_range_start is not None:
                        time_ranges.append(TimeRange(current_range_start, current_time))
                        current_range_start = None
                
                frame_number += 1
            
            # Handle case where video ends while human is still in region
            if current_range_start is not None:
                final_time = frame_count / fps
                time_ranges.append(TimeRange(current_range_start, final_time))
        
        finally:
            cap.release()
        
        return time_ranges
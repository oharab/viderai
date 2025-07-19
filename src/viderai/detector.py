"""Human detection functionality using YOLO models."""

import logging
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Optional, Callable
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
    captured_frame_path: Optional[str] = None
    
    @property
    def duration(self) -> float:
        """Get the duration of this time range in seconds."""
        return self.end_time - self.start_time


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
        self.logger = logging.getLogger(__name__)
    
    def analyze_video(
        self, 
        video_path: str, 
        region: Region, 
        frame_skip: int = 1,
        progress_callback: Optional[Callable[[int, int], None]] = None,
        capture_frames: bool = False,
        min_duration_for_capture: float = 5.0,
        output_dir: Optional[str] = None
    ) -> List[TimeRange]:
        """Analyze video for human presence in specified region.
        
        Args:
            video_path: Path to video file
            region: Region of interest to monitor
            frame_skip: Process every Nth frame (1 = every frame)
            progress_callback: Optional callback for progress updates (current_frame, total_frames)
            capture_frames: Whether to save frames when humans detected for extended periods
            min_duration_for_capture: Minimum duration (seconds) before capturing frames
            output_dir: Directory to save captured frames (defaults to 'captures')
            
        Returns:
            List of time ranges when humans were detected in region
        """
        self.logger.info(f"Starting analysis of video: {video_path}")
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video file: {video_path}")
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        self.logger.info(f"Video properties: {frame_count} frames, {fps:.2f} fps, duration: {frame_count/fps:.2f}s")
        self.logger.info(f"Region of interest: center=({region.center_x}, {region.center_y}), size={region.width}x{region.height}")
        self.logger.info(f"Processing every {frame_skip} frame(s)")
        
        # Set up frame capture if enabled
        if capture_frames:
            if output_dir is None:
                output_dir = "captures"
            os.makedirs(output_dir, exist_ok=True)
            self.logger.info(f"Frame capture enabled: min duration {min_duration_for_capture}s, output dir: {output_dir}")
        
        time_ranges = []
        current_range_start = None
        current_range_frames = []  # Store frames for potential capture
        frame_number = 0
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Update progress
                if progress_callback:
                    progress_callback(frame_number, frame_count)
                
                # Skip frames based on frame_skip parameter
                if frame_number % frame_skip != 0:
                    frame_number += 1
                    continue
                
                current_time = frame_number / fps
                
                # Detect humans in current frame
                self.logger.debug(f"Processing frame {frame_number} at time {current_time:.2f}s")
                detections = self.detector.detect_humans(frame)
                self.logger.debug(f"Found {len(detections)} human detections")
                
                # Check if any detection overlaps with region
                human_in_region = any(
                    detection.overlaps_with_region(region) 
                    for detection in detections
                )
                
                if human_in_region:
                    if current_range_start is None:
                        current_range_start = current_time
                        current_range_frames = []  # Reset frame storage
                        self.logger.info(f"Human detected in region starting at {current_time:.2f}s")
                    
                    # Store frame for potential capture
                    if capture_frames:
                        current_range_frames.append((frame.copy(), current_time, frame_number))
                
                else:
                    if current_range_start is not None:
                        duration = current_time - current_range_start
                        captured_frame_path = None
                        
                        # Capture frame if duration exceeds threshold
                        if capture_frames and duration >= min_duration_for_capture and current_range_frames:
                            captured_frame_path = self._save_captured_frame(
                                current_range_frames, video_path, current_range_start, 
                                duration, output_dir, region
                            )
                        
                        time_range = TimeRange(current_range_start, current_time, captured_frame_path)
                        time_ranges.append(time_range)
                        self.logger.info(f"Human left region at {current_time:.2f}s (duration: {duration:.2f}s)")
                        if captured_frame_path:
                            self.logger.info(f"Captured frame saved: {captured_frame_path}")
                        
                        current_range_start = None
                        current_range_frames = []
                
                frame_number += 1
            
            # Handle case where video ends while human is still in region
            if current_range_start is not None:
                final_time = frame_count / fps
                duration = final_time - current_range_start
                captured_frame_path = None
                
                # Capture frame if duration exceeds threshold
                if capture_frames and duration >= min_duration_for_capture and current_range_frames:
                    captured_frame_path = self._save_captured_frame(
                        current_range_frames, video_path, current_range_start, 
                        duration, output_dir, region
                    )
                
                time_range = TimeRange(current_range_start, final_time, captured_frame_path)
                time_ranges.append(time_range)
                self.logger.info(f"Video ended with human still in region (final duration: {duration:.2f}s)")
                if captured_frame_path:
                    self.logger.info(f"Captured frame saved: {captured_frame_path}")
            
            self.logger.info(f"Analysis complete. Found {len(time_ranges)} time range(s) with human presence")
        
        finally:
            cap.release()
        
        return time_ranges
    
    def _save_captured_frame(
        self,
        frame_data: List[Tuple[np.ndarray, float, int]],
        video_path: str,
        start_time: float,
        duration: float,
        output_dir: str,
        region: Region
    ) -> str:
        """Save a representative frame from a detection range.
        
        Args:
            frame_data: List of (frame, timestamp, frame_number) tuples
            video_path: Original video path for filename generation
            start_time: Start time of detection range
            duration: Duration of detection range
            output_dir: Output directory for saved frames
            region: Region of interest for annotation
            
        Returns:
            Path to saved frame file
        """
        if not frame_data:
            return None
        
        # Select frame from middle of the range for best representation
        middle_idx = len(frame_data) // 2
        frame, timestamp, frame_number = frame_data[middle_idx]
        
        # Create filename with video name, timestamp, and duration
        video_name = Path(video_path).stem
        filename = f"{video_name}_t{start_time:.1f}s_d{duration:.1f}s_f{frame_number}.jpg"
        output_path = os.path.join(output_dir, filename)
        
        # Draw region overlay on the frame
        annotated_frame = self._annotate_frame(frame, region, timestamp, duration)
        
        # Save the frame
        cv2.imwrite(output_path, annotated_frame)
        
        return output_path
    
    def _annotate_frame(
        self,
        frame: np.ndarray,
        region: Region,
        timestamp: float,
        duration: float
    ) -> np.ndarray:
        """Annotate frame with region overlay and detection info.
        
        Args:
            frame: Original frame
            region: Region of interest
            timestamp: Time when frame was captured
            duration: Duration of detection range
            
        Returns:
            Annotated frame with extended canvas for metadata
        """
        # Calculate metadata area height
        metadata_height = 80  # Space for two lines of text plus padding
        
        # Create extended canvas
        original_height, original_width = frame.shape[:2]
        extended_height = original_height + metadata_height
        extended_frame = np.zeros((extended_height, original_width, 3), dtype=np.uint8)
        
        # Copy original frame to top of extended canvas
        extended_frame[:original_height, :] = frame
        
        # Fill metadata area with dark background
        extended_frame[original_height:, :] = (40, 40, 40)  # Dark gray
        
        # Create a translucent overlay for the region on the original frame area
        overlay = extended_frame[:original_height, :].copy()
        
        # Draw region rectangle with thinner line
        cv2.rectangle(
            overlay,
            (region.x1, region.y1),
            (region.x2, region.y2),
            (0, 255, 0),  # Green
            1  # Thinner line
        )
        
        # Draw smaller, more subtle center point
        cv2.circle(
            overlay,
            (region.center_x, region.center_y),
            2,  # Smaller radius
            (0, 255, 0),  # Green
            -1
        )
        
        # Add smaller corner markers
        corner_size = 4
        corners = [
            (region.x1, region.y1),  # Top-left
            (region.x2, region.y1),  # Top-right
            (region.x1, region.y2),  # Bottom-left
            (region.x2, region.y2),  # Bottom-right
        ]
        
        for corner in corners:
            cv2.rectangle(
                overlay,
                (corner[0] - corner_size//2, corner[1] - corner_size//2),
                (corner[0] + corner_size//2, corner[1] + corner_size//2),
                (0, 255, 0),  # Green
                -1
            )
        
        # Blend overlay with original frame area for translucency
        alpha = 0.7  # Translucency factor
        extended_frame[:original_height, :] = cv2.addWeighted(
            overlay, alpha, extended_frame[:original_height, :], 1 - alpha, 0
        )
        
        # Add metadata text in the bottom area
        metadata_y_start = original_height + 25
        
        # Add timestamp and duration text
        info_text = f"Time: {timestamp:.1f}s | Duration: {duration:.1f}s"
        cv2.putText(
            extended_frame,
            info_text,
            (15, metadata_y_start),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),  # White text
            2
        )
        
        # Add region info on second line
        region_text = f"Region: ({region.center_x}, {region.center_y}) {region.width}x{region.height}"
        cv2.putText(
            extended_frame,
            region_text,
            (15, metadata_y_start + 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (200, 200, 200),  # Light gray text
            1
        )
        
        return extended_frame
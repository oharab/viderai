"""Interactive region selection using OpenCV."""

import cv2
import numpy as np
from typing import Optional, Tuple
from .detector import Region


class InteractiveRegionSelector:
    """Interactive region selector for video frames."""
    
    def __init__(self, frame: np.ndarray, initial_region: Optional[Region] = None):
        """Initialize the region selector.
        
        Args:
            frame: Video frame to display
            initial_region: Starting region (defaults to center of frame)
        """
        self.frame = frame.copy()
        self.original_frame = frame.copy()
        self.height, self.width = frame.shape[:2]
        
        # Initialize region (center of frame if not provided)
        if initial_region:
            self.region = initial_region
        else:
            self.region = Region(
                center_x=self.width // 2,
                center_y=self.height // 2,
                width=min(200, self.width // 4),
                height=min(150, self.height // 4)
            )
        
        # Movement and resize step sizes
        self.move_step = 10
        self.resize_step = 10
        self.min_size = 20
        
        # Colors for visualization
        self.region_color = (0, 255, 0)  # Green
        self.text_color = (255, 255, 255)  # White
        self.text_bg_color = (0, 0, 0)  # Black
        
    def _clamp_region(self) -> None:
        """Ensure region stays within frame boundaries."""
        # Clamp dimensions
        self.region.width = max(self.min_size, min(self.region.width, self.width))
        self.region.height = max(self.min_size, min(self.region.height, self.height))
        
        # Clamp position to keep region within frame
        half_width = self.region.width // 2
        half_height = self.region.height // 2
        
        self.region.center_x = max(half_width, min(self.region.center_x, self.width - half_width))
        self.region.center_y = max(half_height, min(self.region.center_y, self.height - half_height))
    
    def _draw_region(self) -> np.ndarray:
        """Draw the region overlay on the frame."""
        # Start with original frame
        display_frame = self.original_frame.copy()
        
        # Draw rectangle
        cv2.rectangle(
            display_frame,
            (self.region.x1, self.region.y1),
            (self.region.x2, self.region.y2),
            self.region_color,
            2
        )
        
        # Draw center point
        cv2.circle(
            display_frame,
            (self.region.center_x, self.region.center_y),
            4,
            self.region_color,
            -1
        )
        
        # Draw instructions
        instructions = [
            "Arrow Keys: Move region",
            "Z/X: Resize smaller/larger", 
            "Enter: Confirm",
            "Esc: Cancel",
            f"Region: ({self.region.center_x}, {self.region.center_y}) {self.region.width}x{self.region.height}"
        ]
        
        y_offset = 30
        for i, instruction in enumerate(instructions):
            # Add text background
            text_size = cv2.getTextSize(instruction, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)[0]
            cv2.rectangle(
                display_frame,
                (10, y_offset + i * 25 - 20),
                (20 + text_size[0], y_offset + i * 25 + 5),
                self.text_bg_color,
                -1
            )
            # Add text
            cv2.putText(
                display_frame,
                instruction,
                (15, y_offset + i * 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                self.text_color,
                1
            )
        
        return display_frame
    
    def select_region(self, window_name: str = "Select Region") -> Optional[Region]:
        """Interactive region selection.
        
        Args:
            window_name: Name of the OpenCV window
            
        Returns:
            Selected Region or None if cancelled
        """
        cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)
        
        while True:
            # Draw current state
            display_frame = self._draw_region()
            cv2.imshow(window_name, display_frame)
            
            # Handle keyboard input
            key = cv2.waitKey(30) & 0xFF
            
            if key == 27:  # Esc - cancel
                cv2.destroyWindow(window_name)
                return None
                
            elif key == 13 or key == 10:  # Enter - confirm
                cv2.destroyWindow(window_name)
                return self.region
                
            elif key == ord('q'):  # Q - quit
                cv2.destroyWindow(window_name)
                return None
                
            # Arrow keys for movement
            elif key == 0 or key == 82:  # Up arrow
                self.region.center_y -= self.move_step
                self._clamp_region()
                
            elif key == 1 or key == 84:  # Down arrow  
                self.region.center_y += self.move_step
                self._clamp_region()
                
            elif key == 2 or key == 81:  # Left arrow
                self.region.center_x -= self.move_step
                self._clamp_region()
                
            elif key == 3 or key == 83:  # Right arrow
                self.region.center_x += self.move_step
                self._clamp_region()
                
            # Size adjustment
            elif key == ord('z') or key == ord('Z'):  # Smaller
                self.region.width = max(self.min_size, self.region.width - self.resize_step)
                self.region.height = max(self.min_size, self.region.height - self.resize_step)
                self._clamp_region()
                
            elif key == ord('x') or key == ord('X'):  # Larger
                self.region.width += self.resize_step
                self.region.height += self.resize_step
                self._clamp_region()


def select_region_interactively(video_path: str, initial_region: Optional[Region] = None) -> Optional[Region]:
    """Select a region interactively from the first frame of a video.
    
    Args:
        video_path: Path to video file
        initial_region: Starting region (optional)
        
    Returns:
        Selected Region or None if cancelled/error
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video file: {video_path}")
    
    try:
        # Read first frame
        ret, frame = cap.read()
        if not ret:
            raise ValueError("Could not read first frame from video")
        
        # Create selector and run
        selector = InteractiveRegionSelector(frame, initial_region)
        return selector.select_region()
        
    finally:
        cap.release()
        cv2.destroyAllWindows()
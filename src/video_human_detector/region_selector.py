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
    
    def _create_instruction_panel(self) -> np.ndarray:
        """Create a side panel with instructions and region info."""
        panel_width = 300
        panel_height = self.height
        
        # Create dark panel
        panel = np.zeros((panel_height, panel_width, 3), dtype=np.uint8)
        panel[:] = (40, 40, 40)  # Dark gray background
        
        # Instructions and info
        instructions = [
            "INTERACTIVE REGION SELECTOR",
            "",
            "MOVEMENT:",
            "↑↓←→  Move region",
            "",
            "RESIZE BOTH:",
            "Z     Both smaller",
            "X     Both larger", 
            "",
            "RESIZE WIDTH ONLY:",
            "M     Narrower",
            "N     Wider",
            "",
            "RESIZE HEIGHT ONLY:",
            "H     Shorter",
            "J     Taller",
            "",
            "ACTIONS:",
            "Enter Confirm selection",
            "Esc   Cancel & exit",
            "Q     Quit",
            "",
            "CURRENT REGION:",
            f"Center: ({self.region.center_x}, {self.region.center_y})",
            f"Size: {self.region.width} × {self.region.height}",
            f"Area: {self.region.width * self.region.height} pixels"
        ]
        
        y_pos = 25
        for instruction in instructions:
            if instruction == "":
                y_pos += 15
                continue
                
            # Different colors for different sections
            if instruction.startswith("INTERACTIVE") or instruction.endswith(":"):
                color = (100, 255, 100)  # Bright green for headers
                font_scale = 0.5 if instruction.startswith("INTERACTIVE") else 0.45
                thickness = 2 if instruction.startswith("INTERACTIVE") else 1
            elif instruction.startswith(("↑", "Z", "X", "M", "N", "H", "J", "Enter", "Esc", "Q")):
                color = (255, 255, 100)  # Yellow for controls
                font_scale = 0.4
                thickness = 1
            else:
                color = (200, 200, 200)  # Light gray for descriptions
                font_scale = 0.4
                thickness = 1
            
            cv2.putText(
                panel,
                instruction,
                (10, y_pos),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                color,
                thickness
            )
            y_pos += 20
        
        return panel
    
    def _draw_region(self) -> np.ndarray:
        """Draw the region overlay on the frame with side panel."""
        # Start with original frame
        video_frame = self.original_frame.copy()
        
        # Draw rectangle
        cv2.rectangle(
            video_frame,
            (self.region.x1, self.region.y1),
            (self.region.x2, self.region.y2),
            self.region_color,
            2
        )
        
        # Draw center point
        cv2.circle(
            video_frame,
            (self.region.center_x, self.region.center_y),
            4,
            self.region_color,
            -1
        )
        
        # Add corner markers for better visibility
        corner_size = 8
        corners = [
            (self.region.x1, self.region.y1),  # Top-left
            (self.region.x2, self.region.y1),  # Top-right
            (self.region.x1, self.region.y2),  # Bottom-left
            (self.region.x2, self.region.y2),  # Bottom-right
        ]
        
        for corner in corners:
            cv2.rectangle(
                video_frame,
                (corner[0] - corner_size//2, corner[1] - corner_size//2),
                (corner[0] + corner_size//2, corner[1] + corner_size//2),
                self.region_color,
                -1
            )
        
        # Create instruction panel
        instruction_panel = self._create_instruction_panel()
        
        # Combine video frame and instruction panel horizontally
        combined_frame = np.hstack((video_frame, instruction_panel))
        
        return combined_frame
    
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
                
            # Size adjustment - both dimensions
            elif key == ord('z') or key == ord('Z'):  # Both smaller
                self.region.width = max(self.min_size, self.region.width - self.resize_step)
                self.region.height = max(self.min_size, self.region.height - self.resize_step)
                self._clamp_region()
                
            elif key == ord('x') or key == ord('X'):  # Both larger
                self.region.width += self.resize_step
                self.region.height += self.resize_step
                self._clamp_region()
                
            # Width adjustment only
            elif key == ord('m') or key == ord('M'):  # Width narrower
                self.region.width = max(self.min_size, self.region.width - self.resize_step)
                self._clamp_region()
                
            elif key == ord('n') or key == ord('N'):  # Width wider
                self.region.width += self.resize_step
                self._clamp_region()
                
            # Height adjustment only
            elif key == ord('h') or key == ord('H'):  # Height shorter
                self.region.height = max(self.min_size, self.region.height - self.resize_step)
                self._clamp_region()
                
            elif key == ord('j') or key == ord('J'):  # Height taller
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
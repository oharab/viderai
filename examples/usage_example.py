"""Example usage of the video human detector."""

from video_human_detector import HumanDetector
from video_human_detector.detector import Region, YOLODetector


def main():
    """Example of how to use the HumanDetector programmatically."""
    
    # Define region of interest (center point + dimensions)
    region = Region(
        center_x=320,  # X coordinate of center
        center_y=240,  # Y coordinate of center
        width=200,     # Width in pixels
        height=150     # Height in pixels
    )
    
    # Initialize detector with custom settings
    yolo_detector = YOLODetector(
        model_name="yolov8n.pt",  # Use nano model for speed
        confidence_threshold=0.6   # Higher confidence threshold
    )
    detector = HumanDetector(yolo_detector)
    
    # Analyze video (replace with actual video path)
    video_path = "path/to/your/video.mp4"
    
    try:
        # Progress callback for custom progress tracking
        def progress_callback(current_frame: int, total_frames: int) -> None:
            progress = (current_frame / total_frames) * 100
            print(f"Progress: {progress:.1f}% ({current_frame}/{total_frames})")
        
        time_ranges = detector.analyze_video(
            video_path=video_path,
            region=region,
            frame_skip=5,  # Process every 5th frame for speed
            progress_callback=progress_callback  # Custom progress tracking
        )
        
        if time_ranges:
            print(f"Human detected in region during {len(time_ranges)} time range(s):")
            for i, tr in enumerate(time_ranges, 1):
                duration = tr.end_time - tr.start_time
                print(f"  {i}. {tr.start_time:.2f}s - {tr.end_time:.2f}s (duration: {duration:.2f}s)")
        else:
            print("No humans detected in the specified region.")
            
    except ValueError as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()
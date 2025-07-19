"""Command-line interface for video human detection."""

import click
from pathlib import Path
from typing import Optional

from .detector import HumanDetector, YOLODetector, Region


@click.command()
@click.argument('video_path', type=click.Path(exists=True, path_type=Path))
@click.option('--center-x', type=int, required=True, help='X coordinate of region center')
@click.option('--center-y', type=int, required=True, help='Y coordinate of region center')
@click.option('--width', type=int, required=True, help='Width of region in pixels')
@click.option('--height', type=int, required=True, help='Height of region in pixels')
@click.option('--frame-skip', type=int, default=1, help='Process every Nth frame (default: 1)')
@click.option('--confidence', type=float, default=0.5, help='Detection confidence threshold (default: 0.5)')
@click.option('--model', type=str, default='yolov8n.pt', help='YOLO model to use (default: yolov8n.pt)')
@click.option('--output-format', type=click.Choice(['human', 'json']), default='human', help='Output format')
def main(
    video_path: Path,
    center_x: int,
    center_y: int,
    width: int,
    height: int,
    frame_skip: int,
    confidence: float,
    model: str,
    output_format: str
) -> None:
    """Detect humans in a specified region of a video file.
    
    VIDEO_PATH: Path to the video file to analyze
    """
    try:
        # Create region of interest
        region = Region(center_x, center_y, width, height)
        
        # Initialize detector
        yolo_detector = YOLODetector(model, confidence)
        detector = HumanDetector(yolo_detector)
        
        # Analyze video
        click.echo(f"Analyzing video: {video_path}")
        click.echo(f"Region: center=({center_x}, {center_y}), size={width}x{height}")
        click.echo(f"Frame skip: {frame_skip}, Confidence: {confidence}")
        
        time_ranges = detector.analyze_video(str(video_path), region, frame_skip)
        
        # Output results
        if output_format == 'json':
            import json
            output = {
                'video_path': str(video_path),
                'region': {
                    'center_x': center_x,
                    'center_y': center_y,
                    'width': width,
                    'height': height
                },
                'time_ranges': [
                    {'start': tr.start_time, 'end': tr.end_time}
                    for tr in time_ranges
                ]
            }
            click.echo(json.dumps(output, indent=2))
        else:
            if time_ranges:
                click.echo(f"\nHuman detected in region during {len(time_ranges)} time range(s):")
                for i, tr in enumerate(time_ranges, 1):
                    duration = tr.end_time - tr.start_time
                    click.echo(f"  {i}. {tr.start_time:.2f}s - {tr.end_time:.2f}s (duration: {duration:.2f}s)")
            else:
                click.echo("\nNo humans detected in the specified region.")
    
    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        raise click.Abort()


if __name__ == '__main__':
    main()
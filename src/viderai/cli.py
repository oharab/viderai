"""Command-line interface for video human detection."""

import logging
import sys
from typing import Optional

import click
from pathlib import Path
from tqdm import tqdm

from .detector import HumanDetector, YOLODetector, Region
from .region_selector import select_region_interactively

# Get version directly to avoid circular import
try:
    from ._version import __version__
except ImportError:
    try:
        from importlib.metadata import version, PackageNotFoundError
        try:
            __version__ = version("viderai")
        except PackageNotFoundError:
            __version__ = "dev"
    except ImportError:
        __version__ = "dev"


@click.command()
@click.argument('video_path', type=click.Path(exists=True, path_type=Path))
@click.version_option(version=__version__, prog_name="viderai")
@click.option('--center-x', type=int, help='X coordinate of region center')
@click.option('--center-y', type=int, help='Y coordinate of region center')
@click.option('--width', type=int, help='Width of region in pixels')
@click.option('--height', type=int, help='Height of region in pixels')
@click.option('--interactive', '-i', is_flag=True, help='Interactive region selection mode')
@click.option('--frame-skip', type=int, default=1, help='Process every Nth frame (default: 1)')
@click.option('--confidence', type=float, default=0.5, help='Detection confidence threshold (default: 0.5)')
@click.option('--model', type=str, default='yolov8n.pt', help='YOLO model to use (default: yolov8n.pt)')
@click.option('--output-format', type=click.Choice(['human', 'json']), default='human', help='Output format')
@click.option('--capture-frames', is_flag=True, help='Save frames when humans detected for extended periods')
@click.option('--min-duration', type=float, default=5.0, help='Minimum duration (seconds) before capturing frames (default: 5.0)')
@click.option('--output-dir', type=str, help='Directory to save captured frames (default: captures)')
@click.option('--verbose', '-v', is_flag=True, help='Enable verbose logging')
@click.option('--quiet', '-q', is_flag=True, help='Suppress progress bar (useful for scripting)')
def main(
    video_path: Path,
    center_x: Optional[int],
    center_y: Optional[int],
    width: Optional[int],
    height: Optional[int],
    interactive: bool,
    frame_skip: int,
    confidence: float,
    model: str,
    output_format: str,
    capture_frames: bool,
    min_duration: float,
    output_dir: Optional[str],
    verbose: bool,
    quiet: bool
) -> None:
    """Detect humans in a specified region of a video file.
    
    VIDEO_PATH: Path to the video file to analyze
    
    You can specify the region either:
    1. Using --center-x, --center-y, --width, --height options
    2. Using --interactive/-i for visual region selection
    """
    try:
        # Configure logging
        setup_logging(verbose)
        
        # Determine region selection method
        if interactive:
            if not quiet:
                click.echo("Starting interactive region selection...")
                click.echo("Controls: Arrow keys (move), Z/X (resize both), M/N (width), H/J (height), Enter (confirm), Esc (cancel)")
            
            # Use interactive selection
            initial_region = None
            if all(param is not None for param in [center_x, center_y, width, height]):
                initial_region = Region(center_x, center_y, width, height)
            
            region = select_region_interactively(str(video_path), initial_region)
            if region is None:
                click.echo("Region selection cancelled.")
                return
                
        else:
            # Validate manual region parameters
            if any(param is None for param in [center_x, center_y, width, height]):
                click.echo("Error: Must specify either --interactive or all of --center-x, --center-y, --width, --height", err=True)
                raise click.Abort()
            
            # Create region from manual parameters
            region = Region(center_x, center_y, width, height)
        
        # Initialize detector
        yolo_detector = YOLODetector(model, confidence)
        detector = HumanDetector(yolo_detector)
        
        # Show initial info (unless quiet and json output)
        if not (quiet and output_format == 'json'):
            click.echo(f"Analyzing video: {video_path}")
            click.echo(f"Region: center=({region.center_x}, {region.center_y}), size={region.width}x{region.height}")
            click.echo(f"Frame skip: {frame_skip}, Confidence: {confidence}")
            if capture_frames:
                output_location = output_dir or "captures"
                click.echo(f"Frame capture enabled: min duration {min_duration}s, output: {output_location}")
        
        # Set up progress tracking
        pbar = None
        
        def progress_callback(current_frame: int, total_frames: int) -> None:
            nonlocal pbar
            if not quiet:
                if pbar is None:
                    pbar = tqdm(total=total_frames, desc="Processing frames", unit="frame")
                pbar.n = current_frame
                pbar.refresh()
        
        # Analyze video with progress tracking
        time_ranges = detector.analyze_video(
            str(video_path), 
            region, 
            frame_skip,
            progress_callback if not quiet else None,
            capture_frames,
            min_duration,
            output_dir
        )
        
        # Close progress bar
        if pbar:
            pbar.close()
        
        # Output results
        if output_format == 'json':
            import json
            output = {
                'video_path': str(video_path),
                'region': {
                    'center_x': region.center_x,
                    'center_y': region.center_y,
                    'width': region.width,
                    'height': region.height
                },
                'time_ranges': [
                    {
                        'start': tr.start_time, 
                        'end': tr.end_time,
                        'duration': tr.duration,
                        'captured_frame': tr.captured_frame_path
                    }
                    for tr in time_ranges
                ]
            }
            click.echo(json.dumps(output, indent=2))
        else:
            if time_ranges:
                click.echo(f"\nHuman detected in region during {len(time_ranges)} time range(s):")
                for i, tr in enumerate(time_ranges, 1):
                    duration = tr.duration
                    output_line = f"  {i}. {tr.start_time:.2f}s - {tr.end_time:.2f}s (duration: {duration:.2f}s)"
                    if tr.captured_frame_path:
                        output_line += f" [Frame saved: {tr.captured_frame_path}]"
                    click.echo(output_line)
            else:
                click.echo("\nNo humans detected in the specified region.")
    
    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        raise click.Abort()


def setup_logging(verbose: bool = False) -> None:
    """Configure logging based on verbosity level."""
    level = logging.DEBUG if verbose else logging.WARNING
    
    # Configure root logger to suppress most third-party logs
    logging.basicConfig(
        level=logging.WARNING,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler(sys.stderr)]
    )
    
    # Set our package logger to the desired level
    logger = logging.getLogger('video_human_detector')
    logger.setLevel(level)
    
    # Suppress ultralytics verbose output unless explicitly requested
    if not verbose:
        logging.getLogger('ultralytics').setLevel(logging.ERROR)


if __name__ == '__main__':
    main()
#!/usr/bin/env python3
"""
Video Frame Extraction Script for DeepFake Detection Dataset
===========================================================

This script extracts frames from video files and organizes them into a structure
suitable for training image classification models. It supports various video formats
and provides options for frame sampling, resizing, and quality control.

Usage:
    python extract_frames.py --input_dir /path/to/videos --output_dir /path/to/frames
    
Features:
- Supports multiple video formats (mp4, avi, mov, mkv, etc.)
- Configurable frame sampling rate
- Automatic face detection and cropping (optional)
- Maintains directory structure for train/val/test splits
- Progress tracking and logging
- Resume capability for interrupted extractions
"""

import os
import cv2
import argparse
import json
from pathlib import Path
from tqdm import tqdm
import logging
from typing import Dict, List, Optional, Tuple
import hashlib
import time

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('frame_extraction.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class FrameExtractor:
    """
    A comprehensive frame extractor for video datasets.
    """
    
    def __init__(self, 
                 input_dir: str,
                 output_dir: str,
                 frame_interval: int = 5,
                 target_size: Tuple[int, int] = (224, 224),
                 max_frames_per_video: Optional[int] = None,
                 face_detection: bool = False,
                 video_extensions: List[str] = None):
        """
        Initialize the frame extractor.
        
        Args:
            input_dir: Root directory containing video files
            output_dir: Directory to save extracted frames
            frame_interval: Extract every Nth frame (1 = all frames)
            target_size: Resize frames to this size (width, height)
            max_frames_per_video: Maximum frames to extract per video (None = all)
            face_detection: Whether to detect and crop faces
            video_extensions: List of video file extensions to process
        """
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.frame_interval = frame_interval
        self.target_size = target_size
        self.max_frames_per_video = max_frames_per_video
        self.face_detection = face_detection
        
        if video_extensions is None:
            self.video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv', '.m4v']
        else:
            self.video_extensions = video_extensions
            
        # Initialize face detector if needed
        self.face_cascade = None
        if self.face_detection:
            try:
                self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
                logger.info("Face detection enabled")
            except Exception as e:
                logger.warning(f"Could not load face detector: {e}. Proceeding without face detection.")
                self.face_detection = False
                
        # Progress tracking
        self.progress_file = self.output_dir / 'extraction_progress.json'
        self.processed_videos = self.load_progress()
        
    def load_progress(self) -> Dict[str, bool]:
        """Load previously processed videos to enable resume capability."""
        if self.progress_file.exists():
            try:
                with open(self.progress_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Could not load progress file: {e}")
        return {}
    
    def save_progress(self):
        """Save current progress."""
        try:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            with open(self.progress_file, 'w') as f:
                json.dump(self.processed_videos, f, indent=2)
        except Exception as e:
            logger.error(f"Could not save progress: {e}")
    
    def get_video_hash(self, video_path: Path) -> str:
        """Generate a hash for the video file to track processing."""
        stat = video_path.stat()
        return hashlib.md5(f"{video_path.name}_{stat.st_size}_{stat.st_mtime}".encode()).hexdigest()
    
    def detect_and_crop_face(self, frame):
        """
        Detect the largest face in frame and crop around it.
        Returns the cropped face or original frame if no face detected.
        """
        if self.face_cascade is None:
            return frame
            
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)
        
        if len(faces) > 0:
            # Get the largest face
            largest_face = max(faces, key=lambda x: x[2] * x[3])
            x, y, w, h = largest_face
            
            # Add some padding around the face
            padding = int(min(w, h) * 0.3)
            x1 = max(0, x - padding)
            y1 = max(0, y - padding)
            x2 = min(frame.shape[1], x + w + padding)
            y2 = min(frame.shape[0], y + h + padding)
            
            return frame[y1:y2, x1:x2]
        
        return frame
    
    def extract_frames_from_video(self, video_path: Path, output_subdir: Path) -> Dict[str, any]:
        """
        Extract frames from a single video file.
        
        Returns:
            Dictionary with extraction statistics
        """
        stats = {
            'video_path': str(video_path),
            'total_frames': 0,
            'extracted_frames': 0,
            'failed_frames': 0,
            'duration_seconds': 0,
            'fps': 0,
            'success': False
        }
        
        try:
            # Open video
            cap = cv2.VideoCapture(str(video_path))
            if not cap.isOpened():
                logger.error(f"Could not open video: {video_path}")
                return stats
            
            # Get video properties
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            stats['total_frames'] = total_frames
            stats['fps'] = fps
            stats['duration_seconds'] = total_frames / fps if fps > 0 else 0
            
            # Create output directory
            output_subdir.mkdir(parents=True, exist_ok=True)
            
            frame_count = 0
            extracted_count = 0
            failed_count = 0
            
            # Extract frames
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Check if we should extract this frame
                if frame_count % self.frame_interval == 0:
                    try:
                        # Apply face detection if enabled
                        if self.face_detection:
                            frame = self.detect_and_crop_face(frame)
                        
                        # Resize frame
                        if self.target_size:
                            frame = cv2.resize(frame, self.target_size, interpolation=cv2.INTER_AREA)
                        
                        # Save frame
                        frame_filename = f"frame_{frame_count:06d}.jpg"
                        frame_path = output_subdir / frame_filename
                        
                        success = cv2.imwrite(
                            str(frame_path), 
                            frame, 
                            [cv2.IMWRITE_JPEG_QUALITY, 95]
                        )
                        
                        if success:
                            extracted_count += 1
                        else:
                            failed_count += 1
                            logger.warning(f"Failed to save frame {frame_count} from {video_path}")
                            
                    except Exception as e:
                        failed_count += 1
                        logger.error(f"Error processing frame {frame_count} from {video_path}: {e}")
                
                frame_count += 1
                
                # Check if we've reached the maximum frames limit
                if (self.max_frames_per_video and 
                    extracted_count >= self.max_frames_per_video):
                    break
            
            cap.release()
            
            stats['extracted_frames'] = extracted_count
            stats['failed_frames'] = failed_count
            stats['success'] = extracted_count > 0
            
            logger.info(f"Extracted {extracted_count} frames from {video_path.name} "
                       f"({failed_count} failed)")
            
        except Exception as e:
            logger.error(f"Error processing video {video_path}: {e}")
            stats['success'] = False
            
        return stats
    
    def find_video_files(self) -> List[Path]:
        """Find all video files in the input directory."""
        video_files = []
        
        for ext in self.video_extensions:
            video_files.extend(self.input_dir.rglob(f"*{ext}"))
            video_files.extend(self.input_dir.rglob(f"*{ext.upper()}"))
        
        logger.info(f"Found {len(video_files)} video files")
        return sorted(video_files)
    
    def extract_all_frames(self):
        """
        Extract frames from all videos in the input directory.
        Maintains the directory structure in the output.
        """
        video_files = self.find_video_files()
        
        if not video_files:
            logger.error("No video files found in the input directory")
            return
        
        total_stats = {
            'total_videos': len(video_files),
            'processed_videos': 0,
            'successful_videos': 0,
            'failed_videos': 0,
            'total_frames_extracted': 0,
            'total_frames_failed': 0,
            'start_time': time.time()
        }
        
        logger.info(f"Starting extraction of {len(video_files)} videos...")
        
        # Process each video
        for video_path in tqdm(video_files, desc="Extracting frames"):
            # Check if already processed
            video_hash = self.get_video_hash(video_path)
            if video_hash in self.processed_videos:
                logger.info(f"Skipping already processed video: {video_path.name}")
                continue
            
            # Determine output path (maintain relative structure)
            rel_path = video_path.relative_to(self.input_dir)
            output_subdir = self.output_dir / rel_path.parent / rel_path.stem
            
            # Extract frames
            stats = self.extract_frames_from_video(video_path, output_subdir)
            
            # Update statistics
            total_stats['processed_videos'] += 1
            if stats['success']:
                total_stats['successful_videos'] += 1
                total_stats['total_frames_extracted'] += stats['extracted_frames']
            else:
                total_stats['failed_videos'] += 1
            
            total_stats['total_frames_failed'] += stats['failed_frames']
            
            # Mark as processed
            self.processed_videos[video_hash] = True
            
            # Save progress periodically
            if total_stats['processed_videos'] % 10 == 0:
                self.save_progress()
        
        # Final save
        self.save_progress()
        
        # Report final statistics
        total_stats['end_time'] = time.time()
        total_stats['total_time_seconds'] = total_stats['end_time'] - total_stats['start_time']
        
        logger.info("="*50)
        logger.info("EXTRACTION COMPLETE")
        logger.info("="*50)
        logger.info(f"Total videos: {total_stats['total_videos']}")
        logger.info(f"Processed videos: {total_stats['processed_videos']}")
        logger.info(f"Successful videos: {total_stats['successful_videos']}")
        logger.info(f"Failed videos: {total_stats['failed_videos']}")
        logger.info(f"Total frames extracted: {total_stats['total_frames_extracted']}")
        logger.info(f"Total frames failed: {total_stats['total_frames_failed']}")
        logger.info(f"Total time: {total_stats['total_time_seconds']:.2f} seconds")
        logger.info(f"Output directory: {self.output_dir}")
        
        # Save final statistics
        stats_file = self.output_dir / 'extraction_stats.json'
        with open(stats_file, 'w') as f:
            json.dump(total_stats, f, indent=2)


def main():
    parser = argparse.ArgumentParser(
        description="Extract frames from video files for machine learning training",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage
  python extract_frames.py --input_dir ./videos --output_dir ./frames
  
  # Extract every 10th frame, resize to 128x128
  python extract_frames.py --input_dir ./videos --output_dir ./frames --frame_interval 10 --width 128 --height 128
  
  # Enable face detection and limit frames per video
  python extract_frames.py --input_dir ./videos --output_dir ./frames --face_detection --max_frames 100
  
  # Resume interrupted extraction
  python extract_frames.py --input_dir ./videos --output_dir ./frames --resume
        """
    )
    
    parser.add_argument('--input_dir', type=str, required=True,
                       help='Directory containing video files')
    parser.add_argument('--output_dir', type=str, required=True,
                       help='Directory to save extracted frames')
    parser.add_argument('--frame_interval', type=int, default=5,
                       help='Extract every Nth frame (default: 5)')
    parser.add_argument('--width', type=int, default=224,
                       help='Target frame width (default: 224)')
    parser.add_argument('--height', type=int, default=224,
                       help='Target frame height (default: 224)')
    parser.add_argument('--max_frames', type=int, default=None,
                       help='Maximum frames to extract per video (default: no limit)')
    parser.add_argument('--face_detection', action='store_true',
                       help='Enable face detection and cropping')
    parser.add_argument('--extensions', nargs='+', 
                       default=['.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv', '.m4v'],
                       help='Video file extensions to process')
    parser.add_argument('--resume', action='store_true',
                       help='Resume from previous extraction (automatic by default)')
    
    args = parser.parse_args()
    
    # Validate inputs
    input_path = Path(args.input_dir)
    if not input_path.exists():
        logger.error(f"Input directory does not exist: {args.input_dir}")
        return 1
    
    if not input_path.is_dir():
        logger.error(f"Input path is not a directory: {args.input_dir}")
        return 1
    
    # Create extractor and run
    extractor = FrameExtractor(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        frame_interval=args.frame_interval,
        target_size=(args.width, args.height),
        max_frames_per_video=args.max_frames,
        face_detection=args.face_detection,
        video_extensions=args.extensions
    )
    
    try:
        extractor.extract_all_frames()
        logger.info("Frame extraction completed successfully!")
        return 0
    except KeyboardInterrupt:
        logger.info("Extraction interrupted by user. Progress has been saved.")
        return 0
    except Exception as e:
        logger.error(f"Extraction failed: {e}")
        return 1


if __name__ == '__main__':
    exit(main())
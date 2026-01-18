#!/usr/bin/env python3
"""
Perimeter Monitor - Vehicle detection and recording system.

Monitors video streams (RTSP or files) for vehicles and records clips
when vehicles are detected.

Usage:
  RTSP stream:  python mon_perim.py "rtsp://user:pass@ip:port/stream" -d
  Video file:   python mon_perim.py video.mp4 -d
"""

import argparse
import cv2
import os
import time
from datetime import datetime

from config import (
  CONFIDENCE_THRESHOLD,
  CROP_Y1, CROP_Y2, CROP_X1, CROP_X2,
  DEFAULT_FPS,
  STATUS_INTERVAL_SECONDS,
  QUEUE_WARNING_INTERVAL
)
from video_reader import VideoStreamReader
from vehicle_detector import VehicleDetector
from clip_recorder import ClipRecorder

# Import local configuration for clips directory (with fallback)
try:
  from config_local import CLIPS_DIR
except ImportError:
  CLIPS_DIR = "clips"  # Default fallback if config_local.py doesn't exist


def parse_args():
  """Parse command line arguments."""
  parser = argparse.ArgumentParser(
    description='Perimeter monitor - Vehicle detection and recording'
  )
  parser.add_argument(
    'video_source',
    help='Path to video file or RTSP URL (e.g., rtsp://user:pass@ip:port/stream)'
  )
  parser.add_argument(
    '-o', '--output',
    help='Path to output video file (default: CLIPS_DIR/input_output.avi for files, CLIPS_DIR for RTSP)',
    default=None
  )
  parser.add_argument(
    '-c', '--confidence',
    type=float,
    default=CONFIDENCE_THRESHOLD,
    help=f'Confidence threshold (default: {CONFIDENCE_THRESHOLD})'
  )
  parser.add_argument(
    '-d', '--display',
    action='store_true',
    help='Display live video window during processing'
  )
  parser.add_argument(
    '-t', '--rtsp-transport',
    choices=['tcp', 'udp'],
    default='tcp',
    help='RTSP transport protocol (default: tcp, ignored for video files)'
  )
  return parser.parse_args()


def setup_video_reader(video_source, is_rtsp, rtsp_transport):
  """Initialize video reader and measure FPS (for RTSP only)."""
  print("Initializing video reader...")
  reader = VideoStreamReader(
    video_source,
    is_rtsp=is_rtsp,
    rtsp_transport=rtsp_transport
  )

  if not reader.isOpened():
    print(f"Error: Could not open video source {video_source}")
    return None, None

  # For RTSP: measure actual FPS (stream metadata is often wrong)
  # For files: use embedded FPS from file metadata
  if is_rtsp:
    print("Measuring actual source frame rate...")
    measured_fps = reader.measure_fps()
    if measured_fps:
      print(f"  Measured FPS: {measured_fps:.2f}")
  else:
    measured_fps = None  # Will use file's embedded FPS

  # Start threaded reader
  print("Starting threaded reader...")
  reader.start()

  return reader, measured_fps


def validate_frame_size(reader, crop_x1, crop_x2, crop_y1, crop_y2):
  """Validate frame dimensions are sufficient for crop region."""
  frame_width = int(reader.get(cv2.CAP_PROP_FRAME_WIDTH))
  frame_height = int(reader.get(cv2.CAP_PROP_FRAME_HEIGHT))
  print(f"Stream resolution: {frame_width}x{frame_height}")

  if frame_width < crop_x2 or frame_height < crop_y2:
    print(f"\nERROR: Stream resolution ({frame_width}x{frame_height}) is too small!")
    print(f"Required minimum: {crop_x2}x{crop_y2} (to support crop region)")
    print(f"Crop region: x({crop_x1}:{crop_x2}), y({crop_y1}:{crop_y2})")
    print("\nPlease configure your camera to output a higher resolution stream.")
    return False

  return True


def determine_fps(reader, measured_fps, is_rtsp):
  """Determine output FPS from measured or reported values."""
  reported_fps = reader.get(cv2.CAP_PROP_FPS)

  if is_rtsp:
    # RTSP mode: prefer measured FPS over reported
    if measured_fps is not None:
      fps = measured_fps
      print(f"Stream reports FPS: {reported_fps:.2f}, Using measured: {fps:.2f}")
    else:
      fps = reported_fps if reported_fps > 0 else DEFAULT_FPS
      print(f"Stream reports FPS: {reported_fps:.2f}, Measurement failed, Using: {fps:.2f}")
  else:
    # File mode: use file's embedded FPS
    fps = reported_fps if reported_fps > 0 else DEFAULT_FPS
    print(f"Input file FPS: {fps:.2f}")

  return fps


def print_rtsp_status(frame_count, overall_start_time, fps_counter, fps_start_time):
  """Print status line for RTSP mode."""
  current_time = time.time()
  interval_fps = fps_counter / (current_time - fps_start_time) if (current_time - fps_start_time) > 0 else 0
  running_time = current_time - overall_start_time

  timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
  hours = int(running_time // 3600)
  minutes = int((running_time % 3600) // 60)
  seconds = int(running_time % 60)

  print(f"[{timestamp}] Runtime: {hours:02d}:{minutes:02d}:{seconds:02d} | Frames: {frame_count} | FPS: {interval_fps:.2f}")


def print_file_status(current_fps, frame_count, total_frames):
  """Print status line for file mode."""
  if total_frames > 0:
    percent_complete = min(100.0, (frame_count / total_frames) * 100)
    print(f"FPS: {current_fps:.2f} | {percent_complete:.1f}% complete")
  else:
    print(f"FPS: {current_fps:.2f} | % complete: N/A")


def print_timing_stats(loop_times, read_times, track_times):
  """Print performance timing statistics."""
  print("\nPerformance Timing Statistics:")
  print(f"{'Metric':<20} {'Min (ms)':>10} {'Max (ms)':>10} {'Avg (ms)':>10} {'Frames':>10}")
  print("-" * 62)

  stats = [
    ("Loop time", loop_times),
    ("reader.read()", read_times),
    ("model.track()", track_times)
  ]

  for label, times in stats:
    if times:
      min_ms = min(times) * 1000
      max_ms = max(times) * 1000
      avg_ms = (sum(times) / len(times)) * 1000
      print(f"{label:<20} {min_ms:>10.2f} {max_ms:>10.2f} {avg_ms:>10.2f} {len(times):>10}")
    else:
      print(f"{label:<20} {'N/A':>10} {'N/A':>10} {'N/A':>10} {0:>10}")


def main():
  args = parse_args()

  video_source = args.video_source
  is_rtsp = video_source.startswith('rtsp://')

  # Setup output paths
  if args.output is None:
    if is_rtsp:
      output_video = None
      output_dir = CLIPS_DIR
    else:
      # For video files, write output to network drive
      input_filename = os.path.basename(video_source)
      base_name = os.path.splitext(input_filename)[0]
      output_video = os.path.join(CLIPS_DIR, f"{base_name}_output.avi")
      output_dir = None
  else:
    output_video = args.output
    output_dir = None

  # Print configuration
  print(f"Video source: {video_source}")
  if is_rtsp:
    print(f"RTSP mode: Clips will be saved on vehicle detection")
    print(f"RTSP transport: {args.rtsp_transport}")
    if output_dir:
      print(f"Clip output directory: {output_dir}")
  else:
    print(f"Output video: {output_video}")
  print(f"Confidence threshold: {args.confidence}")
  print(f"Live display: {'enabled' if args.display else 'disabled'}")

  # Initialize detector
  detector = VehicleDetector(confidence=args.confidence)

  # Initialize video reader
  reader, measured_fps = setup_video_reader(
    video_source, is_rtsp, args.rtsp_transport
  )
  if reader is None:
    return

  # Validate frame size
  if not validate_frame_size(reader, CROP_X1, CROP_X2, CROP_Y1, CROP_Y2):
    reader.stop()
    return

  # Print frame info (only for file mode - RTSP streams don't have total frames)
  total_frames = int(reader.get(cv2.CAP_PROP_FRAME_COUNT))
  if not is_rtsp and total_frames > 0:
    print(f"Total frames in input: {total_frames}")

  # Determine FPS for output
  fps = determine_fps(reader, measured_fps, is_rtsp)

  # Calculate crop dimensions
  width = CROP_X2 - CROP_X1
  height = CROP_Y2 - CROP_Y1

  # Setup output (recorder for RTSP, simple writer for files)
  if is_rtsp:
    recorder = ClipRecorder(output_dir, fps, width, height)
    out = None
  else:
    recorder = None
    # Ensure output directory exists for file mode
    output_file_dir = os.path.dirname(output_video)
    if output_file_dir:
      os.makedirs(output_file_dir, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    out = cv2.VideoWriter(output_video, fourcc, fps, (width, height))

    # Create training frames directory for file mode
    output_base = os.path.splitext(output_video)[0]
    training_frames_dir = f"{output_base}_frames"
    os.makedirs(training_frames_dir, exist_ok=True)
    print(f"Training frames will be saved to: {training_frames_dir}")

  # Processing state
  print("Starting processing...")
  print("Press 'q' to quit (or Ctrl+C)")
  frame_count = 0
  fps_start_time = time.time()
  fps_counter = 0
  overall_start_time = time.time()
  loop_times = []
  read_times = []
  track_times = []

  # RTSP-specific state
  if is_rtsp:
    last_status_time = time.time()
    last_queue_warning_time = 0

  try:
    while reader.isOpened():
      loop_start = time.time()

      # Read frame
      read_start = time.time()
      success, frame = reader.read()
      read_times.append(time.time() - read_start)

      if not success:
        break

      # Crop frame
      cropped_frame = frame[CROP_Y1:CROP_Y2, CROP_X1:CROP_X2]

      # Detect vehicles
      track_start = time.time()
      results = detector.detect(cropped_frame)
      track_times.append(time.time() - track_start)

      # Get annotated frame and detection results
      annotated_frame = detector.get_annotated_frame(results)
      vehicle_detected, detections_by_class = detector.get_detections(results)
      detections = detector.get_detection_boxes(results)

      # Log detections
      if detections_by_class:
        for class_name, track_ids in detections_by_class.items():
          print(f"Frame {frame_count}: {class_name.upper()} DETECTED! (IDs: {track_ids})")

      # Handle recording
      if is_rtsp:
        if vehicle_detected:
          recorder.on_vehicle_detected(annotated_frame, cropped_frame, detections)
        else:
          recorder.on_no_vehicle(annotated_frame, cropped_frame, detections)

        # Queue monitoring
        queue_size = reader.qsize()
        current_time = time.time()
        if queue_size >= 2 and (current_time - last_queue_warning_time) >= QUEUE_WARNING_INTERVAL:
          timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
          print(f"[{timestamp}] WARNING: Processing too slow! {queue_size} frames buffered")
          last_queue_warning_time = current_time
      else:
        out.write(annotated_frame)

        # Save training frame (JPEG + YOLO annotation)
        frame_basename = f"frame_{frame_count:06d}"
        jpg_path = os.path.join(training_frames_dir, f"{frame_basename}.jpg")
        txt_path = os.path.join(training_frames_dir, f"{frame_basename}.txt")

        cv2.imwrite(jpg_path, cropped_frame)
        with open(txt_path, 'w') as f:
          for class_id, x, y, w, h, conf in detections:
            f.write(f"{class_id} {x:.6f} {y:.6f} {w:.6f} {h:.6f} {conf:.6f}\n")

      # Status updates
      if is_rtsp:
        current_time = time.time()
        if current_time - last_status_time >= STATUS_INTERVAL_SECONDS:
          print_rtsp_status(frame_count, overall_start_time, fps_counter, fps_start_time)
          fps_counter = 0
          fps_start_time = current_time
          last_status_time = current_time
        else:
          fps_counter += 1
      else:
        fps_counter += 1
        if fps_counter >= 10:
          current_fps = fps_counter / (time.time() - fps_start_time)
          print_file_status(current_fps, frame_count, total_frames)
          fps_counter = 0
          fps_start_time = time.time()

        if frame_count % 30 == 0:
          print(f"Processed {frame_count} frames...")

      # Display
      if args.display:
        cv2.imshow('Perimeter Monitor', annotated_frame)
      key = cv2.waitKey(1) & 0xFF
      if key == ord('q'):
        print("\nQuitting...")
        break

      frame_count += 1
      loop_times.append(time.time() - loop_start)

  except KeyboardInterrupt:
    print("\nStopping early...")

  finally:
    # Cleanup
    reader.stop()

    if is_rtsp:
      recorder.close()
    elif out is not None:
      out.release()

    if args.display:
      cv2.destroyAllWindows()

    # Print summary
    total_time = time.time() - overall_start_time
    if total_time > 0:
      overall_fps = frame_count / total_time
      print(f"Overall FPS: {overall_fps:.2f} (processed {frame_count} frames)")

    print_timing_stats(loop_times, read_times, track_times)

    if is_rtsp:
      print(f"Done! {recorder.get_clip_count()} clip(s) saved to {recorder.get_output_dir()}")
    else:
      print(f"Done! Output saved to {output_video}")


if __name__ == "__main__":
  main()

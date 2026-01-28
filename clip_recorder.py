"""
Clip recorder for saving video clips when vehicles are detected.
Handles start/stop recording with configurable buffer time.
"""

import cv2
import os
from datetime import datetime

from config import CLIP_BUFFER_SECONDS, CLASS_NAMES


class ClipRecorder:
  """
  Records video clips when vehicles are detected.
  Continues recording for a buffer period after the last detection.
  """

  def __init__(self, output_dir, fps, width, height, buffer_seconds=CLIP_BUFFER_SECONDS):
    """
    Initialize the clip recorder.

    Args:
      output_dir: Directory to save clips (created if doesn't exist)
      fps: Frames per second for output video
      width: Frame width
      height: Frame height
      buffer_seconds: Seconds to continue recording after last detection
    """
    self.output_dir = output_dir
    self.fps = fps
    self.width = width
    self.height = height
    self.buffer_frames = int(fps * buffer_seconds)

    # Recording state
    self.is_recording = False
    self.frames_since_last_vehicle = 0
    self.clip_count = 0
    self.current_writer = None
    self.current_clip_filename = None
    self.current_images_dir = None
    self.current_labels_dir = None
    self.current_clip_basename = None
    self.frame_number = 0

    # Video codec
    self.fourcc = cv2.VideoWriter_fourcc(*'MJPG')

    # Create output directory if needed
    if output_dir:
      os.makedirs(output_dir, exist_ok=True)

  def on_vehicle_detected(self, frame, clean_frame=None, detections=None):
    """
    Call when a vehicle is detected in the current frame.
    Starts recording if not already, or resets the buffer timer.

    Args:
      frame: The annotated frame to record
      clean_frame: The clean frame without annotations (for training export)
      detections: List of (class_id, x, y, w, h) tuples for YOLO annotations

    Returns:
      True if a new clip was started, False otherwise
    """
    started_new_clip = False

    if not self.is_recording:
      # Start new clip
      self.clip_count += 1

      # Determine vehicle type from first detection
      if detections and len(detections) > 0:
        first_class_id = detections[0][0]
        vehicle_type = CLASS_NAMES.get(first_class_id, 'vehicle')
      else:
        vehicle_type = 'vehicle'

      # Format timestamp as local time (ISO 8601 style)
      now = datetime.now()
      date_dir = now.strftime('%Y-%m-%d')
      timestamp = now.strftime('%Y-%m-%dT%H-%M-%S')
      self.current_clip_basename = f"{vehicle_type}_{timestamp}"

      # Create date subdirectory with images/ and labels/ subdirs
      day_output_dir = os.path.join(self.output_dir, date_dir)
      self.current_images_dir = os.path.join(day_output_dir, 'images')
      self.current_labels_dir = os.path.join(day_output_dir, 'labels')
      os.makedirs(self.current_images_dir, exist_ok=True)
      os.makedirs(self.current_labels_dir, exist_ok=True)

      self.current_clip_filename = os.path.join(
        day_output_dir,
        f"{self.current_clip_basename}.avi"
      )
      self.frame_number = 0

      self.current_writer = cv2.VideoWriter(
        self.current_clip_filename,
        self.fourcc,
        self.fps,
        (self.width, self.height)
      )
      self.is_recording = True
      self.frames_since_last_vehicle = 0
      print(f"Started recording clip: {self.current_clip_filename}")
      started_new_clip = True
    else:
      # Continue recording, reset buffer
      self.frames_since_last_vehicle = 0

    # Write frame
    if self.current_writer is not None:
      self.current_writer.write(frame)

    # Save training frame
    if clean_frame is not None:
      self._save_training_frame(clean_frame, detections)

    return started_new_clip

  def on_no_vehicle(self, frame, clean_frame=None, detections=None):
    """
    Call when no vehicle is detected in the current frame.
    Continues recording during buffer period, then stops.

    Args:
      frame: The annotated frame (recorded if still in buffer period)
      clean_frame: The clean frame without annotations (for training export)
      detections: List of (class_id, x, y, w, h) tuples for YOLO annotations

    Returns:
      True if recording was stopped, False otherwise
    """
    stopped_recording = False

    if self.is_recording:
      self.frames_since_last_vehicle += 1

      # Still in buffer period - write frame
      if self.current_writer is not None:
        self.current_writer.write(frame)

      # Save training frame
      if clean_frame is not None:
        self._save_training_frame(clean_frame, detections)

      # Check if buffer expired
      if self.frames_since_last_vehicle >= self.buffer_frames:
        self._stop_recording()
        print(f"Stopped recording clip (buffer expired)")
        stopped_recording = True

    return stopped_recording

  def _save_training_frame(self, clean_frame, detections):
    """
    Save a clean JPEG frame and YOLO annotation file.

    Args:
      clean_frame: The frame without annotations
      detections: List of (class_id, x, y, w, h, conf) tuples (normalized 0-1)
    """
    if self.current_images_dir is None or self.current_labels_dir is None:
      return

    self.frame_number += 1
    # Format: {vehicle}_{timestamp}_{frame_number}
    frame_basename = f"{self.current_clip_basename}_{self.frame_number:04d}"

    # Save JPEG to images/
    jpg_path = os.path.join(self.current_images_dir, f"{frame_basename}.jpg")
    cv2.imwrite(jpg_path, clean_frame)

    # Save YOLO annotation to labels/
    txt_path = os.path.join(self.current_labels_dir, f"{frame_basename}.txt")
    with open(txt_path, 'w') as f:
      if detections:
        for class_id, x, y, w, h, conf in detections:
          f.write(f"{class_id} {x:.6f} {y:.6f} {w:.6f} {h:.6f} {conf:.6f}\n")
      # Empty file if no detections

  def _stop_recording(self):
    """Internal method to stop recording and close the writer."""
    if self.current_writer is not None:
      self.current_writer.release()
      self.current_writer = None
    self.is_recording = False
    self.frames_since_last_vehicle = 0
    self.current_images_dir = None
    self.current_labels_dir = None
    self.current_clip_basename = None
    self.frame_number = 0

  def close(self):
    """
    Finalize recording. Call when shutting down.

    Returns:
      True if a partial clip was saved, False otherwise
    """
    saved_partial = False
    if self.is_recording:
      self._stop_recording()
      print("Saved partial clip (interrupted)")
      saved_partial = True
    return saved_partial

  def get_clip_count(self):
    """Return the total number of clips recorded."""
    return self.clip_count

  def get_output_dir(self):
    """Return the output directory path."""
    return self.output_dir

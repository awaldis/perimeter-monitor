"""
Clip recorder for saving video clips when vehicles are detected.
Handles start/stop recording with configurable buffer time.
"""

import cv2
import os
import time

from config import CLIP_BUFFER_SECONDS


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

    # Video codec
    self.fourcc = cv2.VideoWriter_fourcc(*'MJPG')

    # Create output directory if needed
    if output_dir:
      os.makedirs(output_dir, exist_ok=True)

  def on_vehicle_detected(self, frame):
    """
    Call when a vehicle is detected in the current frame.
    Starts recording if not already, or resets the buffer timer.

    Args:
      frame: The annotated frame to record

    Returns:
      True if a new clip was started, False otherwise
    """
    started_new_clip = False

    if not self.is_recording:
      # Start new clip
      self.clip_count += 1
      self.current_clip_filename = os.path.join(
        self.output_dir,
        f"clip_{int(time.time())}_{self.clip_count}.avi"
      )
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

    return started_new_clip

  def on_no_vehicle(self, frame):
    """
    Call when no vehicle is detected in the current frame.
    Continues recording during buffer period, then stops.

    Args:
      frame: The annotated frame (recorded if still in buffer period)

    Returns:
      True if recording was stopped, False otherwise
    """
    stopped_recording = False

    if self.is_recording:
      self.frames_since_last_vehicle += 1

      # Still in buffer period - write frame
      if self.current_writer is not None:
        self.current_writer.write(frame)

      # Check if buffer expired
      if self.frames_since_last_vehicle >= self.buffer_frames:
        self._stop_recording()
        print(f"Stopped recording clip (buffer expired)")
        stopped_recording = True

    return stopped_recording

  def _stop_recording(self):
    """Internal method to stop recording and close the writer."""
    if self.current_writer is not None:
      self.current_writer.release()
      self.current_writer = None
    self.is_recording = False
    self.frames_since_last_vehicle = 0

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

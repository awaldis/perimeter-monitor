"""
Threaded video stream reader for efficient frame capture.
Supports both video files and RTSP streams.
"""

import cv2
import time
import os
import threading
from datetime import datetime
from queue import Queue, Empty

from config import (
  READER_QUEUE_SIZE,
  FPS_DRAIN_FRAMES,
  FPS_MEASURE_FRAMES,
  RECONNECT_DELAY_SECONDS
)


class VideoStreamReader:
  """
  Threaded video stream reader that decouples frame capture from processing.
  Continuously reads frames in a background thread and stores them in a queue.
  """

  def __init__(self, src, queue_size=READER_QUEUE_SIZE, is_rtsp=False, rtsp_transport='tcp'):
    """
    Initialize the video stream reader.

    Args:
      src: Video source (file path or RTSP URL)
      queue_size: Maximum number of frames to buffer
      is_rtsp: Whether source is an RTSP stream
      rtsp_transport: 'tcp' or 'udp' for RTSP streams
    """
    # Remember source config so the stream can be reopened after a drop
    self.src = src
    self.is_rtsp = is_rtsp
    self.rtsp_transport = rtsp_transport

    self._open_stream()

    self.queue = Queue(maxsize=queue_size)
    self.stopped = False
    self.read_error = False
    self.reconnecting = False

    # Track frame arrival times to measure actual source FPS
    self.frame_intervals = []
    self.last_frame_time = None
    self.measured_fps = None

  def _open_stream(self):
    """Create the video capture for this source (also used when reconnecting)."""
    if self.is_rtsp:
      if self.rtsp_transport == 'tcp':
        os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"
        self.stream = cv2.VideoCapture(self.src, cv2.CAP_FFMPEG)
      else:
        self.stream = cv2.VideoCapture(self.src)
      self.stream.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    else:
      self.stream = cv2.VideoCapture(self.src)

  def start(self):
    """Start the background thread for reading frames."""
    self._thread = threading.Thread(target=self._reader, daemon=True)
    self._thread.start()
    return self

  def _reader(self):
    """Background thread that continuously reads frames."""
    while not self.stopped:
      if not self.queue.full():
        ret, frame = self.stream.read()
        frame_arrival_time = time.time()  # Timestamp AFTER read completes

        if not ret:
          # For files a failed read means end of video - stop normally.
          if not self.is_rtsp:
            self.read_error = True
            self.stopped = True
            return

          # For RTSP the server may have gone quiet (FFmpeg gives up after its
          # internal read timeout). Reconnect instead of killing the reader.
          if not self._reconnect():
            return
          continue

        # Measure actual frame intervals for FPS calculation
        if self.last_frame_time is not None:
          interval = frame_arrival_time - self.last_frame_time
          self.frame_intervals.append(interval)

          # Keep only last 30 intervals for running average
          if len(self.frame_intervals) > 30:
            self.frame_intervals.pop(0)

          # Calculate measured FPS after we have enough samples
          if len(self.frame_intervals) >= 10:
            avg_interval = sum(self.frame_intervals) / len(self.frame_intervals)
            self.measured_fps = 1.0 / avg_interval if avg_interval > 0 else None

        self.last_frame_time = frame_arrival_time
        self.queue.put((ret, frame))
      else:
        # Queue is full, wait a bit before trying again
        time.sleep(0.001)

  def _reconnect(self):
    """
    Reopen the stream after a read failure, retrying until it comes back.

    Returns:
      True once reconnected, False if stop() was called while retrying.
    """
    self.reconnecting = True
    attempt = 0

    try:
      while not self.stopped:
        attempt += 1
        self._log(f"Stream read failed - reconnect attempt {attempt}")

        try:
          self.stream.release()
        except Exception:
          pass  # Already dead; nothing to salvage

        self._open_stream()

        # Opening an RTSP URL can succeed while no data flows, so require an
        # actual frame before declaring the stream healthy again.
        if self.stream.isOpened():
          ret, _ = self.stream.read()
          if ret:
            self._log(f"Reconnected after {attempt} attempt(s)")
            # Don't let the outage gap skew the measured FPS
            self.last_frame_time = None
            return True

        self._log(f"Reconnect failed - retrying in {RECONNECT_DELAY_SECONDS:g}s")
        self._sleep_unless_stopped(RECONNECT_DELAY_SECONDS)

      return False
    finally:
      self.reconnecting = False

  def _sleep_unless_stopped(self, seconds):
    """Sleep in short slices so stop() stays responsive during a retry wait."""
    deadline = time.time() + seconds
    while not self.stopped and time.time() < deadline:
      time.sleep(0.1)

  def _log(self, message):
    """Print a timestamped reader message."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {message}")

  def read(self):
    """
    Get the next frame from the queue.

    Blocks while the reader is reconnecting, so a stream outage doesn't look
    like end-of-stream to the caller.

    Returns:
      Tuple of (success, frame) like cv2.VideoCapture.read()
    """
    while True:
      # Only report failure once the buffer is drained and the reader is done
      if self.queue.empty() and (self.stopped or self.read_error):
        return False, None
      try:
        # Use timeout to allow re-checking the stop condition
        return self.queue.get(timeout=0.5)
      except Empty:
        continue

  def qsize(self):
    """Return the number of frames currently in the queue."""
    return self.queue.qsize()

  def isOpened(self):
    """
    Check if the video source is opened.

    Reports True while reconnecting so callers looping on isOpened() keep
    waiting through an outage instead of treating it as end-of-stream.
    """
    if self.reconnecting:
      return True
    return self.stream.isOpened()

  def get(self, prop):
    """Get a property from the video stream."""
    return self.stream.get(prop)

  def get_measured_fps(self):
    """
    Get the measured FPS based on actual frame arrival times.

    Returns:
      Measured FPS, or None if not enough samples yet.
    """
    return self.measured_fps

  def measure_fps(self, num_frames=FPS_MEASURE_FRAMES, drain_frames=FPS_DRAIN_FRAMES):
    """
    Measure actual source FPS before starting the reader thread.
    Must be called BEFORE start(). Drains any buffered frames first.

    Args:
      num_frames: Number of frames to measure over
      drain_frames: Number of frames to discard to clear buffers

    Returns:
      Measured FPS, or None if measurement failed
    """
    # Discard frames to clear any internal buffers
    print(f"  Discarding {drain_frames} frames to clear buffers...")
    for _ in range(drain_frames):
      ret, _ = self.stream.read()
      if not ret:
        return None

    # Now measure actual frame intervals
    print(f"  Measuring over {num_frames} frames...")
    intervals = []
    last_time = None

    for i in range(num_frames):
      ret, _ = self.stream.read()
      now = time.time()
      if not ret:
        break
      if last_time is not None:
        intervals.append(now - last_time)
      last_time = now

    if len(intervals) < 2:
      return None

    avg_interval = sum(intervals) / len(intervals)
    self.measured_fps = 1.0 / avg_interval if avg_interval > 0 else None
    return self.measured_fps

  def stop(self):
    """Stop the background thread and release the video stream."""
    self.stopped = True
    # Wait for reader thread to finish before releasing stream
    # This prevents segfault from releasing stream while thread is reading
    if hasattr(self, '_thread') and self._thread.is_alive():
      self._thread.join(timeout=2.0)
    self.stream.release()

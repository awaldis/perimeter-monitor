import cv2
import time
from ultralytics import YOLO
import torch
import argparse
import threading
from queue import Queue

# --- CONFIGURATION ---
CONFIDENCE_THRESHOLD = 0.5          # Sensitivity (0.0 to 1.0)

# COCO Dataset Class IDs relevant to vehicles:
# 2: car, 3: motorcycle, 5: bus, 7: truck
VEHICLE_CLASSES = [2, 3, 5, 7]

class VideoStreamReader:
  """
  Threaded video stream reader that decouples frame capture from processing.
  Continuously reads frames in a background thread and stores them in a queue.
  """
  def __init__(self, src, queue_size=3, is_rtsp=False, rtsp_transport='tcp'):
    """
    Args:
      src: Video source (file path or RTSP URL)
      queue_size: Maximum number of frames to buffer
      is_rtsp: Whether source is an RTSP stream
      rtsp_transport: 'tcp' or 'udp' for RTSP streams
    """
    import os

    # Configure video capture based on source type
    if is_rtsp:
      if rtsp_transport == 'tcp':
        os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"
        self.stream = cv2.VideoCapture(src, cv2.CAP_FFMPEG)
        self.stream.set(cv2.CAP_PROP_BUFFERSIZE, 1)
      else:
        self.stream = cv2.VideoCapture(src)
        self.stream.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    else:
      self.stream = cv2.VideoCapture(src)

    self.queue = Queue(maxsize=queue_size)
    self.stopped = False
    self.read_error = False

    # Track frame arrival times to measure actual source FPS
    self.frame_intervals = []
    self.last_frame_time = None
    self.measured_fps = None

  def start(self):
    """Start the background thread for reading frames."""
    thread = threading.Thread(target=self._reader, daemon=True)
    thread.start()
    return self

  def _reader(self):
    """Background thread that continuously reads frames."""
    while not self.stopped:
      if not self.queue.full():
        ret, frame = self.stream.read()
        frame_arrival_time = time.time()  # Timestamp AFTER read completes

        if not ret:
          self.read_error = True
          self.stopped = True
          return

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

  def read(self):
    """
    Get the next frame from the queue.
    Returns: (ret, frame) tuple like cv2.VideoCapture.read()
    """
    if self.queue.empty() and self.read_error:
      return False, None
    return self.queue.get()

  def qsize(self):
    """Return the number of frames currently in the queue."""
    return self.queue.qsize()

  def isOpened(self):
    """Check if the video source is opened."""
    return self.stream.isOpened()

  def get(self, prop):
    """Get a property from the video stream."""
    return self.stream.get(prop)

  def get_measured_fps(self):
    """
    Get the measured FPS based on actual frame arrival times.
    Returns None if not enough samples yet.
    """
    return self.measured_fps

  def measure_fps(self, num_frames=15, drain_frames=30):
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
    self.stream.release()

def main():
  # Parse command-line arguments
  parser = argparse.ArgumentParser(description='Vehicle detection in video using YOLOv8')
  parser.add_argument('video_source', help='Path to video file or RTSP URL (e.g., rtsp://user:pass@ip:port/stream)')
  parser.add_argument('-o', '--output', help='Path to output video file (default: input_output.avi for files, clip_TIMESTAMP.avi for RTSP)',
                      default=None)
  parser.add_argument('-c', '--confidence', type=float, default=CONFIDENCE_THRESHOLD,
                      help=f'Confidence threshold (default: {CONFIDENCE_THRESHOLD})')
  parser.add_argument('-d', '--display', action='store_true',
                      help='Display live video window during processing')
  parser.add_argument('-t', '--rtsp-transport', choices=['tcp', 'udp'], default='tcp',
                      help='RTSP transport protocol (default: tcp, ignored for video files)')
  args = parser.parse_args()

  VIDEO_SOURCE = args.video_source
  is_rtsp = VIDEO_SOURCE.startswith('rtsp://')

  # For files, use specified output or generate default
  # For RTSP, clips will be named with timestamps unless output is specified
  import os
  if args.output is None:
    if is_rtsp:
      OUTPUT_VIDEO = None  # Will generate timestamped filenames per clip
      output_dir = "clips"
      os.makedirs(output_dir, exist_ok=True)
    else:
      base_name = os.path.splitext(args.video_source)[0]
      OUTPUT_VIDEO = f"{base_name}_output.avi"
  else:
    OUTPUT_VIDEO = args.output
    output_dir = None

  confidence_threshold = args.confidence
  show_display = args.display
  rtsp_transport = args.rtsp_transport

  #--------------------------------------------------------------------------
  # Select the Hardware
  #--------------------------------------------------------------------------

  # Force usage of the GPU (device=0). If this fails, it falls back to CPU.
  device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

  print(f"Running inference on: {device}")
  print(f"Video source: {VIDEO_SOURCE}")
  if is_rtsp:
    print(f"RTSP mode: Clips will be saved on vehicle detection")
    print(f"RTSP transport: {rtsp_transport}")
    if output_dir:
      print(f"Clip output directory: {output_dir}")
  else:
    print(f"Output video: {OUTPUT_VIDEO}")
  print(f"Confidence threshold: {confidence_threshold}")
  print(f"Live display: {'enabled' if show_display else 'disabled'}")

  #--------------------------------------------------------------------------
  # Load the Model
  #--------------------------------------------------------------------------
  print("Loading YOLOv8 Nano model...")
  #model = YOLO("yolov8n.pt")
  model = YOLO("yolov8n.onnx", task='detect')

  #--------------------------------------------------------------------------
  # Open Video Source with Threaded Reader
  #--------------------------------------------------------------------------
  print("Initializing video reader...")
  reader = VideoStreamReader(
    VIDEO_SOURCE,
    queue_size=3,
    is_rtsp=is_rtsp,
    rtsp_transport=rtsp_transport
  )

  if not reader.isOpened():
    print(f"Error: Could not open video source {VIDEO_SOURCE}")
    return

  # Measure actual source FPS BEFORE starting the reader thread
  # (to avoid thread-safety issues with FFmpeg)
  print("Measuring actual source frame rate...")
  measured_fps = reader.measure_fps(num_frames=15)
  if measured_fps:
    print(f"  Measured FPS: {measured_fps:.2f}")

  # Now start the threaded reader
  print("Starting threaded reader...")
  reader.start()

  total_frames = int(reader.get(cv2.CAP_PROP_FRAME_COUNT))
  if total_frames > 0:
    print(f"Total frames in input: {total_frames}")
  else:
    print("Total frames in input: Unknown (source did not report frame count)")

  # Get and validate frame dimensions
  frame_width = int(reader.get(cv2.CAP_PROP_FRAME_WIDTH))
  frame_height = int(reader.get(cv2.CAP_PROP_FRAME_HEIGHT))
  print(f"Stream resolution: {frame_width}x{frame_height}")

  #--------------------------------------------------------------------------
  # Define the sub-region of the video frame that we want to analyze.
  #--------------------------------------------------------------------------
  crop_y1, crop_y2 =  202,  682 # Vertical range
  crop_x1, crop_x2 = 2152, 3752 # Horizontal range

  # Validate frame is large enough for crop region
  min_width = crop_x2
  min_height = crop_y2
  if frame_width < min_width or frame_height < min_height:
    print(f"\nERROR: Stream resolution ({frame_width}x{frame_height}) is too small!")
    print(f"Required minimum: {min_width}x{min_height} (to support crop region)")
    print(f"Crop region: x({crop_x1}:{crop_x2}), y({crop_y1}:{crop_y2})")
    print("\nPlease configure your camera to output a higher resolution stream.")
    print("You need at least 4K (3840x2160) resolution for the current crop settings.")
    reader.stop()
    return

  #--------------------------------------------------------------------------
  # Prepare Video Writer to save the output
  #--------------------------------------------------------------------------
  reported_fps = int(reader.get(cv2.CAP_PROP_FPS))

  # Use measured FPS if available, otherwise fall back to reported FPS
  if measured_fps is not None:
    fps = int(round(measured_fps))
    print(f"Stream reports FPS: {reported_fps}, Using measured: {fps}")
  else:
    fps = reported_fps if reported_fps > 0 else 25
    print(f"Stream reports FPS: {reported_fps}, Measurement failed, Using: {fps}")

  # Derive output dimensions from the crop
  width  = crop_x2 - crop_x1
  height = crop_y2 - crop_y1

  # XVID is a safe, widely supported codec for AVI files
  fourcc = cv2.VideoWriter_fourcc(*'XVID')

  # For file input, create the output writer immediately
  # For RTSP, we'll create writers dynamically per clip
  if is_rtsp:
    out = None
    is_recording = False
    last_vehicle_time = None
    buffer_frames = fps  # 1 second buffer = fps frames
    frames_since_last_vehicle = 0
    clip_count = 0
  else:
    out = cv2.VideoWriter(OUTPUT_VIDEO, fourcc, fps, (width, height))

  print("Starting processing...")
  print("Press 'q' to quit (or Ctrl+C, though it may take a moment to respond)")
  frame_count = 0

  # FPS Calculation Init
  fps_start_time = time.time()
  fps_counter = 0
  current_fps = 0.0
  overall_start_time = time.time()
  loop_times = []
  read_times = []
  track_times = []

  # For RTSP mode: status updates every 5 seconds
  if is_rtsp:
    last_status_time = time.time()
    status_interval = 5.0  # seconds

  try:
    while reader.isOpened():
      loop_start = time.time()
      read_start = time.time()
      success, frame = reader.read()
      read_end = time.time()
      if not success:
        break # End of video
      read_times.append(read_end - read_start)

      # Crop the frame to only the region of interest.
      cropped_frame = frame[crop_y1:crop_y2, crop_x1:crop_x2]

      # Run inference.
      track_start = time.time()
      results = model.track( cropped_frame,
                             persist=True,
                             rect=True,
                             imgsz=[480, 1600],
                             conf=confidence_threshold,
                             classes=VEHICLE_CLASSES,
                             device=device,
                             verbose=False,
                             half=False)
      track_end = time.time()
      track_times.append(track_end - track_start)

      # Draw the bounding boxes directly onto the frame.
      annotated_frame = results[0].plot()

      # Log detections to console and handle clip recording for RTSP
      boxes = results[0].boxes
      vehicle_detected = False
      if boxes.id is not None: # If tracking IDs exist
        class_ids = boxes.cls.cpu().numpy().astype(int)
        track_ids = boxes.id.cpu().numpy().astype(int)

        # Check if we see a car (Class 2)
        if 2 in class_ids:
          print(f"Frame {frame_count}: CAR DETECTED! (IDs: {track_ids[class_ids == 2]})")
          vehicle_detected = True

        # Check if we see a truck (Class 7)
        if 7 in class_ids:
          print(f"Frame {frame_count}: TRUCK DETECTED! (IDs: {track_ids[class_ids == 7]})")
          vehicle_detected = True

      # Handle clip recording for RTSP streams
      if is_rtsp:
        if vehicle_detected:
          # Vehicle detected - start or continue recording
          if not is_recording:
            # Start new clip
            clip_count += 1
            if output_dir:
              clip_filename = os.path.join(output_dir, f"clip_{int(time.time())}_{clip_count}.avi")
            else:
              clip_filename = OUTPUT_VIDEO
            out = cv2.VideoWriter(clip_filename, fourcc, fps, (width, height))
            is_recording = True
            frames_since_last_vehicle = 0
            print(f"Started recording clip: {clip_filename}")
          else:
            # Continue recording, reset buffer
            frames_since_last_vehicle = 0
        elif is_recording:
          # No vehicle detected but we're recording - check buffer
          frames_since_last_vehicle += 1
          if frames_since_last_vehicle >= buffer_frames:
            # Buffer expired, stop recording
            out.release()
            print(f"Stopped recording clip (1 second after last vehicle)")
            is_recording = False
            out = None

        # Check queue size to detect if processing is falling behind
        # Only warn once every 5 seconds to avoid spamming console
        queue_size = reader.qsize()
        if queue_size >= 2:
          current_time = time.time()
          if not hasattr(main, 'last_queue_warning_time') or current_time - main.last_queue_warning_time >= 5.0:
            from datetime import datetime
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            print(f"[{timestamp}] WARNING: Processing too slow! {queue_size} frames buffered (falling behind)")
            main.last_queue_warning_time = current_time

      # Status updates - different for RTSP vs file mode
      if is_rtsp:
        # RTSP mode: Print status every 5 seconds
        current_time = time.time()
        if current_time - last_status_time >= status_interval:
          # Calculate FPS over the interval
          interval_fps = fps_counter / (current_time - fps_start_time) if (current_time - fps_start_time) > 0 else 0
          running_time = current_time - overall_start_time

          # Format timestamp and running time
          from datetime import datetime
          timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
          hours = int(running_time // 3600)
          minutes = int((running_time % 3600) // 60)
          seconds = int(running_time % 60)

          print(f"[{timestamp}] Runtime: {hours:02d}:{minutes:02d}:{seconds:02d} | Frames: {frame_count} | FPS: {interval_fps:.2f}")

          # Reset counters for next interval
          fps_counter = 0
          fps_start_time = current_time
          last_status_time = current_time
        else:
          fps_counter += 1
      else:
        # File mode: Existing FPS calculation with percent complete
        fps_counter += 1
        if fps_counter >= 10:
          fps_end_time = time.time()
          current_fps = fps_counter / (fps_end_time - fps_start_time)
          if total_frames > 0:
            percent_complete = min(100.0, (frame_count / total_frames) * 100)
            print(f"FPS: {current_fps:.2f} | {percent_complete:.1f}% complete")
          else:
            print(f"FPS: {current_fps:.2f} | % complete: N/A")
          fps_counter = 0
          fps_start_time = fps_end_time

      # Save cropped frame with bounding boxes (only if recording)
      if is_rtsp:
        # Only write if we're actively recording
        if is_recording and out is not None:
          out.write(annotated_frame)
      else:
        # File mode: always write
        out.write(annotated_frame)

      # Display live video if enabled, or just process events
      if show_display:
        cv2.imshow('Vehicle Detection', annotated_frame)
        key = cv2.waitKey(1) & 0xFF
      else:
        # Still need to call waitKey for OpenCV event processing
        key = cv2.waitKey(1) & 0xFF

      if key == ord('q'):
        print("\nQuitting...")
        break

      frame_count += 1
      loop_times.append(time.time() - loop_start)

      # Write progress indicator to console (file mode only)
      if not is_rtsp and frame_count % 30 == 0:
        print(f"Processed {frame_count} frames...")

  except KeyboardInterrupt:
    print("\nStopping early...")

  finally:
    reader.stop()
    if out is not None:
      out.release()
      if is_rtsp and is_recording:
        print("Saved partial clip (interrupted)")
    if show_display:
      cv2.destroyAllWindows()
    overall_end_time = time.time()
    total_time = overall_end_time - overall_start_time
    if total_time > 0:
      overall_fps = frame_count / total_time
      print(f"Overall FPS: {overall_fps:.2f} (processed {frame_count} frames)")
    else:
      print(f"Overall FPS: N/A (processed {frame_count} frames)")

    # Print timing statistics as a table
    print("\nPerformance Timing Statistics:")
    print(f"{'Metric':<20} {'Min (ms)':>10} {'Max (ms)':>10} {'Avg (ms)':>10} {'Frames':>10}")
    print("-" * 62)

    stats = [
      ("Loop time", loop_times),
      ("cap.read()", read_times),
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

    if is_rtsp:
      print(f"Done! {clip_count} clip(s) saved to {output_dir if output_dir else 'current directory'}")
    else:
      print(f"Done! Output saved to {OUTPUT_VIDEO}")

if __name__ == "__main__":
  main()

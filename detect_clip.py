import cv2
import time
from ultralytics import YOLO
import torch
import argparse

# --- CONFIGURATION ---
CONFIDENCE_THRESHOLD = 0.5          # Sensitivity (0.0 to 1.0)

# COCO Dataset Class IDs relevant to vehicles:
# 2: car, 3: motorcycle, 5: bus, 7: truck
VEHICLE_CLASSES = [2, 3, 5, 7]

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
  # Open Video Source
  #--------------------------------------------------------------------------
  if is_rtsp:
    # Configure RTSP connection
    if rtsp_transport == 'tcp':
      os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"
      cap = cv2.VideoCapture(VIDEO_SOURCE, cv2.CAP_FFMPEG)
      cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Minimize latency
    else:
      cap = cv2.VideoCapture(VIDEO_SOURCE)
      cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
  else:
    cap = cv2.VideoCapture(VIDEO_SOURCE)

  if not cap.isOpened():
    print(f"Error: Could not open video source {VIDEO_SOURCE}")
    return

  total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
  if total_frames > 0:
    print(f"Total frames in input: {total_frames}")
  else:
    print("Total frames in input: Unknown (source did not report frame count)")

  # Get and validate frame dimensions
  frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
  frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
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
    cap.release()
    return

  #--------------------------------------------------------------------------
  # Prepare Video Writer to save the output
  #--------------------------------------------------------------------------
  fps = int(cap.get(cv2.CAP_PROP_FPS))
  if fps <= 0:
    fps = 25  # Default to 25 FPS if stream doesn't report FPS

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

  try:
    while cap.isOpened():
      loop_start = time.time()
      read_start = time.time()
      success, frame = cap.read()
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

      # FPS Calculation
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

      # Write progress indicator to console.
      if frame_count % 30 == 0:
        print(f"Processed {frame_count} frames...")

  except KeyboardInterrupt:
    print("\nStopping early...")

  finally:
    cap.release()
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

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
  parser.add_argument('input_video', help='Path to input video file')
  parser.add_argument('-o', '--output', help='Path to output video file (default: input_output.avi)',
                      default=None)
  parser.add_argument('-c', '--confidence', type=float, default=CONFIDENCE_THRESHOLD,
                      help=f'Confidence threshold (default: {CONFIDENCE_THRESHOLD})')
  args = parser.parse_args()

  INPUT_VIDEO = args.input_video
  if args.output is None:
    # Generate default output filename based on input
    import os
    base_name = os.path.splitext(args.input_video)[0]
    OUTPUT_VIDEO = f"{base_name}_output.avi"
  else:
    OUTPUT_VIDEO = args.output

  confidence_threshold = args.confidence

  # 1. Select the Hardware
  # Force usage of the GPU (device=0). If this fails, it falls back to CPU.
  device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
  print(f"Running inference on: {device}")
  print(f"Input video: {INPUT_VIDEO}")
  print(f"Output video: {OUTPUT_VIDEO}")
  print(f"Confidence threshold: {confidence_threshold}")

  # 2. Load the Model
  # 'yolov8n.pt' will automatically download on first run.
  # We use 'n' (nano) because the MX150 has limited VRAM.
  print("Loading YOLOv8 Nano model...")
  #model = YOLO("yolov8n.pt")
  model = YOLO("yolov8n.onnx")

  # 3. Open Video Source
  cap = cv2.VideoCapture(INPUT_VIDEO)
  if not cap.isOpened():
    print(f"Error: Could not open video file {INPUT_VIDEO}")
    return

  total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
  if total_frames > 0:
    print(f"Total frames in input: {total_frames}")
  else:
    print("Total frames in input: Unknown (source did not report frame count)")

  # 4. Prepare Video Writer to save the output
  fps = int(cap.get(cv2.CAP_PROP_FPS))

  # Define Region of Interest (ROI) once
  crop_y1, crop_y2 = 202, 682  # Vertical range
  crop_x1, crop_x2 = 1832, 3752 # Horizontal range

  # Derive output dimensions from the crop
  width  = crop_x2 - crop_x1
  height = crop_y2 - crop_y1

  # XVID is a safe, widely supported codec for AVI files
  fourcc = cv2.VideoWriter_fourcc(*'XVID')
  out    = cv2.VideoWriter(OUTPUT_VIDEO, fourcc, fps, (width, height))

  print("Starting processing... (Press Ctrl+C to stop early)")
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
                             conf=confidence_threshold,
                             classes=VEHICLE_CLASSES,
                             device=device,
                             verbose=False,
                             half=True)
      track_end = time.time()
      track_times.append(track_end - track_start)

      # Draw the bounding boxes directly onto the frame.
      annotated_frame = results[0].plot()

      # Log detections to console.
      boxes = results[0].boxes
      if boxes.id is not None: # If tracking IDs exist
        class_ids = boxes.cls.cpu().numpy().astype(int)
        track_ids = boxes.id.cpu().numpy().astype(int)

        # Check if we see a car (Class 2)
        if 2 in class_ids:
          print(f"Frame {frame_count}: CAR DETECTED! (IDs: {track_ids[class_ids == 2]})")

        # Check if we see a truck (Class 7)
        if 7 in class_ids:
          print(f"Frame {frame_count}: TRUCK DETECTED! (IDs: {track_ids[class_ids == 7]})")

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

      # Save cropped frame with bounding boxes.
      out.write(annotated_frame)
      frame_count += 1
      loop_times.append(time.time() - loop_start)

      # Write progress indicator to console.
      if frame_count % 30 == 0:
        print(f"Processed {frame_count} frames...")

  except KeyboardInterrupt:
    print("\nStopping early...")

  finally:
    cap.release()
    out.release()
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
    print(f"Done! Output saved to {OUTPUT_VIDEO}")

if __name__ == "__main__":
  main()

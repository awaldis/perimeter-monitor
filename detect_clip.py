import cv2
from ultralytics import YOLO
import torch

# --- CONFIGURATION ---
INPUT_VIDEO = "two_cars_south.mp4"  # Put your mp4 file in the same folder
OUTPUT_VIDEO = "two_cars_south2.avi" # The processed video
CONFIDENCE_THRESHOLD = 0.5          # Sensitivity (0.0 to 1.0)

# COCO Dataset Class IDs relevant to vehicles:
# 2: car, 3: motorcycle, 5: bus, 7: truck
VEHICLE_CLASSES = [2, 3, 5, 7]

def main():
    # 1. Select the Hardware
    # Force usage of the GPU (device=0). If this fails, it falls back to CPU.
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    print(f"Running inference on: {device}")

    # 2. Load the Model
    # 'yolov8n.pt' will automatically download on first run.
    # We use 'n' (nano) because the MX150 has limited VRAM.
    print("Loading YOLOv8 Nano model...")
    model = YOLO("yolov8n.pt")

    # 3. Open Video Source
    cap = cv2.VideoCapture(INPUT_VIDEO)
    if not cap.isOpened():
        print(f"Error: Could not open video file {INPUT_VIDEO}")
        return

    # 4. Prepare Video Writer to save the output
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps    = int(cap.get(cv2.CAP_PROP_FPS))
    
    # Overwrite for cropped video
    width = 2560
    height = 480

    # XVID is a safe, widely supported codec for AVI files
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(OUTPUT_VIDEO, fourcc, fps, (width, height))

    print("Starting processing... (Press Ctrl+C to stop early)")
    frame_count = 0

    try:
        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                break # End of video

            # --- THE FIX: CROP BEFORE INFERENCE ---
            # Define your Region of Interest (ROI)
            # Format: frame[y1:y2, x1:x2]
            crop_y1, crop_y2 = 202, 682  # Vertical range
            crop_x1, crop_x2 = 1192, 3752 # Horizontal range

            cropped_frame = frame[crop_y1:crop_y2, crop_x1:crop_x2]

            # 5. Run Inference
            # stream=True handles memory better for videos
            # classes=... filters the detection to ONLY vehicles, speeding up post-processing
            results = model.track(cropped_frame, persist=True, rect=True, conf=CONFIDENCE_THRESHOLD, 
                                  classes=VEHICLE_CLASSES, device=device, verbose=False)

            # 6. Visualize
            # plot() draws the bounding boxes directly onto the frame
            annotated_frame = results[0].plot()

            # 7. Log detections to console (The "Engineer" check)
            # We look at the boxes to see what was found this frame
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

            # 8. Save Frame
            out.write(annotated_frame)
            frame_count += 1
            
            # Simple progress indicator
            if frame_count % 30 == 0:
                print(f"Processed {frame_count} frames...")

    except KeyboardInterrupt:
        print("\nStopping early...")

    finally:
        cap.release()
        out.release()
        print(f"Done! Output saved to {OUTPUT_VIDEO}")

if __name__ == "__main__":
    main()
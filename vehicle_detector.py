"""
Vehicle detector using YOLOv8 for detecting and tracking vehicles.
"""

import torch
from ultralytics import YOLO

from config import (
  MODEL_PATH,
  MODEL_IMGSZ,
  CONFIDENCE_THRESHOLD,
  VEHICLE_CLASSES
)


class VehicleDetector:
  """
  Wrapper for YOLOv8 vehicle detection and tracking.
  """

  def __init__(self, model_path=MODEL_PATH, confidence=CONFIDENCE_THRESHOLD, device=None):
    """
    Initialize the vehicle detector.

    Args:
      model_path: Path to YOLO model file
      confidence: Detection confidence threshold (0.0 to 1.0)
      device: Device to run inference on ('cuda:0', 'cpu', or None for auto)
    """
    self.confidence = confidence
    self.vehicle_classes = VEHICLE_CLASSES

    # Auto-select device if not specified
    if device is None:
      self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    else:
      self.device = device

    # Load model
    print(f"Loading YOLO model from {model_path}...")
    self.model = YOLO(model_path, task='detect')
    print(f"Running inference on: {self.device}")

  def detect(self, frame):
    """
    Run detection and tracking on a frame.

    Args:
      frame: Input frame (BGR format from OpenCV)

    Returns:
      YOLO results object
    """
    results = self.model.track(
      frame,
      persist=True,
      rect=True,
      imgsz=MODEL_IMGSZ,
      conf=self.confidence,
      classes=self.vehicle_classes,
      device=self.device,
      verbose=False,
      half=False
    )
    return results

  def get_annotated_frame(self, results):
    """
    Get the frame with bounding boxes drawn.

    Args:
      results: YOLO results from detect()

    Returns:
      Annotated frame
    """
    return results[0].plot()

  def get_detections(self, results):
    """
    Extract detection information from results.

    Args:
      results: YOLO results from detect()

    Returns:
      Tuple of (vehicle_detected, cars, trucks) where:
        - vehicle_detected: True if any vehicle was detected
        - cars: List of track IDs for detected cars
        - trucks: List of track IDs for detected trucks
    """
    boxes = results[0].boxes
    cars = []
    trucks = []

    if boxes.id is not None:
      class_ids = boxes.cls.cpu().numpy().astype(int)
      track_ids = boxes.id.cpu().numpy().astype(int)

      # Extract car IDs (class 2)
      car_mask = class_ids == 2
      if car_mask.any():
        cars = track_ids[car_mask].tolist()

      # Extract truck IDs (class 7)
      truck_mask = class_ids == 7
      if truck_mask.any():
        trucks = track_ids[truck_mask].tolist()

    vehicle_detected = len(cars) > 0 or len(trucks) > 0
    return vehicle_detected, cars, trucks

  def get_device(self):
    """Return the device being used for inference."""
    return self.device

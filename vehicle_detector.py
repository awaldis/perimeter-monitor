"""
Vehicle detector using YOLOv8 for detecting and tracking vehicles.
"""

import torch
from ultralytics import YOLO

from config import (
  MODEL_PATH,
  MODEL_IMGSZ,
  CONFIDENCE_THRESHOLD,
  VEHICLE_CLASSES,
  CLASS_NAMES
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
      Tuple of (vehicle_detected, detections) where:
        - vehicle_detected: True if any vehicle was detected
        - detections: Dict mapping class_name -> list of track IDs
    """
    boxes = results[0].boxes
    detections = {}

    if boxes.id is not None:
      class_ids = boxes.cls.cpu().numpy().astype(int)
      track_ids = boxes.id.cpu().numpy().astype(int)

      for class_id, track_id in zip(class_ids, track_ids):
        class_name = CLASS_NAMES.get(class_id, f'class_{class_id}')
        if class_name not in detections:
          detections[class_name] = []
        detections[class_name].append(int(track_id))

    vehicle_detected = len(detections) > 0
    return vehicle_detected, detections

  def get_detection_boxes(self, results):
    """
    Extract bounding boxes for annotation export.

    Args:
      results: YOLO results from detect()

    Returns:
      List of (class_id, x_center, y_center, width, height, confidence) tuples,
      all normalized to 0-1. Uses COCO class IDs directly.
    """
    boxes = results[0].boxes
    detections = []

    if boxes.cls is not None and len(boxes.cls) > 0:
      class_ids = boxes.cls.cpu().numpy().astype(int)
      xywhn = boxes.xywhn.cpu().numpy()  # Normalized [x_center, y_center, w, h]
      confs = boxes.conf.cpu().numpy()  # Confidence scores

      for i in range(len(class_ids)):
        detections.append((
          class_ids[i],
          xywhn[i][0],  # x_center
          xywhn[i][1],  # y_center
          xywhn[i][2],  # width
          xywhn[i][3],  # height
          confs[i]      # confidence
        ))

    return detections

  def get_device(self):
    """Return the device being used for inference."""
    return self.device

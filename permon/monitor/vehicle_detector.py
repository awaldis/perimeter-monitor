"""
Vehicle detector: the stages that turn a cropped BGR image into tracked
vehicle boxes.

  preprocess  BGR image (H, W, 3) uint8 -> RGB tensor (1, 3, H, W) float32 0-1
  inference   tensor -> raw network output (see inference.py)
  nms         raw output -> detection boxes: keep candidates above the
              confidence threshold, then non-maximum suppression drops the
              overlapping duplicates of each vehicle
  track       detection boxes -> boxes with track IDs that follow each
              vehicle from frame to frame
  annotate    draw the boxes on a copy of the image

The network and the hardware it runs on are behind an inference engine
(inference.py). NMS, the tracker and the drawing come from ultralytics.
"""

import numpy as np
import torch
from ultralytics.data.augment import LetterBox
from ultralytics.engine.results import Results
from ultralytics.trackers.bot_sort import BOTSORT
from ultralytics.utils import IterableSimpleNamespace, nms, ops

from permon.config import (
  MODEL_PATH,
  MODEL_IMGSZ,
  INFERENCE_DEVICE,
  CONFIDENCE_THRESHOLD,
  VEHICLE_CLASSES,
  CLASS_NAMES
)
from permon.monitor.inference import create_engine
from permon.monitor.timing import StageTimer

# Boxes overlapping more than this are treated as the same vehicle by NMS
NMS_IOU_THRESHOLD = 0.7
NMS_MAX_DETECTIONS = 300

# BoT-SORT tracker settings (ultralytics' defaults for model.track()), except
# gmc_method: camera motion compensation runs optical flow on every frame and
# is pointless for a fixed camera.
TRACKER_SETTINGS = dict(
  tracker_type="botsort",
  track_high_thresh=0.25,   # First-stage match threshold
  track_low_thresh=0.1,     # Second-stage threshold for low-score matches
  new_track_thresh=0.25,    # Start a new track if no match is at least this
  track_buffer=30,          # Frames a lost track keeps its ID
  match_thresh=0.8,         # Association similarity threshold
  fuse_score=True,          # Fuse detection score with IoU when matching
  gmc_method="none",        # ultralytics default: "sparseOptFlow"
  proximity_thresh=0.5,     # ReID settings (unused: with_reid is off)
  appearance_thresh=0.8,
  with_reid=False,
  model="auto",
)


class VehicleDetector:
  """
  YOLO vehicle detection and tracking, one explicit stage at a time.

  Holds tracker state between frames, so use one detector per video stream.
  """

  def __init__(self, model_path=MODEL_PATH, confidence=CONFIDENCE_THRESHOLD, device=INFERENCE_DEVICE,
               timer=None):
    """
    Initialize the vehicle detector.

    Args:
      model_path: Path to YOLO model file or OpenVINO model folder
      confidence: Detection confidence threshold (0.0 to 1.0)
      device: Inference device (see INFERENCE_DEVICE in config.py)
      timer: StageTimer that records the stage times
    """
    self.confidence = confidence
    self.vehicle_classes = VEHICLE_CLASSES
    self.timer = timer or StageTimer()

    print(f"Loading YOLO model from {model_path}...")
    self.engine = create_engine(model_path, device)
    # Input size (height, width): fixed by exported models, else from config
    self.imgsz = self.engine.imgsz or tuple(MODEL_IMGSZ)
    self.letterbox = LetterBox(self.imgsz, auto=False, stride=self.engine.stride)
    print(f"Running inference with: {self.engine.description}, input {self.imgsz[1]}x{self.imgsz[0]}")

    self.tracker = BOTSORT(IterableSimpleNamespace(**TRACKER_SETTINGS), frame_rate=30)

  # --- Stages ---------------------------------------------------------------

  def preprocess(self, image):
    """
    Stage 'preprocess': put the image in the form the network takes.

    Args:
      image: BGR image, (H, W, 3) uint8

    Returns:
      float32 array, (1, 3, H, W), RGB, values 0-1
    """
    if image.shape[:2] != self.imgsz:
      # Resize keeping the aspect ratio and pad to the model's input size.
      # Skipped when the crop region already matches it.
      image = self.letterbox(image=image)
    tensor = image[..., ::-1].transpose(2, 0, 1)[None]  # BGR -> RGB, HWC -> NCHW
    tensor = np.ascontiguousarray(tensor, dtype=np.float32)
    tensor /= 255
    return tensor

  def infer(self, tensor):
    """
    Stage 'inference': run the YOLO network.

    Args:
      tensor: Output of preprocess()

    Returns:
      Raw predictions, torch tensor (1, 4 + classes, candidates)
    """
    return self.engine.infer(tensor)

  def nms(self, raw, image):
    """
    Stage 'nms': reduce the raw candidates to detection boxes.

    Args:
      raw: Output of infer()
      image: The BGR image the detections belong to

    Returns:
      ultralytics Results with one box per detected vehicle (no track IDs yet)
    """
    detections = nms.non_max_suppression(
      raw,
      self.confidence,
      NMS_IOU_THRESHOLD,
      self.vehicle_classes,
      max_det=NMS_MAX_DETECTIONS,
      end2end=self.engine.end2end,
    )[0]
    # Map boxes from model input coordinates back to the image and clip them
    detections[:, :4] = ops.scale_boxes(self.imgsz, detections[:, :4], image.shape)
    return Results(image, path="", names=self.engine.names, boxes=detections[:, :6])

  def track(self, results):
    """
    Stage 'track': match detections to the vehicles seen in earlier frames.

    Args:
      results: Output of nms()

    Returns:
      Results whose boxes carry track IDs. A detection the tracker hasn't
      confirmed yet (a vehicle's first frame) stays without an ID.
    """
    tracks = self.tracker.update(results.boxes.cpu().numpy(), results.orig_img)
    if len(tracks) == 0:
      return results
    results = results[tracks[:, -1].astype(int)]
    results.update(boxes=torch.as_tensor(tracks[:, :-1]))
    return results

  def annotate(self, results):
    """
    Stage 'annotate': draw the boxes.

    Args:
      results: Results from detect()

    Returns:
      Annotated copy of the image
    """
    return results.plot()

  # --------------------------------------------------------------------------

  def detect(self, frame):
    """
    Run preprocess -> inference -> nms -> track on a frame.

    Args:
      frame: Cropped BGR image from the video reader

    Returns:
      ultralytics Results object
    """
    with self.timer.stage("preprocess"):
      tensor = self.preprocess(frame)
    with self.timer.stage("inference"):
      raw = self.infer(tensor)
    with self.timer.stage("nms"):
      results = self.nms(raw, frame)
    with self.timer.stage("track"):
      return self.track(results)

  def get_annotated_frame(self, results):
    """
    Get the frame with bounding boxes drawn.

    Args:
      results: Results from detect()

    Returns:
      Annotated frame
    """
    with self.timer.stage("annotate"):
      return self.annotate(results)

  def get_detections(self, results):
    """
    Extract detection information from results.

    Args:
      results: Results from detect()

    Returns:
      Tuple of (vehicle_detected, detections, class_ids_by_track, boxes_by_track) where:
        - vehicle_detected: True if any vehicle was detected
        - detections: Dict mapping class_name -> list of track IDs
        - class_ids_by_track: Dict mapping track_id -> class_id
        - boxes_by_track: Dict mapping track_id -> (x_center, y_center, width, height) normalized 0-1
    """
    boxes = results.boxes
    detections = {}
    class_ids_by_track = {}
    boxes_by_track = {}

    if boxes.id is not None:
      class_ids = boxes.cls.cpu().numpy().astype(int)
      track_ids = boxes.id.cpu().numpy().astype(int)
      xywhn = boxes.xywhn.cpu().numpy()

      for i, (class_id, track_id) in enumerate(zip(class_ids, track_ids)):
        class_name = CLASS_NAMES.get(class_id, f'class_{class_id}')
        if class_name not in detections:
          detections[class_name] = []
        detections[class_name].append(int(track_id))
        class_ids_by_track[int(track_id)] = int(class_id)
        boxes_by_track[int(track_id)] = (
          float(xywhn[i][0]), float(xywhn[i][1]),
          float(xywhn[i][2]), float(xywhn[i][3])
        )

    vehicle_detected = len(detections) > 0
    return vehicle_detected, detections, class_ids_by_track, boxes_by_track

  def get_detection_boxes(self, results):
    """
    Extract bounding boxes for annotation export.

    Args:
      results: Results from detect()

    Returns:
      List of (class_id, x_center, y_center, width, height, confidence) tuples,
      all normalized to 0-1.
    """
    boxes = results.boxes
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

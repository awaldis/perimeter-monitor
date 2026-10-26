"""
Per-frame processing pipeline shared by live monitoring and the test rig.

Crops a raw frame, runs detection and tracking, and filters out stalled
(parked) vehicles. Side effects such as recording clips, writing output
files and sending alerts are left to the caller.
"""

from dataclasses import dataclass, field

from config import (
  CROP_Y1, CROP_Y2, CROP_X1, CROP_X2,
  STALL_SECONDS,
  STALL_MOVEMENT_THRESHOLD
)
from stall_tracker import StallTracker


@dataclass
class FrameResult:
  """Everything the pipeline learned about one frame."""
  frame_index: int
  cropped_frame: object
  annotated_frame: object
  # class_name -> track IDs, for all tracked vehicles (including stalled)
  detections_by_class: dict
  # class_name -> track IDs, excluding stalled vehicles
  active_detections_by_class: dict
  # track_id -> class_id
  class_ids_by_track: dict
  # track_id -> (x_center, y_center, width, height) normalized 0-1
  boxes_by_track: dict
  # (class_id, x, y, w, h, conf) tuples for every detection, tracked or not
  detections: list
  stalled_ids: set = field(default_factory=set)

  @property
  def active_vehicle_detected(self):
    """True if any non-stalled vehicle is present."""
    return len(self.active_detections_by_class) > 0


class FramePipeline:
  """
  Runs the detection logic for a stream of frames.

  Holds tracker and stall state between frames, so use one pipeline per
  video stream.
  """

  def __init__(self, detector, fps,
               stall_seconds=STALL_SECONDS,
               movement_threshold=STALL_MOVEMENT_THRESHOLD):
    """
    Args:
      detector: VehicleDetector instance (its tracker state persists across frames)
      fps: Frames per second of the source, used for stall timing
      stall_seconds: Seconds of no movement before a vehicle is stalled
      movement_threshold: Normalized distance below which movement is ignored
    """
    self.detector = detector
    self.stall_tracker = StallTracker(stall_seconds, movement_threshold, fps)
    self.frame_index = 0

  def process(self, frame):
    """
    Process one full-resolution frame.

    Args:
      frame: Raw frame (BGR format from OpenCV), before cropping

    Returns:
      FrameResult for this frame
    """
    cropped_frame = frame[CROP_Y1:CROP_Y2, CROP_X1:CROP_X2]

    results = self.detector.detect(cropped_frame)
    annotated_frame = self.detector.get_annotated_frame(results)
    _, detections_by_class, class_ids_by_track, boxes_by_track = self.detector.get_detections(results)
    detections = self.detector.get_detection_boxes(results)

    # Filter out stalled (parked/stationary) vehicles
    stalled_ids = self.stall_tracker.update(boxes_by_track)
    active_detections_by_class = {}
    for class_name, track_ids in detections_by_class.items():
      active_ids = [tid for tid in track_ids if tid not in stalled_ids]
      if active_ids:
        active_detections_by_class[class_name] = active_ids

    result = FrameResult(
      frame_index=self.frame_index,
      cropped_frame=cropped_frame,
      annotated_frame=annotated_frame,
      detections_by_class=detections_by_class,
      active_detections_by_class=active_detections_by_class,
      class_ids_by_track=class_ids_by_track,
      boxes_by_track=boxes_by_track,
      detections=detections,
      stalled_ids=stalled_ids,
    )
    self.frame_index += 1
    return result

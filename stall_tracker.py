"""
Stall tracker - detects vehicles that have stopped moving.

Monitors each tracked vehicle's position across frames and marks
vehicles as "stalled" when they haven't moved beyond a small
threshold for a configurable duration.
"""

import math


class StallTracker:
  def __init__(self, stall_seconds, movement_threshold, fps):
    """
    Args:
      stall_seconds: Seconds of no movement before a vehicle is stalled.
      movement_threshold: Normalized distance below which movement is ignored.
      fps: Frames per second, used to convert stall_seconds to frame count.
    """
    self.movement_threshold = movement_threshold
    self.stall_frames = int(stall_seconds * fps)
    # Per-track state: {track_id: (ref_x, ref_y, still_frames, stalled)}
    self._tracks = {}

  def update(self, boxes_by_track):
    """Update tracker with current frame's bounding boxes.

    Args:
      boxes_by_track: Dict mapping track_id -> (x1, y1, x2, y2) normalized.

    Returns:
      Set of track IDs that are currently stalled.
    """
    stalled = set()
    current_ids = set(boxes_by_track.keys())

    for track_id, box in boxes_by_track.items():
      cx = (box[0] + box[2]) / 2
      cy = (box[1] + box[3]) / 2

      if track_id not in self._tracks:
        # New track
        self._tracks[track_id] = (cx, cy, 0, False)
      else:
        ref_x, ref_y, still_count, was_stalled = self._tracks[track_id]
        dist = math.hypot(cx - ref_x, cy - ref_y)

        if dist > self.movement_threshold:
          # Moved - reset
          self._tracks[track_id] = (cx, cy, 0, False)
        else:
          # Still stationary
          still_count += 1
          is_stalled = still_count >= self.stall_frames
          self._tracks[track_id] = (ref_x, ref_y, still_count, is_stalled)

    # Collect stalled IDs
    for track_id in current_ids:
      if self._tracks[track_id][3]:
        stalled.add(track_id)

    # Remove tracks no longer present
    gone = set(self._tracks.keys()) - current_ids
    for track_id in gone:
      del self._tracks[track_id]

    return stalled

"""
Stall tracker - detects vehicles that have stopped moving.

Monitors each tracked vehicle's position across frames and marks
vehicles as "stalled" when they haven't moved beyond a small
threshold for a configurable duration.

A track that drops out for a few frames (detection flicker) keeps its
state, so a parked vehicle stays stalled when it reappears.
"""

import math


class StallTracker:
  def __init__(self, stall_seconds, movement_threshold, fps, forget_frames=30):
    """
    Args:
      stall_seconds: Seconds of no movement before a vehicle is stalled.
      movement_threshold: Normalized distance below which movement is ignored.
      fps: Frames per second, used to convert stall_seconds to frame count.
      forget_frames: Frames a track may go undetected before its state is
        dropped. Defaults to 30, matching how long the ByteTrack tracker
        keeps a lost track's ID.
    """
    self.movement_threshold = movement_threshold
    self.stall_frames = int(stall_seconds * fps)
    self.forget_frames = forget_frames
    self._frame = 0
    # Per-track state: {track_id: (ref_x, ref_y, still_since_frame, last_seen_frame)}
    self._tracks = {}

  def update(self, boxes_by_track):
    """Update tracker with current frame's bounding boxes.

    Args:
      boxes_by_track: Dict mapping track_id -> (x_center, y_center, width, height) normalized 0-1.

    Returns:
      Set of track IDs that are currently stalled.
    """
    frame = self._frame
    self._frame += 1
    stalled = set()

    for track_id, box in boxes_by_track.items():
      cx, cy = box[0], box[1]

      if track_id not in self._tracks:
        # New track
        self._tracks[track_id] = (cx, cy, frame, frame)
        continue

      ref_x, ref_y, still_since, _ = self._tracks[track_id]
      if math.hypot(cx - ref_x, cy - ref_y) > self.movement_threshold:
        # Moved - restart the stillness clock from the new position
        self._tracks[track_id] = (cx, cy, frame, frame)
        continue

      # Still near the reference position. Frames where the track went
      # undetected count toward stillness: no detection is not movement.
      self._tracks[track_id] = (ref_x, ref_y, still_since, frame)
      if frame - still_since >= self.stall_frames:
        stalled.add(track_id)

    # Drop tracks that have been missing too long
    for track_id in [tid for tid, state in self._tracks.items()
                     if frame - state[3] > self.forget_frames]:
      del self._tracks[track_id]

    return stalled

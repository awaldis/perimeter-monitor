"""Unit tests for StallTracker (no video or model needed)."""

from permon.monitor.stall_tracker import StallTracker

# 1 fps and 5 stall seconds -> stalled after 5 frames without movement
STALL_FRAMES = 5
BOX = (0.5, 0.5, 0.1, 0.2)  # (x_center, y_center, width, height)


def make_tracker():
  return StallTracker(stall_seconds=STALL_FRAMES, movement_threshold=0.01, fps=1, forget_frames=3)


def feed(tracker, frames):
  """Feed a list of boxes_by_track dicts; return the stalled set for each frame."""
  return [tracker.update(boxes) for boxes in frames]


def test_stationary_vehicle_stalls_after_stall_frames():
  stalled = feed(make_tracker(), [{1: BOX}] * (STALL_FRAMES + 1))
  assert all(s == set() for s in stalled[:STALL_FRAMES])
  assert stalled[STALL_FRAMES] == {1}


def test_box_size_change_is_not_movement():
  # Boxes are (x_center, y_center, w, h): a parked car whose box grows and
  # shrinks must still stall.
  frames = [{1: (0.5, 0.5, 0.1 + 0.05 * (i % 2), 0.2)} for i in range(STALL_FRAMES + 1)]
  assert feed(make_tracker(), frames)[-1] == {1}


def test_center_movement_resets_stall():
  frames = [{1: BOX}] * STALL_FRAMES + [{1: (0.6, 0.5, 0.1, 0.2)}]
  assert feed(make_tracker(), frames)[-1] == set()


def test_short_detection_gap_keeps_stall_state():
  # Detection flickers off for a couple of frames: the vehicle stays stalled.
  frames = [{1: BOX}] * (STALL_FRAMES + 1) + [{}] * 2 + [{1: BOX}]
  assert feed(make_tracker(), frames)[-1] == {1}


def test_gap_frames_count_toward_stillness():
  # Seen, missing, seen again in the same place: elapsed frames count.
  frames = [{1: BOX}] + [{}] * 3 + [{1: BOX}] * 2
  assert feed(make_tracker(), frames)[-1] == {1}


def test_long_gap_forgets_track():
  # Missing longer than forget_frames: the track starts over when it returns.
  frames = [{1: BOX}] * (STALL_FRAMES + 1) + [{}] * 4 + [{1: BOX}]
  assert feed(make_tracker(), frames)[-1] == set()

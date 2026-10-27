"""
Video regression tests: run recorded clips through the live processing
pipeline and check the detected vehicle class frame by frame, and how many
clips the live recorder would save.

Cases are listed in video_cases.yaml. Run with:
  pytest -v
"""

from collections import Counter
from pathlib import Path

import cv2
import pytest
import yaml

from permon.config import MODEL_PATH, CONFIDENCE_THRESHOLD, DEFAULT_FPS, CROP_X1, CROP_X2, CROP_Y1, CROP_Y2
from permon.monitor.cli import record_and_alert
from permon.monitor.clip_recorder import ClipRecorder
from permon.monitor.frame_pipeline import FramePipeline
from permon.monitor.vehicle_detector import VehicleDetector
from permon.monitor.video_reader import VideoStreamReader

REPO_ROOT = Path(__file__).resolve().parent.parent
CASES_FILE = Path(__file__).resolve().parent / "video_cases.yaml"


def load_cases():
  """Load test cases from video_cases.yaml as pytest params."""
  with open(CASES_FILE) as f:
    spec = yaml.safe_load(f)
  video_dir = REPO_ROOT / spec["video_dir"]
  return [
    pytest.param(case, video_dir / case["file"], id=Path(case["file"]).stem)
    for case in spec["cases"]
  ]


def run_video(video_path, clips_dir):
  """
  Run a video through a fresh pipeline and clip recorder, the same way
  the monitor does in live mode (Telegram alerts excluded).

  Returns:
    Tuple of (list of FrameResult, one per frame; ClipRecorder; fps)
  """
  # Fresh detector per video: the YOLO tracker keeps state between calls
  detector = VehicleDetector(confidence=CONFIDENCE_THRESHOLD)

  reader = VideoStreamReader(str(video_path))
  fps = reader.get(cv2.CAP_PROP_FPS) or DEFAULT_FPS
  pipeline = FramePipeline(detector, fps)
  recorder = ClipRecorder(str(clips_dir), fps, CROP_X2 - CROP_X1, CROP_Y2 - CROP_Y1)
  reader.start()

  results = []
  try:
    while True:
      success, frame = reader.read()
      if not success:
        break
      result = pipeline.process(frame)
      record_and_alert(result, recorder)
      # Drop the images so a long video doesn't hold every frame in memory
      result.cropped_frame = None
      result.annotated_frame = None
      results.append(result)
  finally:
    reader.stop()
    recorder.close()
  return results, recorder, fps


def describe(result):
  """One-line summary of what a frame detected, for failure messages."""
  active = result.active_detections_by_class or "nothing"
  line = f"frame {result.frame_index:5d}: active={active}"
  if result.stalled_ids:
    line += f" stalled_ids={sorted(result.stalled_ids)}"
  if not result.detections_by_class and result.detections:
    line += f" (untracked detections: {len(result.detections)})"
  return line


def check_frame_range(case, results):
  """Return failure text if expected_class isn't the only active class on every frame in range."""
  start, end = case["start_frame"], case["end_frame"]
  expected = case["expected_class"]
  if end >= len(results):
    return f"end_frame {end} is past the end of the video ({len(results)} frames)"

  failures = [
    r for r in results[start:end + 1]
    if set(r.active_detections_by_class) != {expected}
  ]
  if not failures:
    return None
  return (
    f"expected only '{expected}' on frames {start}-{end}; "
    f"{len(failures)} of {end - start + 1} frames failed:\n"
    + "\n".join(describe(r) for r in failures)
  )


def check_clip_count(case, results, recorder):
  """Return failure text if the recorder didn't save exactly the expected number of clips."""
  expected = case["clips"]
  actual = recorder.get_clip_count()
  if actual == expected:
    return None

  # Show the frames where a vehicle became active after none were
  starts = [
    describe(r) for prev, r in zip([None] + results, results)
    if r.active_vehicle_detected and (prev is None or not prev.active_vehicle_detected)
  ]
  return (
    f"expected {expected} clip(s), recorder saved {actual}. "
    f"Frames where an active vehicle appeared:\n" + "\n".join(starts)
  )


def check_clip_lengths(case, clips_dir, fps):
  """Return failure text if any saved clip is longer than max_clip_seconds."""
  limit = case["max_clip_seconds"]
  # The recorder saves one screenshot per clip frame, named {clip}_{NNNN}.jpg
  frames_by_clip = Counter(
    jpg.stem.rsplit("_", 1)[0] for jpg in Path(clips_dir).glob("*/images/*.jpg")
  )
  too_long = {
    clip: frames / fps for clip, frames in frames_by_clip.items()
    if frames / fps > limit
  }
  if not too_long:
    return None
  return f"clips longer than {limit} s: " + ", ".join(
    f"{clip} ({seconds:.1f} s)" for clip, seconds in sorted(too_long.items())
  )


@pytest.mark.parametrize("case, video_path", load_cases())
def test_video_case(case, video_path, tmp_path):
  if not video_path.exists():
    pytest.skip(f"video not found: {video_path}")

  print(f"\nmodel={MODEL_PATH} conf={CONFIDENCE_THRESHOLD} video={video_path.name}")
  results, recorder, fps = run_video(video_path, tmp_path)

  problems = []
  if "expected_class" in case:
    problems.append(check_frame_range(case, results))
  if "clips" in case:
    problems.append(check_clip_count(case, results, recorder))
  if "max_clip_seconds" in case:
    problems.append(check_clip_lengths(case, tmp_path, fps))
  problems = [p for p in problems if p]

  if problems:
    pytest.fail("\n\n".join(problems), pytrace=False)

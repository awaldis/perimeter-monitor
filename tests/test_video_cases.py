"""
Video regression tests: run recorded clips through the live processing
pipeline and check the detected vehicle class frame by frame.

Cases are listed in video_cases.yaml. Run with:
  pytest -v
"""

from pathlib import Path

import cv2
import pytest
import yaml

from config import MODEL_PATH, CONFIDENCE_THRESHOLD, DEFAULT_FPS
from frame_pipeline import FramePipeline
from vehicle_detector import VehicleDetector
from video_reader import VideoStreamReader

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


def run_video(video_path):
  """
  Run a video through a fresh pipeline, the same way mon_perim.py does.

  Returns:
    List of FrameResult, one per frame
  """
  # Fresh detector per video: the YOLO tracker keeps state between calls
  detector = VehicleDetector(confidence=CONFIDENCE_THRESHOLD)

  reader = VideoStreamReader(str(video_path))
  fps = reader.get(cv2.CAP_PROP_FPS) or DEFAULT_FPS
  pipeline = FramePipeline(detector, fps)
  reader.start()

  results = []
  try:
    while True:
      success, frame = reader.read()
      if not success:
        break
      result = pipeline.process(frame)
      # Drop the images so a long video doesn't hold every frame in memory
      result.cropped_frame = None
      result.annotated_frame = None
      results.append(result)
  finally:
    reader.stop()
  return results


def describe(result):
  """One-line summary of what a frame detected, for failure messages."""
  active = result.active_detections_by_class or "nothing"
  line = f"frame {result.frame_index:5d}: active={active}"
  if result.stalled_ids:
    line += f" stalled_ids={sorted(result.stalled_ids)}"
  if not result.detections_by_class and result.detections:
    line += f" (untracked detections: {len(result.detections)})"
  return line


@pytest.mark.parametrize("case, video_path", load_cases())
def test_video_case(case, video_path):
  if not video_path.exists():
    pytest.skip(f"video not found: {video_path}")

  start, end = case["start_frame"], case["end_frame"]
  expected = case["expected_class"]
  print(f"\nmodel={MODEL_PATH} conf={CONFIDENCE_THRESHOLD} video={video_path.name}")

  results = run_video(video_path)
  assert end < len(results), (
    f"end_frame {end} is past the end of the video ({len(results)} frames)"
  )

  failures = [
    r for r in results[start:end + 1]
    if set(r.active_detections_by_class) != {expected}
  ]
  if failures:
    checked = end - start + 1
    pytest.fail(
      f"expected only '{expected}' on frames {start}-{end}; "
      f"{len(failures)} of {checked} frames failed:\n"
      + "\n".join(describe(r) for r in failures),
      pytrace=False
    )

#!/usr/bin/env python3
"""
Time each pipeline stage by itself on this machine.

The monitor overlaps decoding (reader thread) with detection (main thread),
so its stage times include the two fighting over the CPU. This tool runs the
same stage code strictly one stage after another in a single thread, so each
number is what that step costs alone:

  wall ms   time the stage took per frame
  CPU ms    processor time it used per frame, summed over all cores
  cores     CPU / wall: how many cores the stage kept busy
  alone     frames per second if this were the only stage

Use it to compare hardware and settings, for example:
  python tools/benchmark_stages.py input_4K/video.mp4
  python tools/benchmark_stages.py input_4K/video.mp4 --device GPU
  python tools/benchmark_stages.py input_4K/video.mp4 --hwaccel qsv
  python tools/benchmark_stages.py input_4K/video.mp4 --decode-threads 0

Usage (from the repo root):
  python tools/benchmark_stages.py <video file> [options]
"""

import argparse
import statistics
import sys
import tempfile
import time
from pathlib import Path

# Make the permon package importable when run as a script
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from permon.config import (
  CONFIDENCE_THRESHOLD,
  CROP_X1, CROP_Y1, CROP_X2, CROP_Y2,
  DECODE_HWACCEL,
  DECODE_THREADS,
  DEFAULT_FPS,
  INFERENCE_DEVICE,
  MODEL_PATH
)
from permon.monitor.clip_recorder import ClipRecorder
from permon.monitor.vehicle_detector import VehicleDetector
from permon.monitor.video_source import PyAVSource

WARMUP_FRAMES = 5  # Not counted: first calls include one-time setup

# (stage, what goes in -> what comes out)
STAGES = [
  ("receive",    "file -> compressed packet"),
  ("decode",     "packet -> full YUV frame"),
  ("crop",       "full frame -> YUV detection region"),
  ("convert",    "YUV region -> BGR image"),
  ("preprocess", "BGR image -> RGB float tensor"),
  ("inference",  "tensor -> raw model output"),
  ("nms",        "raw output -> detection boxes"),
  ("track",      "boxes -> boxes with track IDs"),
  ("annotate",   "draw boxes on the image"),
  ("record",     "write clip frame, JPEG and label"),
]


class Measurements:
  """Wall and CPU time per stage."""

  def __init__(self):
    self.wall = {name: [] for name, _ in STAGES}
    self.cpu = {name: [] for name, _ in STAGES}
    self.enabled = False

  def run(self, name, fn, *args):
    """Call fn(*args) as one run of stage `name` and return its result."""
    cpu_start = time.process_time()
    wall_start = time.perf_counter()
    result = fn(*args)
    wall = time.perf_counter() - wall_start
    cpu = time.process_time() - cpu_start
    if self.enabled:
      self.wall[name].append(wall)
      self.cpu[name].append(cpu)
    return result


def parse_args():
  parser = argparse.ArgumentParser(description="Time each pipeline stage by itself")
  parser.add_argument("video", help="Video file to run through the stages")
  parser.add_argument("-n", "--frames", type=int, default=150,
                      help="Frames to measure (default: 150)")
  parser.add_argument("--device", default=INFERENCE_DEVICE,
                      help=f"Inference device (default: INFERENCE_DEVICE, {INFERENCE_DEVICE})")
  parser.add_argument("--decode-threads", type=int, default=DECODE_THREADS,
                      help=f"FFmpeg decoder threads, 0 = every core (default: DECODE_THREADS, {DECODE_THREADS})")
  parser.add_argument("--hwaccel", default=DECODE_HWACCEL,
                      help=f"Hardware decoder, e.g. qsv or cuda (default: DECODE_HWACCEL, {DECODE_HWACCEL})")
  parser.add_argument("--model", default=MODEL_PATH,
                      help=f"Model file or folder (default: MODEL_PATH, {MODEL_PATH})")
  return parser.parse_args()


def main():
  args = parse_args()

  detector = VehicleDetector(model_path=args.model, confidence=CONFIDENCE_THRESHOLD, device=args.device)
  crop = (CROP_X1, CROP_Y1, CROP_X2, CROP_Y2)
  source = PyAVSource(args.video, crop, threads=args.decode_threads, hwaccel=args.hwaccel)
  if not source.open():
    return 1
  print(f"Video decoding: {source.describe()}")
  fps = source.fps or DEFAULT_FPS

  m = Measurements()
  frames_done = 0
  pixel_format = None
  with tempfile.TemporaryDirectory() as clips_dir:
    recorder = ClipRecorder(clips_dir, fps, CROP_X2 - CROP_X1, CROP_Y2 - CROP_Y1)
    print(f"Measuring {args.frames} frames after {WARMUP_FRAMES} warm-up frames...")

    while frames_done < args.frames + WARMUP_FRAMES:
      packet = m.run("receive", source.receive)
      if packet is None:
        break
      for frame in m.run("decode", source.decode, packet):
        pixel_format = frame.format.name
        region, fmt = m.run("crop", source.crop, frame)
        image = m.run("convert", source.convert, region, fmt)
        tensor = m.run("preprocess", detector.preprocess, image)
        raw = m.run("inference", detector.infer, tensor)
        results = m.run("nms", detector.nms, raw, image)
        results = m.run("track", detector.track, results)
        annotated = m.run("annotate", detector.annotate, results)
        # Worst case for recording: treat every frame as part of a clip
        m.run("record", recorder.on_vehicle_detected, annotated, image,
              detector.get_detection_boxes(results))
        frames_done += 1
        m.enabled = frames_done >= WARMUP_FRAMES
    recorder.close()
  source.close()

  measured = len(m.wall["inference"])
  if measured == 0:
    print("Not enough frames to measure.")
    return 1

  budget_ms = 1000.0 / fps
  print(f"\nStage costs, each run alone ({measured} frames, decoder output {pixel_format}, "
        f"{source.width}x{source.height} -> crop {crop[2] - crop[0]}x{crop[3] - crop[1]})")
  header = (f"{'Stage':<11} {'In -> out':<36} {'wall ms':>8} {'median':>8} {'CPU ms':>8} "
            f"{'cores':>6} {'alone':>9}")
  print(header)
  print("-" * len(header))
  total_wall = total_cpu = 0.0
  for name, what in STAGES:
    if not m.wall[name]:
      continue
    # Per frame: a stage may run more or less than once per frame (decode)
    wall = sum(m.wall[name]) / measured * 1000
    cpu = sum(m.cpu[name]) / measured * 1000
    median = statistics.median(m.wall[name]) * 1000
    total_wall += wall
    total_cpu += cpu
    cores = cpu / wall if wall > 0.05 else 0
    print(f"{name:<11} {what:<36} {wall:>8.2f} {median:>8.2f} {cpu:>8.2f} {cores:>6.1f} "
          f"{1000 / wall if wall > 0 else 0:>7.0f}/s")
  print("-" * len(header))
  print(f"{'TOTAL':<11} {'':<36} {total_wall:>8.2f} {'':>8} {total_cpu:>8.2f} "
        f"{total_cpu / total_wall:>6.1f} {1000 / total_wall:>7.1f}/s")

  print(f"\nSource frame rate: {fps:.2f} fps, so {budget_ms:.0f} ms is available per frame.")
  print(f"One frame needs {total_wall:.0f} ms of wall time when the stages run back to back "
        f"({total_wall / budget_ms * 100:.0f}% of that) and {total_cpu:.0f} ms of CPU time.")
  print(f"Keeping up with {fps:.2f} fps takes {total_cpu * fps / 1000:.2f} cores' worth of CPU on average.")
  print("'record' is the cost while a vehicle is in view; other frames skip it.")
  if args.decode_threads != 1:
    print("Note: with more than one decoder thread, decoding runs ahead in the background, so "
          "part of its cost is charged to whichever stage was running. Use --decode-threads 1 "
          "for exact numbers.")
  return 0


if __name__ == "__main__":
  sys.exit(main())

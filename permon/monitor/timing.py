"""
Per-stage timing for the processing pipeline.

Every step a frame goes through, from the network to the saved clip, is a
named stage. The stages are listed in STAGES in the order a frame meets
them; each one records how long it took, and report() prints the table.

Times are wall-clock. Two things to keep in mind when reading them:
  - The reader-thread stages run at the same time as the main-thread
    stages, so the two groups are not added together.
  - A stage that hands work to other threads (multi-threaded decoding,
    inference on all cores) is charged only for the time the caller waited.
    The "Process CPU" line shows what the whole process really used.
tools/benchmark_stages.py runs each stage by itself for clean numbers.
"""

import statistics
import time
from collections import deque
from contextlib import contextmanager

# (name, thread, what the stage does), in pipeline order
STAGES = [
  ("receive",        "reader", "network/file -> compressed packet (live: includes waiting for the camera)"),
  ("decode",         "reader", "compressed packet -> full-size YUV frame"),
  ("decode+convert", "reader", "OpenCV decoder: packet -> full-size BGR frame"),
  ("crop",           "reader", "full frame -> detection region"),
  ("convert",        "reader", "YUV region -> BGR image"),
  ("wait for frame", "main",   "main thread idle until the reader has a frame"),
  ("preprocess",     "main",   "BGR image -> RGB float tensor the model takes"),
  ("inference",      "main",   "tensor -> raw model output (the YOLO network)"),
  ("nms",            "main",   "raw output -> detection boxes"),
  ("track",          "main",   "detection boxes -> boxes with track IDs"),
  ("stall filter",   "main",   "drop parked vehicles"),
  ("annotate",       "main",   "draw boxes on a copy of the image"),
  ("record",         "main",   "clip/screenshot/label writing and alerts"),
  ("display",        "main",   "live window"),
]
_ORDER = {name: i for i, (name, _, _) in enumerate(STAGES)}
_THREAD = {name: thread for name, thread, _ in STAGES}


class _Stat:
  """Running totals for one stage, plus recent samples for median/p95."""

  def __init__(self, window):
    self.count = 0
    self.total = 0.0
    self.max = 0.0
    self.recent = deque(maxlen=window)

  def add(self, seconds):
    self.count += 1
    self.total += seconds
    if seconds > self.max:
      self.max = seconds
    self.recent.append(seconds)


class StageTimer:
  """
  Collects stage durations from the reader and main threads.

  Memory use is fixed: totals are kept for the whole run, individual samples
  only for the last `window` calls of each stage.
  """

  def __init__(self, window=1000):
    self.window = window
    self._stats = {}
    self._wall_start = time.perf_counter()
    self._cpu_start = time.process_time()

  @contextmanager
  def stage(self, name):
    """Time the enclosed block as one call of stage `name`."""
    start = time.perf_counter()
    try:
      yield
    finally:
      self.add(name, time.perf_counter() - start)

  def add(self, name, seconds):
    """Record one call of stage `name` that took `seconds`."""
    stat = self._stats.get(name)
    if stat is None:
      stat = self._stats.setdefault(name, _Stat(self.window))
    stat.add(seconds)

  def mean_ms(self, name):
    """Mean duration of a stage in ms over the whole run (0 if it never ran)."""
    stat = self._stats.get(name)
    return stat.total / stat.count * 1000 if stat and stat.count else 0.0

  def report(self, fps=None):
    """
    Build the timing table.

    Args:
      fps: Source frame rate. If given, each stage is also shown as a share
        of the time available per frame (1/fps).

    Returns:
      The table as a string
    """
    budget_ms = 1000.0 / fps if fps else None
    lines = ["", "Pipeline stage timing (wall-clock ms per call; median and p95 over the "
             f"last {self.window} calls)"]
    header = f"{'Stage':<16} {'Thread':<7} {'Median':>8} {'Mean':>8} {'p95':>8} {'Max':>8} {'Calls':>8}"
    if budget_ms:
      header += f" {'Budget':>7}"
    lines += [header, "-" * len(header)]

    totals = {"reader": 0.0, "main": 0.0}
    names = sorted(self._stats, key=lambda n: _ORDER.get(n, len(_ORDER)))
    for name in names:
      stat = self._stats[name]
      if not stat.count:
        continue
      thread = _THREAD.get(name, "main")
      recent = sorted(stat.recent)
      mean_ms = stat.total / stat.count * 1000
      p95_ms = recent[min(len(recent) - 1, int(len(recent) * 0.95))] * 1000
      line = (f"{name:<16} {thread:<7} {statistics.median(recent) * 1000:>8.2f} {mean_ms:>8.2f} "
              f"{p95_ms:>8.2f} {stat.max * 1000:>8.2f} {stat.count:>8}")
      # Waiting is idle time, not work, so it gets no budget share
      is_wait = name in ("receive", "wait for frame")
      if budget_ms and not is_wait:
        line += f" {mean_ms / budget_ms * 100:>6.1f}%"
      if not is_wait:
        totals[thread] += mean_ms
      lines.append(line)

    lines.append("-" * len(header))
    for thread in ("reader", "main"):
      label = "receive" if thread == "reader" else "wait for frame"
      line = f"{thread + ' thread work':<24} {'':>8} {totals[thread]:>8.2f}  (mean, excluding '{label}')"
      if budget_ms:
        line += f"  = {totals[thread] / budget_ms * 100:.0f}% of budget"
      lines.append(line)
    if budget_ms:
      line = f"Frame budget at {fps:.2f} fps: {budget_ms:.1f} ms."
      # The slower thread sets the pace
      slowest = max(totals.values())
      if slowest > 0:
        line += f" Fastest sustainable rate: {1000 / slowest:.1f} fps."
      lines.append(line)

    wall = time.perf_counter() - self._wall_start
    cpu = time.process_time() - self._cpu_start
    if wall > 0:
      lines.append(f"Process CPU: {cpu:.1f} s over {wall:.1f} s = {cpu / wall:.2f} cores busy on average "
                   "(all threads, including decoder and inference workers)")
    return "\n".join(lines)

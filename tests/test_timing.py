"""Unit tests for StageTimer."""

import pytest

from permon.monitor.timing import StageTimer


def test_stage_context_records_a_call():
  timer = StageTimer()
  with timer.stage("inference"):
    pass
  assert timer.mean_ms("inference") >= 0
  assert "inference" in timer.report()


def test_report_lists_stages_in_pipeline_order_with_budget_share():
  timer = StageTimer()
  # Recorded out of order, as the reader and main threads would
  timer.add("inference", 0.050)
  timer.add("decode", 0.020)
  timer.add("wait for frame", 0.500)
  timer.add("receive", 0.090)
  report = timer.report(fps=10)  # 100 ms per frame

  assert report.index("receive") < report.index("decode") < report.index("inference")
  decode_line = next(line for line in report.splitlines() if line.startswith("decode"))
  assert decode_line.rstrip().endswith("20.0%")
  # Waiting is idle time: no budget share, and left out of the totals
  wait_line = next(line for line in report.splitlines() if line.startswith("wait for frame"))
  assert "%" not in wait_line
  assert "= 20% of budget" in report   # reader thread: decode only
  assert "= 50% of budget" in report   # main thread: inference only
  assert "Fastest sustainable rate: 20.0 fps" in report


def test_mean_covers_whole_run_but_samples_are_bounded():
  timer = StageTimer(window=10)
  for _ in range(100):
    timer.add("crop", 0.001)
  timer.add("crop", 0.102)
  assert timer.mean_ms("crop") == pytest.approx(2.0)
  assert len(timer._stats["crop"].recent) == 10


def test_unknown_stage_is_reported():
  timer = StageTimer()
  timer.add("my new stage", 0.001)
  assert "my new stage" in timer.report()

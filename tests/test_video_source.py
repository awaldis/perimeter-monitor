"""
Unit tests for the video sources and reader (no camera video or model needed).

A small H.264 test video is generated on the fly, so these run anywhere.
"""

import av
import cv2
import numpy as np
import pytest

from permon.monitor import video_reader
from permon.monitor.timing import StageTimer
from permon.monitor.video_reader import VideoStreamReader
from permon.monitor.video_source import OpenCVSource, PyAVSource, create_video_source

WIDTH, HEIGHT, FRAMES, FPS = 320, 240, 12, 10
CROP = (40, 20, 240, 180)       # x1, y1, x2, y2; all even
ODD_CROP = (41, 21, 241, 181)


@pytest.fixture(scope="module")
def video_path(tmp_path_factory):
  """A short video with a gradient background and a box that moves each frame."""
  path = tmp_path_factory.mktemp("video") / "moving_box.mp4"
  container = av.open(str(path), "w")
  stream = container.add_stream("libx264", rate=FPS)
  stream.width, stream.height, stream.pix_fmt = WIDTH, HEIGHT, "yuv420p"
  for i in range(FRAMES):
    image = np.zeros((HEIGHT, WIDTH, 3), np.uint8)
    image[:, :, 0] = np.arange(WIDTH)[None, :] * 255 // WIDTH
    image[:, :, 1] = (np.arange(HEIGHT)[:, None] + i * 10) % 256
    cv2.rectangle(image, (20 + i * 15, 60), (80 + i * 15, 140), (0, 0, 255), -1)
    for packet in stream.encode(av.VideoFrame.from_ndarray(image, format="bgr24")):
      container.mux(packet)
  for packet in stream.encode():
    container.mux(packet)
  container.close()
  return str(path)


def read_all(source):
  """Open a source and return every frame it produces."""
  assert source.open()
  frames = []
  while (frame := source.read()) is not None:
    frames.append(frame)
  source.close()
  return frames


def assert_same_frames(actual, expected):
  assert len(actual) == len(expected) == FRAMES
  for a, e in zip(actual, expected):
    assert np.array_equal(a, e)


def test_pyav_matches_opencv(video_path):
  """Cropping the YUV planes before converting gives OpenCV's exact pixels."""
  frames = read_all(PyAVSource(video_path, CROP))
  assert frames[0].shape == (CROP[3] - CROP[1], CROP[2] - CROP[0], 3)
  assert frames[0].dtype == np.uint8
  assert_same_frames(frames, read_all(OpenCVSource(video_path, CROP)))


def test_frames_differ_from_each_other(video_path):
  """Guard against a source that returns the same frame every time."""
  frames = read_all(PyAVSource(video_path, CROP))
  assert not np.array_equal(frames[0], frames[-1])


def test_multithreaded_decode_gives_same_frames(video_path):
  """A threaded decoder delays its output; no frames may be lost at the end."""
  assert_same_frames(read_all(PyAVSource(video_path, CROP, threads=0)),
                     read_all(PyAVSource(video_path, CROP, threads=1)))


def test_odd_crop_falls_back_to_whole_frame_conversion(video_path):
  """A region on odd pixels can't be cut from half-size chroma planes."""
  assert_same_frames(read_all(PyAVSource(video_path, ODD_CROP)),
                     read_all(OpenCVSource(video_path, ODD_CROP)))


def test_nv12_frames_give_same_pixels(video_path):
  """nv12 (what hardware decoders output) must convert like software yuv420p."""
  source = PyAVSource(video_path, CROP)
  assert source.open()
  compared = 0
  while (packet := source.receive()) is not None:
    for frame in source.decode(packet):
      expected = source.convert(*source.crop(frame))
      region, fmt = source.crop(frame.reformat(format="nv12"))
      assert fmt == "yuv420p"
      assert np.array_equal(source.convert(region, fmt), expected)
      compared += 1
  source.close()
  assert compared == FRAMES


def test_stream_facts(video_path):
  for source in (PyAVSource(video_path, CROP), OpenCVSource(video_path, CROP)):
    assert source.open()
    assert (source.width, source.height) == (WIDTH, HEIGHT)
    assert source.fps == pytest.approx(FPS)
    assert source.frame_count == FRAMES
    source.close()
    # Still readable after close (the monitor reports them at exit)
    assert source.fps == pytest.approx(FPS)
    assert not source.is_open


def test_missing_file_fails_to_open(tmp_path):
  assert not PyAVSource(str(tmp_path / "missing.mp4"), CROP).open()
  assert not OpenCVSource(str(tmp_path / "missing.mp4"), CROP).open()


def test_open_failure_does_not_log_stream_credentials(capsys):
  url = "rtsp://someuser:secretpw@127.0.0.1:9/stream"
  assert not PyAVSource(url, CROP, is_live=True).open()
  output = capsys.readouterr().out
  assert "Could not open video source" in output
  assert "secretpw" not in output and "someuser" not in output


def test_create_video_source_selects_decoder(video_path):
  assert isinstance(create_video_source(video_path, CROP, decoder="pyav"), PyAVSource)
  assert isinstance(create_video_source(video_path, CROP, decoder="opencv"), OpenCVSource)
  with pytest.raises(ValueError):
    create_video_source(video_path, CROP, decoder="gstreamer")


def test_stage_times_are_recorded(video_path):
  timer = StageTimer()
  read_all(PyAVSource(video_path, CROP, timer=timer))
  report = timer.report(fps=FPS)
  for stage in ("receive", "decode", "crop", "convert"):
    assert stage in report
  assert f"Frame budget at {FPS:.2f} fps: 100.0 ms" in report


def test_reader_delivers_every_frame_in_order(video_path, monkeypatch):
  monkeypatch.setattr(video_reader, "CROP_X1", CROP[0])
  monkeypatch.setattr(video_reader, "CROP_Y1", CROP[1])
  monkeypatch.setattr(video_reader, "CROP_X2", CROP[2])
  monkeypatch.setattr(video_reader, "CROP_Y2", CROP[3])
  reader = VideoStreamReader(video_path)
  assert reader.isOpened()
  assert (reader.width, reader.height, reader.frame_count) == (WIDTH, HEIGHT, FRAMES)
  reader.start()
  frames = []
  while True:
    success, frame = reader.read()
    if not success:
      break
    frames.append(frame)
  reader.stop()
  assert_same_frames(frames, read_all(PyAVSource(video_path, CROP)))


class FlakySource:
  """Live source that fails after two frames, refuses one reopen, then recovers."""

  width, height, fps, frame_count, is_open = WIDTH, HEIGHT, FPS, 0, True

  def __init__(self):
    self.opens = 0
    self.reads_since_open = 0

  def open(self):
    self.opens += 1
    self.reads_since_open = 0
    return self.opens != 2  # The first reconnect attempt fails

  def read(self):
    self.reads_since_open += 1
    if self.opens == 1 and self.reads_since_open > 2:
      return None  # The stream drops
    return np.full((2, 2, 3), self.opens, np.uint8)

  def close(self):
    pass


def test_reader_reconnects_after_stream_drop(monkeypatch):
  source = FlakySource()
  monkeypatch.setattr(video_reader, "create_video_source", lambda *args, **kwargs: source)
  monkeypatch.setattr(video_reader, "RECONNECT_DELAY_SECONDS", 0.01)

  reader = VideoStreamReader("rtsp://camera/stream", is_rtsp=True).start()
  frames = [reader.read() for _ in range(5)]
  reader.stop()

  assert all(success for success, _ in frames)
  # Two frames from the first connection, then frames from the third open()
  assert [int(frame[0, 0, 0]) for _, frame in frames] == [1, 1, 3, 3, 3]

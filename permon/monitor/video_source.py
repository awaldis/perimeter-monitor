"""
Video sources: the stages that turn a file or RTSP stream into the cropped
BGR image the detector works on.

PyAVSource (the default) calls FFmpeg's libraries directly through PyAV and
keeps every step separate:

  receive   network/file -> compressed packet (HEVC/H.264, a few kB)
  decode    packet       -> full-size frame, still in the camera's YUV 4:2:0
                            format (3840x2160: 12 MB)
  crop      full frame   -> detection region, still YUV (1600x480: 1.2 MB)
  convert   YUV region   -> BGR image, height x width x 3 uint8 (2.3 MB)

Cropping before converting matters: only the detection region is ever
converted to BGR, about 4% of a 4K frame.

OpenCVSource wraps cv2.VideoCapture, which uses the same FFmpeg decoder but
hides the steps: one call receives, decodes on all cores and converts the
whole frame to BGR. It is kept as a fallback (DECODER = "opencv").

Both produce the same pixels.
"""

from collections import deque
from datetime import datetime
import os
import re

import av
import cv2
import numpy as np
from av.codec.hwaccel import HWAccel
from av.video.reformatter import ColorRange

from permon.config import (
  DECODER,
  DECODE_THREADS,
  DECODE_HWACCEL,
  RTSP_TIMEOUT_SECONDS
)
from permon.monitor.timing import StageTimer


def mask_credentials(text):
  """Replace the user:password part of any URL in text: rtsp://***@host/..."""
  return re.sub(r"://[^/@\s]+@", "://***@", str(text))


def _log(message):
  """Print a timestamped source message, with any URL credentials removed."""
  timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
  # FFmpeg error text quotes the URL: keep rtsp://user:pass@host out of the log
  print(f"[{timestamp}] {mask_credentials(message)}")


class PyAVSource:
  """
  Video source built on PyAV (FFmpeg's libavformat/libavcodec/libswscale).

  read() runs the four stages for one frame. They are also public methods so
  tools/benchmark_stages.py can time each one by itself.
  """

  # Formats crop() can cut directly from the decoder's planes. Software
  # decoders give yuv420p (yuvj420p for full-range cameras); hardware
  # decoders usually give nv12.
  PLANAR_FORMATS = ("yuv420p", "yuvj420p")
  CROPPABLE_FORMATS = PLANAR_FORMATS + ("nv12",)

  # FFmpeg offers hardware decoding in two ways. Some device types plug into
  # the normal decoder (cuda): av.open() is told which device to use. Others
  # are separate decoders named <codec>_<type>, such as hevc_qsv for Intel
  # Quick Sync: those are created here and fed the packets directly.
  DEDICATED_DECODER_TYPES = ("qsv",)

  def __init__(self, src, crop, is_live=False, rtsp_transport='tcp', timer=None,
               threads=DECODE_THREADS, hwaccel=DECODE_HWACCEL):
    """
    Args:
      src: File path or RTSP URL
      crop: Detection region as (x1, y1, x2, y2) in full-frame pixels
      is_live: True for RTSP streams
      rtsp_transport: 'tcp' or 'udp' for RTSP streams
      timer: StageTimer that records the stage times
      threads: FFmpeg decoder threads; 1 decodes in the calling thread,
        0 lets FFmpeg use every core
      hwaccel: Hardware decoder type ('qsv', 'cuda', ...) or None for software
    """
    self.src = src
    self.crop_box = crop
    self.is_live = is_live
    self.rtsp_transport = rtsp_transport
    self.timer = timer or StageTimer()
    self.threads = threads
    self.hwaccel = hwaccel
    self.decode_errors = 0

    self.container = None
    self._packets = None
    self._frames = deque()
    self._warned_format = None

    # Stream facts, filled in by open()
    self.width = 0
    self.height = 0
    self.fps = 0.0          # Frame rate from the stream metadata (0 if unknown)
    self.frame_count = 0    # Total frames, for files (0 if unknown)

    # Hardware decoding state: _hw_decoder is a dedicated decoder (see
    # DEDICATED_DECODER_TYPES); hwaccel_active turns True once hardware
    # decoding is known to work
    self.hwaccel_active = False
    self._hw_decoder = None
    self._hw_pending = []   # Packets sent to an unconfirmed hardware decoder

    # Chroma planes are half size, so the region can only be cut straight
    # from the planes if its edges are on even pixels
    self._even_crop = all(v % 2 == 0 for v in crop)

  def open(self):
    """Open the source. Returns True on success."""
    options = {}
    timeout = None
    if self.is_live:
      options['rtsp_transport'] = self.rtsp_transport
      # (open, read): give up on a silent stream so the reader can reconnect
      timeout = (RTSP_TIMEOUT_SECONDS, RTSP_TIMEOUT_SECONDS)
    dedicated = self.hwaccel in self.DEDICATED_DECODER_TYPES
    hwaccel = None
    if self.hwaccel and not dedicated:
      hwaccel = HWAccel(device_type=self.hwaccel, allow_software_fallback=True)

    try:
      self.container = av.open(self.src, options=options, timeout=timeout, hwaccel=hwaccel)
      stream = self.container.streams.video[0]
    except (av.error.FFmpegError, OSError, IndexError) as e:
      _log(f"Could not open video source: {e}")
      self.container = None
      return False

    codec = stream.codec_context
    codec.thread_count = self.threads
    codec.thread_type = 'AUTO'
    self._packets = self.container.demux(stream)
    self._frames.clear()

    # Stream facts, kept so they can still be read after close()
    self.width = codec.width
    self.height = codec.height
    self.fps = float(stream.average_rate) if stream.average_rate else 0.0
    self.frame_count = stream.frames
    self._codec_description = f"{codec.name} {codec.pix_fmt}"

    self._hw_decoder = None
    self._hw_pending = []
    if dedicated:
      # Whether it works is only known once it decodes a frame
      self.hwaccel_active = False
      self._hw_decoder = self._create_dedicated_decoder(stream)
    else:
      # False if hardware decoding was requested but FFmpeg fell back to software
      self.hwaccel_active = bool(self.hwaccel) and bool(codec.is_hwaccel)
    return True

  def _dedicated_decoder_name(self, codec_name):
    """Name of FFmpeg's separate hardware decoder for a codec, e.g. hevc_qsv."""
    return f"{codec_name}_{self.hwaccel}"

  def _create_dedicated_decoder(self, stream):
    """Create the separate hardware decoder for this stream, or None if FFmpeg has none."""
    name = self._dedicated_decoder_name(stream.codec_context.name)
    try:
      decoder = av.CodecContext.create(name, "r")
      # Stream headers (parameter sets) the decoder needs before the first packet
      decoder.extradata = stream.codec_context.extradata
    except (av.error.FFmpegError, ValueError) as e:
      _log(f"Hardware decoder {name} is not in this FFmpeg build ({e}); using software decoding")
      return None
    self._hw_name = name
    return decoder

  @property
  def is_open(self):
    return self.container is not None

  def describe(self):
    """One line saying how this source decodes."""
    if self.hwaccel_active:
      how = f"hardware ({self._hw_name if self._hw_decoder else self.hwaccel})"
    elif self._hw_decoder is not None:
      how = f"hardware ({self._hw_name}) requested; confirmed or refused when the first frame decodes"
    else:
      how = "software, " + ("all cores" if self.threads == 0 else f"{self.threads} thread(s)")
      if self.hwaccel:
        how += f" (hardware decoder '{self.hwaccel}' requested but not available)"
    return f"PyAV/FFmpeg {self._codec_description} {self.width}x{self.height}, decode: {how}"

  # --- Stages ---------------------------------------------------------------

  def receive(self):
    """
    Stage 'receive': get the next compressed packet from the file or network.

    For a live stream this blocks until the camera sends the next frame.

    Returns:
      av.Packet, or None at end of stream or on a network failure
    """
    try:
      return next(self._packets, None)
    except av.error.FFmpegError as e:
      # Timeouts and dropped connections land here; end of stream ends the
      # generator normally
      _log(f"Stream read failed: {e}")
      return None

  def decode(self, packet):
    """
    Stage 'decode': decompress a packet.

    Args:
      packet: av.Packet from receive()

    Returns:
      List of av.VideoFrame at full resolution in the decoder's pixel format.
      Usually one frame; none while a multi-threaded decoder fills its
      pipeline or if the packet was damaged.
    """
    if self._hw_decoder is not None:
      return self._decode_dedicated(packet)
    try:
      return packet.decode()
    except av.error.InvalidDataError:
      # Damaged packet (UDP loss, joining mid-stream): skip it
      self.decode_errors += 1
      return []

  def _decode_dedicated(self, packet):
    """
    Decode with the dedicated hardware decoder (see DEDICATED_DECODER_TYPES).

    Until it has produced a frame the decoder is unconfirmed: the packets it
    was given are kept, and if it fails they are decoded in software instead,
    so no frames are lost. The failure is logged with FFmpeg's own reason.
    """
    if self.hwaccel_active:
      try:
        return self._hw_decoder.decode(packet)
      except av.error.InvalidDataError:
        self.decode_errors += 1
        return []
      except av.error.FFmpegError as e:
        _log(f"Hardware decoder {self._hw_name} failed ({e}); switching to software decoding")
        self._hw_decoder = None
        self.hwaccel_active = False
        return self.decode(packet)

    self._hw_pending.append(packet)
    # Collect FFmpeg's error messages: the exception alone doesn't say why
    av.logging.set_level(av.logging.ERROR)
    try:
      with av.logging.Capture() as ffmpeg_log:
        frames = self._hw_decoder.decode(packet)
    except av.error.FFmpegError as e:
      reasons = [message.strip() for _, _, message in ffmpeg_log] or [str(e)]
      _log(f"Hardware decoder {self._hw_name} failed: {'; '.join(reasons)}. Using software decoding")
      self._hw_decoder = None
      pending, self._hw_pending = self._hw_pending, []
      return [frame for unconfirmed in pending for frame in self.decode(unconfirmed)]
    finally:
      av.logging.set_level(None)

    if frames:
      self.hwaccel_active = True
      self._hw_pending = []
      _log(f"Hardware decoding active: {self._hw_name}, frames arrive as {frames[0].format.name}")
    return frames

  def crop(self, frame):
    """
    Stage 'crop': cut the detection region out of the decoded frame's planes.

    A YUV 4:2:0 frame is a full-size brightness plane (Y) and colour planes
    (U, V) at half size in each direction, so the region is cut from each
    plane and packed into one array: Y rows, then U, then V. Hardware
    decoders' nv12 (U and V interleaved in one plane) is unpacked into the
    same layout, so both kinds of decoder give identical pixels.

    Args:
      frame: av.VideoFrame from decode()

    Returns:
      (region, format): the region as a (height * 3/2, width) uint8 array and
      the pixel format it is in. If the format can't be cut directly, returns
      the whole frame and convert() crops after converting.
    """
    fmt = frame.format.name
    if fmt not in self.CROPPABLE_FORMATS or not self._even_crop:
      if self._warned_format != fmt:
        self._warned_format = fmt
        _log(f"Pixel format {fmt} (crop {self.crop_box}) can't be cropped before conversion; "
             "converting whole frames")
      return frame, None

    x1, y1, x2, y2 = self.crop_box
    w, h = x2 - x1, y2 - y1
    y_plane = self._plane(frame, 0)[y1:y2, x1:x2]
    region = np.empty((h * 3 // 2, w), np.uint8)
    flat = region.reshape(-1)
    flat[:h * w].reshape(h, w)[:] = y_plane

    ch, cw = h // 2, w // 2
    u_out = flat[h * w:h * w + ch * cw].reshape(ch, cw)
    v_out = flat[h * w + ch * cw:].reshape(ch, cw)
    if fmt in self.PLANAR_FORMATS:
      # Separate U and V planes, each (h/2, w/2)
      rows, cols = slice(y1 // 2, y2 // 2), slice(x1 // 2, x2 // 2)
      u_out[:] = self._plane(frame, 1)[rows, cols]
      v_out[:] = self._plane(frame, 2)[rows, cols]
      return region, fmt

    # nv12: one plane of interleaved U,V pairs, (h/2, w) bytes
    uv = self._plane(frame, 1)[y1 // 2:y2 // 2, x1:x2]
    u_out[:] = uv[:, 0::2]
    v_out[:] = uv[:, 1::2]
    full_range = frame.color_range == ColorRange.JPEG
    return region, "yuvj420p" if full_range else "yuv420p"

  def convert(self, region, fmt):
    """
    Stage 'convert': YUV -> BGR with FFmpeg's libswscale.

    Args:
      region, fmt: Output of crop()

    Returns:
      BGR image, (height, width, 3) uint8
    """
    if fmt is None:
      # crop() passed the whole frame through
      x1, y1, x2, y2 = self.crop_box
      return np.ascontiguousarray(region.to_ndarray(format='bgr24')[y1:y2, x1:x2])
    return av.VideoFrame.from_ndarray(region, format=fmt).to_ndarray(format='bgr24')

  # --------------------------------------------------------------------------

  def read(self):
    """
    Run receive -> decode -> crop -> convert for the next frame.

    Returns:
      Cropped BGR image, or None at end of stream or on failure
    """
    while not self._frames:
      with self.timer.stage("receive"):
        packet = self.receive()
      if packet is None:
        return None
      with self.timer.stage("decode"):
        self._frames.extend(self.decode(packet))

    frame = self._frames.popleft()
    with self.timer.stage("crop"):
      region, fmt = self.crop(frame)
    with self.timer.stage("convert"):
      return self.convert(region, fmt)

  def close(self):
    """Release the source."""
    if self.container is not None:
      try:
        self.container.close()
      except av.error.FFmpegError:
        pass  # Already dead; nothing to salvage
      self.container = None

  @staticmethod
  def _plane(frame, index):
    """View one plane of a frame as a 2-D array (rows may be padded past the width)."""
    plane = frame.planes[index]
    return np.frombuffer(plane, np.uint8).reshape(plane.height, plane.line_size)


class OpenCVSource:
  """
  Video source built on cv2.VideoCapture (fallback; DECODER = "opencv").

  OpenCV does receive, decode and whole-frame BGR conversion in one call, so
  they show up as a single 'decode+convert' stage.
  """

  def __init__(self, src, crop, is_live=False, rtsp_transport='tcp', timer=None):
    """
    Args:
      src: File path or RTSP URL
      crop: Detection region as (x1, y1, x2, y2) in full-frame pixels
      is_live: True for RTSP streams
      rtsp_transport: 'tcp' or 'udp' for RTSP streams
      timer: StageTimer that records the stage times
    """
    self.src = src
    self.crop_box = crop
    self.is_live = is_live
    self.rtsp_transport = rtsp_transport
    self.timer = timer or StageTimer()
    self.stream = None

    # Stream facts, filled in by open()
    self.width = 0
    self.height = 0
    self.fps = 0.0          # Frame rate from the stream metadata (0 if unknown)
    self.frame_count = 0    # Total frames, for files (0 if unknown)

  def open(self):
    """Open the source. Returns True on success."""
    if self.is_live:
      if self.rtsp_transport == 'tcp':
        os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"
        self.stream = cv2.VideoCapture(self.src, cv2.CAP_FFMPEG)
      else:
        self.stream = cv2.VideoCapture(self.src)
      self.stream.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    else:
      self.stream = cv2.VideoCapture(self.src)
    if not self.stream.isOpened():
      return False

    # Stream facts, kept so they can still be read after close()
    self.width = int(self.stream.get(cv2.CAP_PROP_FRAME_WIDTH))
    self.height = int(self.stream.get(cv2.CAP_PROP_FRAME_HEIGHT))
    self.fps = self.stream.get(cv2.CAP_PROP_FPS)
    self.frame_count = int(self.stream.get(cv2.CAP_PROP_FRAME_COUNT))
    return True

  @property
  def is_open(self):
    return self.stream is not None and self.stream.isOpened()

  def describe(self):
    """One line saying how this source decodes."""
    return f"OpenCV VideoCapture {self.width}x{self.height}, decode: software, all cores, whole frame to BGR"

  def read(self):
    """
    Read the next frame and crop it.

    Returns:
      Cropped BGR image, or None at end of stream or on failure
    """
    with self.timer.stage("decode+convert"):
      success, frame = self.stream.read()
    if not success:
      return None
    with self.timer.stage("crop"):
      x1, y1, x2, y2 = self.crop_box
      # Copy so the queue doesn't keep the whole 4K frame alive
      return np.ascontiguousarray(frame[y1:y2, x1:x2])

  def close(self):
    """Release the source."""
    if self.stream is not None:
      self.stream.release()


def create_video_source(src, crop, is_live=False, rtsp_transport='tcp', timer=None, decoder=DECODER):
  """Create the source selected by DECODER ('pyav' or 'opencv')."""
  if decoder == "pyav":
    return PyAVSource(src, crop, is_live, rtsp_transport, timer)
  if decoder == "opencv":
    return OpenCVSource(src, crop, is_live, rtsp_transport, timer)
  raise ValueError(f"DECODER must be 'pyav' or 'opencv', got {decoder!r}")

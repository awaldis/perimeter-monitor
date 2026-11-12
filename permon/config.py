"""
Configuration constants shared by the monitor, viewer and labeling tools.

Machine-specific overrides go in config_local.py at the repo root (not in
git; see config_local_template.py).
"""

from pathlib import Path

# Detection settings
CONFIDENCE_THRESHOLD = 0.5

# Class names mapping (class_id -> name) - COCO defaults
CLASS_NAMES = {
  2: 'car',
  3: 'motorcycle',
  5: 'bus',
  7: 'truck',
}

# Which classes to detect
VEHICLE_CLASSES = [2, 3, 5, 7]

# Crop region for 4K video (specific to camera setup)
CROP_Y1, CROP_Y2 = 202, 682    # Vertical range
CROP_X1, CROP_X2 = 2152, 3752  # Horizontal range

# Model settings
MODEL_PATH = "models/yolov8n.pt"  # COCO model, auto-downloaded on first run
MODEL_IMGSZ = [480, 1600]      # Model input [height, width]; exported models carry their own
# Where the network runs. OpenVINO models: "AUTO" (OpenVINO picks), "CPU",
# "GPU" (Intel graphics) or "NPU". Other models: "AUTO" (CUDA if present),
# "cpu" or "cuda:0".
INFERENCE_DEVICE = "AUTO"

# Video decoding
DECODER = "pyav"               # "pyav": FFmpeg via PyAV, each step separate. "opencv": cv2.VideoCapture
DECODE_THREADS = 1             # FFmpeg decoder threads: 1 = in the reader thread, 0 = every core
# Hardware decoder for "pyav": None (software), "qsv" (Intel Quick Sync) or
# "cuda" (NVIDIA). Falls back to software, logging FFmpeg's reason, if the
# hardware, its driver or its runtime library is missing.
DECODE_HWACCEL = None

# Recording settings
CLIPS_DIR = "clips"            # Where clips and screenshots are saved (and the viewer reads)
CLIP_BUFFER_SECONDS = 1.0      # Continue recording for 1 second after last detection
DEFAULT_FPS = 25               # Fallback FPS if detection fails

# Status/logging settings
STATUS_INTERVAL_SECONDS = 5.0  # How often to print status in RTSP mode
TIMING_REPORT_INTERVAL_SECONDS = 0  # RTSP mode: also print the stage timing table this often (0 = only at exit)
QUEUE_WARNING_INTERVAL = 5.0   # How often to warn about slow processing

# Video reader settings
READER_QUEUE_SIZE = 3          # Number of frames to buffer
FPS_DRAIN_FRAMES = 30          # Frames to discard when measuring FPS
FPS_MEASURE_FRAMES = 15        # Frames to measure FPS over

# RTSP reconnection (stream drops, or server stops sending data)
RECONNECT_DELAY_SECONDS = 5.0  # Wait between reconnect attempts (retries forever)
RTSP_TIMEOUT_SECONDS = 30.0    # Treat the stream as dropped after this long without data

# Telegram alerts (disabled by default - configure in config_local.py)
TELEGRAM_BOT_TOKEN = None      # Set in config_local.py
TELEGRAM_CHAT_ID = None        # Set in config_local.py
ALERT_VEHICLE_CLASSES = []     # List of class IDs that trigger alerts (e.g., [4, 5, 6, 7])
ALERT_EDGE_MARGIN = 0.10       # Min distance from left/right frame edge (0.05 = 5% of frame width)

# Stall detection (ignore parked/stationary vehicles)
STALL_SECONDS = 5.0            # Seconds of no movement before a vehicle is considered stalled
STALL_MOVEMENT_THRESHOLD = 0.01  # Normalized distance (1% of frame) below which a vehicle is "not moving"

# Web ports (the viewer and the FiftyOne app can run on the same machine)
VIEWER_PORT = 8081
FIFTYONE_PORT = 5151

# Override any settings with machine-specific config (not in git)
_LOCAL_CONFIG = Path(__file__).resolve().parent.parent / "config_local.py"
if _LOCAL_CONFIG.exists():
  exec(compile(_LOCAL_CONFIG.read_text(), str(_LOCAL_CONFIG), "exec"))

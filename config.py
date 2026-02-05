"""
Configuration constants for the perimeter monitor.
"""

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
MODEL_PATH = "yolov8n.pt"
MODEL_IMGSZ = [480, 1600]

# Recording settings
CLIP_BUFFER_SECONDS = 1.0      # Continue recording for 1 second after last detection
DEFAULT_FPS = 25               # Fallback FPS if detection fails

# Status/logging settings
STATUS_INTERVAL_SECONDS = 5.0  # How often to print status in RTSP mode
QUEUE_WARNING_INTERVAL = 5.0   # How often to warn about slow processing

# Video reader settings
READER_QUEUE_SIZE = 3          # Number of frames to buffer
FPS_DRAIN_FRAMES = 30          # Frames to discard when measuring FPS
FPS_MEASURE_FRAMES = 15        # Frames to measure FPS over

# Telegram alerts (disabled by default - configure in config_local.py)
TELEGRAM_BOT_TOKEN = None      # Set in config_local.py
TELEGRAM_CHAT_ID = None        # Set in config_local.py
ALERT_VEHICLE_CLASSES = []     # List of class IDs that trigger alerts (e.g., [4, 5, 6, 7])
ALERT_EDGE_MARGIN = 0.10       # Min distance from left/right frame edge (0.05 = 5% of frame width)

# Override any settings with machine-specific config (not in git)
try:
  from config_local import *
except ImportError:
  pass

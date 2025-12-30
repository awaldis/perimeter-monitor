"""
Configuration constants for the perimeter monitor.
"""

# Detection settings
CONFIDENCE_THRESHOLD = 0.5
VEHICLE_CLASSES = [2, 3, 5, 7]  # COCO: 2=car, 3=motorcycle, 5=bus, 7=truck

# Crop region for 4K video (specific to camera setup)
CROP_Y1, CROP_Y2 = 202, 682    # Vertical range
CROP_X1, CROP_X2 = 2152, 3752  # Horizontal range

# Model settings
MODEL_PATH = "yolov8n.onnx"
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

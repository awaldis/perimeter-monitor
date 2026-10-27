"""
Local configuration template for perimeter monitor.
Copy this file to config_local.py (in the repo root, next to this file) and
customize for your environment. Any setting in permon/config.py can be
overridden here.
"""

# Directory where video clips will be saved
# For network drives, use the mount point path:
#   - Linux: "/mnt/nas/perimeter_clips"
#   - Windows: "Z:/perimeter_clips" or "//server/share/perimeter_clips"
CLIPS_DIR = "/mnt/nas/perimeter_clips"

# Custom model (a .pt file or an OpenVINO model folder), usually in models/
#MODEL_PATH = "models/permon_2026_02_11_model_openvino_model"

# Class names for the outputs of the model in MODEL_PATH (class_id -> name).
# The labeling export uses this list too, so the IDs must be 0..N-1.
#CLASS_NAMES = {
#  0: 'car',
#  1: 'suv',
#  2: 'pickup_truck',
#  3: 'truck',
#  4: 'usmail_truck',
#  5: 'fedex_truck',
#  6: 'ups_truck',
#  7: 'amazon_truck',
#}
#VEHICLE_CLASSES = [0, 1, 2, 3, 4, 5, 6, 7]

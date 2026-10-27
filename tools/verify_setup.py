#!/usr/bin/env python3
"""
Verify that the perimeter-monitor installation is working correctly.

Usage (from the repo root):
  python tools/verify_setup.py
"""

import sys
from pathlib import Path

# Make the permon package importable when run as a script
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

def check_import(module_name, package_name=None):
    """Try to import a module and report status."""
    package_name = package_name or module_name
    try:
        __import__(module_name)
        print(f"✓ {package_name}")
        return True
    except ImportError as e:
        print(f"✗ {package_name}: {e}")
        return False

def main():
    print("=== Verifying Perimeter Monitor Setup ===\n")

    all_ok = True

    # Check Python version
    print("Python version:")
    print(f"  {sys.version}")
    if sys.version_info < (3, 10):
        print("  ✗ Python 3.10+ required")
        all_ok = False
    else:
        print("  ✓ Version OK")
    print()

    # Check required packages
    print("Checking required packages:")
    packages = [
        ('cv2', 'opencv-python'),
        ('numpy', 'numpy'),
        ('torch', 'torch'),
        ('torchvision', 'torchvision'),
        ('ultralytics', 'ultralytics'),
        ('openvino', 'openvino'),
    ]

    for module, package in packages:
        if not check_import(module, package):
            all_ok = False
    print()

    # Check device availability
    print("Checking compute devices:")
    try:
        import torch
        if torch.cuda.is_available():
            print(f"  ✓ CUDA available: {torch.cuda.get_device_name(0)}")
        else:
            print("  ℹ CUDA not available (will use CPU)")
        print(f"  ✓ CPU available")
    except Exception as e:
        print(f"  ✗ Error checking devices: {e}")
        all_ok = False
    print()

    # Check OpenVINO devices
    print("Checking OpenVINO devices:")
    try:
        import openvino as ov
        devices = ov.Core().available_devices
        print(f"  Available devices: {', '.join(devices)}")
        if any(d.startswith('GPU') for d in devices):
            print("  ✓ Intel GPU available to OpenVINO")
        else:
            print("  ℹ Using CPU execution")
    except Exception as e:
        print(f"  ✗ Error checking OpenVINO: {e}")
        all_ok = False
    print()

    # Check for YOLO model
    print("Checking for YOLO model:")
    import os
    from permon.config import MODEL_PATH
    if os.path.exists(MODEL_PATH):
        print(f"  ✓ {MODEL_PATH} found")
    else:
        print(f"  ℹ {MODEL_PATH} not found (yolov8n.pt auto-downloads on first run; set MODEL_PATH in config_local.py)")
    print()

    # Check OpenCV capabilities
    print("Checking OpenCV capabilities:")
    try:
        import cv2
        print(f"  OpenCV version: {cv2.__version__}")
        # Check for FFmpeg support
        backends = cv2.videoio_registry.getBackends()
        if cv2.CAP_FFMPEG in backends:
            print("  ✓ FFmpeg support available")
        else:
            print("  ✗ FFmpeg support not available")
            all_ok = False
    except Exception as e:
        print(f"  ✗ Error checking OpenCV: {e}")
        all_ok = False
    print()

    # Final result
    print("=" * 45)
    if all_ok:
        print("✓ Setup verification PASSED")
        print("\nYou can now run:")
        print("  python -m permon.monitor --help")
        return 0
    else:
        print("✗ Setup verification FAILED")
        print("\nPlease install missing dependencies:")
        print("  pip install -r requirements.txt")
        return 1

if __name__ == '__main__':
    sys.exit(main())

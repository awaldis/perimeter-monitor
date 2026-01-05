#!/usr/bin/env python3
"""
Verify that the perimeter-monitor installation is working correctly.
"""

import sys

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
        ('onnxruntime', 'onnxruntime'),
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

    # Check ONNX Runtime providers
    print("Checking ONNX Runtime providers:")
    try:
        import onnxruntime as ort
        providers = ort.get_available_providers()
        print(f"  Available providers: {', '.join(providers)}")
        if 'CUDAExecutionProvider' in providers:
            print("  ✓ GPU acceleration available (CUDA)")
        elif 'OpenVINOExecutionProvider' in providers:
            print("  ✓ GPU acceleration available (OpenVINO)")
        else:
            print("  ℹ Using CPU execution (consider installing onnxruntime-openvino for Intel GPU)")
    except Exception as e:
        print(f"  ✗ Error checking ONNX Runtime: {e}")
        all_ok = False
    print()

    # Check for YOLO model
    print("Checking for YOLO model:")
    import os
    if os.path.exists('yolov8n.onnx'):
        print("  ✓ yolov8n.onnx found")
    else:
        print("  ℹ yolov8n.onnx not found (will auto-download on first run)")
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
        print("  python mon_perim.py --help")
        return 0
    else:
        print("✗ Setup verification FAILED")
        print("\nPlease install missing dependencies:")
        print("  pip install -r requirements.txt")
        return 1

if __name__ == '__main__':
    sys.exit(main())

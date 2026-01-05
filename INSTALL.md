# Installation Guide

## Quick Start on New Machine

### 1. Clone Repository
```bash
git clone https://github.com/awaldis/perimeter-monitor.git
cd perimeter-monitor
```

### 2. Run Setup Script
```bash
chmod +x setup.sh
./setup.sh
```

### 3. Verify Installation
```bash
source venv/bin/activate
python verify_setup.py
```

### 4. Test Run
```bash
# Test with your RTSP stream
python mon_perim.py -t udp "rtsp://admin:Dexter8@192.168.50.60:554/rtsp/streaming?channel=02&subtype=0"
```

## Intel GPU Optimization (Recommended for i5-1135G7)

Your target machine has **Intel Iris Xe Graphics** which supports hardware acceleration through OpenVINO.

### Install OpenVINO Support
```bash
source venv/bin/activate
pip install onnxruntime-openvino
```

### Verify OpenVINO
```bash
python verify_setup.py
# Should show: "GPU acceleration available (OpenVINO)"
```

### Enable in Code
The code auto-detects available execution providers, but you can verify by checking the output when running:
```bash
python mon_perim.py --help
# Will show which execution provider is being used
```

## System Requirements by CPU

### Intel i5-1135G7 (Your Target)
- **CPU**: 4 cores, 8 threads @ 2.40GHz (up to 4.2GHz boost)
- **iGPU**: Intel Iris Xe Graphics (80 EUs)
- **Expected Performance**:
  - CPU only: ~10-15 FPS for 4K YOLO inference
  - With OpenVINO: ~20-30 FPS for 4K YOLO inference
  - Recommended: Use 1080p stream for real-time (30 FPS)

### Comparison: Your WSL Machine (MX150)
- CUDA GPU acceleration
- Current: Processing 4K @ 7.31 FPS smoothly

## Performance Tuning for Intel CPU/GPU

### 1. Use Lower Resolution Stream
```bash
# Use subtype=1 for 720p instead of subtype=0 for 4K
python mon_perim.py -t udp "rtsp://admin:Dexter8@192.168.50.60:554/rtsp/streaming?channel=02&subtype=1"
```

### 2. Reduce Detection Image Size
Edit `config.py`:
```python
MODEL_IMGSZ = 640  # Default is 1280 for 4K
```

### 3. Increase Queue Size for Buffering
Edit `config.py`:
```python
READER_QUEUE_SIZE = 60  # Default is 30
```

### 4. Use Lighter YOLO Model
```bash
# Download YOLOv8n (nano) if not already using it
wget https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.onnx

# Or use even smaller model (faster but less accurate)
wget https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8s.onnx
```

## Troubleshooting on New Machine

### Issue: "ModuleNotFoundError"
```bash
# Make sure virtual environment is activated
source venv/bin/activate

# Reinstall dependencies
pip install -r requirements.txt
```

### Issue: Poor Performance
```bash
# Check which execution provider is being used
python verify_setup.py

# If not using OpenVINO on Intel GPU:
pip install onnxruntime-openvino
```

### Issue: "Could not open video source"
```bash
# Test RTSP stream with ffplay
ffplay -rtsp_transport udp "rtsp://admin:Dexter8@192.168.50.60:554/rtsp/streaming?channel=02&subtype=1"

# If that works but mon_perim.py doesn't:
# - Check firewall rules
# - Try TCP transport: -t tcp
# - Check camera supports multiple connections
```

### Issue: High CPU Usage
This is normal for CPU-based inference. To reduce:
1. Lower resolution (use subtype=1 or 2)
2. Reduce MODEL_IMGSZ in config.py
3. Skip frames: modify video_reader.py to read every 2nd frame

## Files to Push to New Machine

The setup.sh script will install everything, but these are the files needed:

### Required Files
- ✓ All Python source files (*.py)
- ✓ setup.sh
- ✓ requirements.txt
- ✓ README.md
- ✓ config.py

### Optional Files
- yolov8n.onnx (will auto-download if missing)
- clips/ directory (will be created)

### Not Needed on New Machine
- ✗ venv/ directory (created by setup.sh)
- ✗ __pycache__/ directories
- ✗ *.pyc files
- ✗ clips/*.mp4 (old recordings)

## Expected Performance Comparison

| System | GPU | Resolution | FPS | Notes |
|--------|-----|------------|-----|-------|
| Current WSL | NVIDIA MX150 (CUDA) | 4K (3840×2160) | 7.31 | Smooth processing |
| Target Server | Intel Iris Xe (CPU) | 4K (3840×2160) | ~10-15 | May lag behind stream |
| Target Server | Intel Iris Xe (OpenVINO) | 4K (3840×2160) | ~20-30 | Better, still may lag |
| Target Server | Intel Iris Xe (OpenVINO) | 1080p (1920×1080) | 30+ | Real-time capable |

**Recommendation**: For real-time monitoring on Intel i5-1135G7, use 1080p stream with OpenVINO.

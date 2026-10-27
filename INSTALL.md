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
python tools/verify_setup.py
```

### 4. Test Run
```bash
# Test with your RTSP stream
python -m permon.monitor -t udp "rtsp://USERNAME:PASSWORD@CAMERA_IP:554/rtsp/streaming?channel=02&subtype=0"
```

## Intel GPU Optimization (Recommended for i5-1135G7)

Your target machine has **Intel Iris Xe Graphics** which supports hardware acceleration through OpenVINO.

OpenVINO is installed from `requirements.txt`.

### Verify OpenVINO
```bash
python tools/verify_setup.py
# Should show: "Intel GPU available to OpenVINO"
```

### Use an OpenVINO Model
Export the trained `.pt` model on the dev machine, copy the exported folder to
`models/`, and point `MODEL_PATH` at it in `config_local.py`:
```bash
python -m permon.labeling.export_openvino models/permon_2026_02_11_model.pt
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
python -m permon.monitor -t udp "rtsp://USERNAME:PASSWORD@CAMERA_IP:554/rtsp/streaming?channel=02&subtype=1"
```

### 2. Reduce Detection Image Size
Edit `config_local.py` (re-export OpenVINO models at the new size):
```python
MODEL_IMGSZ = 640  # Default is 1280 for 4K
```

### 3. Increase Queue Size for Buffering
Edit `config_local.py`:
```python
READER_QUEUE_SIZE = 60  # Default is 30
```

### 4. Use Lighter YOLO Model
Train from a smaller base model (YOLOv8n, nano, is the smallest), or use the
COCO `yolov8n.pt` by leaving `MODEL_PATH` unset.

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
# Check which devices OpenVINO can use
python tools/verify_setup.py

# If not using an OpenVINO model on Intel GPU, export one (see above)
```

### Issue: "Could not open video source"
```bash
# Test RTSP stream with ffplay
ffplay -rtsp_transport udp "rtsp://USERNAME:PASSWORD@CAMERA_IP:554/rtsp/streaming?channel=02&subtype=1"

# If that works but the monitor doesn't:
# - Check firewall rules
# - Try TCP transport: -t tcp
# - Check camera supports multiple connections
```

### Issue: High CPU Usage
This is normal for CPU-based inference. To reduce:
1. Lower resolution (use subtype=1 or 2)
2. Reduce MODEL_IMGSZ in config_local.py
3. Skip frames: modify permon/monitor/video_reader.py to read every 2nd frame

## Files to Push to New Machine

The setup.sh script will install everything, but these are the files needed:

### Required Files
- ✓ The permon/ package
- ✓ setup.sh
- ✓ requirements.in / requirements.txt
- requirements-dev.in / requirements-dev.txt (only for the labeling workflow and tests)
- ✓ README.md
- ✓ config_local.py (machine-specific; not in git)

### Optional Files
- models/ (your custom models; yolov8n.pt auto-downloads if MODEL_PATH is unset)
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

# Perimeter Monitor

Real-time vehicle detection system using YOLOv8 for RTSP streams and video files.

## Features

- **RTSP Live Streaming**: Monitor IP cameras with vehicle detection
- **Clip Recording**: Automatically save clips when vehicles are detected
- **Video File Processing**: Process pre-recorded video files
- **Multi-GPU Support**: Works with NVIDIA CUDA, Intel GPU (OpenVINO), or CPU
- **Threaded Video Reader**: Efficient frame processing with background buffering

## System Requirements

### Minimum
- Ubuntu 20.04 or later (also works on WSL)
- Python 3.10+
- 4GB RAM
- CPU with AVX support

### Recommended
- 8GB+ RAM
- Intel GPU (Iris Xe or newer) or NVIDIA GPU
- SSD storage for clip recording

## Installation

### Quick Setup

```bash
# Clone the repository
git clone https://github.com/awaldis/perimeter-monitor.git
cd perimeter-monitor

# Run setup script
chmod +x setup.sh
./setup.sh

# Activate virtual environment
source venv/bin/activate
```

### Manual Setup

1. **Install system dependencies**:
   ```bash
   sudo apt-get update
   sudo apt-get install -y python3 python3-pip python3-venv git ffmpeg \
       libsm6 libxext6 libxrender-dev libgomp1
   ```

2. **Create virtual environment**:
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Install Python dependencies**:
   ```bash
   # CPU-only PyTorch (smaller, faster for inference)
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

   # Other dependencies
   pip install -r requirements.txt
   ```

4. **Download YOLO model** (optional, will auto-download on first run):
   ```bash
   wget https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.onnx
   ```

## GPU Configuration

### Intel GPU (Iris Xe, UHD Graphics)

For better performance on Intel integrated graphics:

```bash
# Install OpenVINO execution provider
pip install onnxruntime-openvino
```

Then modify `vehicle_detector.py` to use OpenVINO provider:
```python
providers = ['OpenVINOExecutionProvider', 'CPUExecutionProvider']
```

### NVIDIA GPU

If moving to a system with NVIDIA GPU:

```bash
# Uninstall CPU version
pip uninstall torch torchvision onnxruntime

# Install CUDA versions
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install onnxruntime-gpu
```

## Usage

### RTSP Live Monitoring

```bash
# Basic usage with UDP transport
python mon_perim.py -t udp "rtsp://user:pass@192.168.1.100:554/stream"

# With live display window
python mon_perim.py -t udp -d "rtsp://user:pass@192.168.1.100:554/stream"

# TCP transport (more reliable, higher latency)
python mon_perim.py -t tcp "rtsp://user:pass@192.168.1.100:554/stream"
```

### Video File Processing

```bash
# Process a video file
python mon_perim.py path/to/video.mp4

# With live display
python mon_perim.py -d path/to/video.mp4
```

### Command Line Options

```
positional arguments:
  video_source          RTSP URL or path to video file

optional arguments:
  -h, --help            Show help message
  -t {tcp,udp}          RTSP transport protocol (default: tcp)
  -d, --display         Show live video display window
  -o OUTPUT_DIR         Output directory for clips (default: clips)
  -c CONFIDENCE         Detection confidence threshold (default: 0.5)
```

## Configuration

Edit `config.py` to customize:

- **Detection region**: `CROP_X1, CROP_X2, CROP_Y1, CROP_Y2`
- **Vehicle classes**: `VEHICLE_CLASSES` (2=car, 3=motorcycle, 5=bus, 7=truck)
- **Recording duration**: `RECORD_BUFFER_SECONDS` (buffer after last detection)
- **Queue size**: `READER_QUEUE_SIZE` (frame buffer size)
- **Status update interval**: `STATUS_INTERVAL_SECONDS`

## Output

### RTSP Mode
Clips are saved to the `clips/` directory when vehicles are detected:
- Format: `clip_YYYYMMDD_HHMMSS.mp4`
- Includes 1-second buffer after last detection
- Uses same FPS and resolution as source

### File Mode
Processed video saved as:
- `output_<original_filename>.mp4`
- Full video with detection overlays

## Troubleshooting

### "Assertion fctx->async_lock failed"
This is a threading issue with older FFmpeg. Upgrade FFmpeg:
```bash
sudo apt-get update && sudo apt-get upgrade ffmpeg
```

### Poor Performance on Intel GPU
1. Verify Intel GPU is detected: `lspci | grep -i graphics`
2. Install OpenVINO execution provider (see GPU Configuration)
3. Reduce input resolution using camera's substream
4. Lower frame rate in camera settings

### RTSP Connection Issues
- Check network connectivity: `ping <camera-ip>`
- Verify RTSP URL with: `ffplay "rtsp://..."`
- Try different transport: use `-t tcp` if UDP fails
- Check camera stream settings (resolution, framerate)

### Memory Usage
- Reduce `READER_QUEUE_SIZE` in `config.py`
- Lower camera resolution
- Use smaller YOLO model if needed

## Architecture

```
mon_perim.py          - Main entry point
├── config.py         - Configuration constants
├── video_reader.py   - Threaded video capture
├── vehicle_detector.py - YOLO wrapper
└── clip_recorder.py  - Video recording manager
```

## License

MIT License - See LICENSE file for details

## Credits

- YOLOv8: Ultralytics (https://github.com/ultralytics/ultralytics)
- ONNX Runtime: Microsoft (https://onnxruntime.ai/)

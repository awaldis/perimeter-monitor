# Perimeter Monitor

Real-time vehicle detection system using YOLOv8 for RTSP streams and video files.

The repo has three parts, all in the `permon` package:

| Part | Package | Runs on | Purpose |
|------|---------|---------|---------|
| Monitor | `permon.monitor` | production machine | Detects and tracks vehicles in the camera stream, saves clips and screenshots, sends Telegram alerts |
| Viewer | `permon.viewer` | production machine | Web gallery of the saved screenshots |
| Labeling | `permon.labeling` | dev machine | FiftyOne workflow for choosing and labeling screenshots, and exporting training sets and models |

## Features

- **RTSP Live Streaming**: Monitor IP cameras with vehicle detection
- **Clip Recording**: Automatically save clips when vehicles are detected
- **Parked Vehicle Filtering**: Vehicles that stop moving don't keep triggering recordings
- **Telegram Alerts**: Notifications for selected vehicle classes
- **Video File Processing**: Process pre-recorded video files
- **CPU, NVIDIA or Intel**: PyTorch `.pt` models on CPU or CUDA, OpenVINO models on Intel CPU/GPU
- **Threaded Video Reader**: Efficient frame processing with background buffering

## Project Layout

```
permon/
├── config.py               - Shared settings (overridden by config_local.py)
├── monitor/                - Real-time processing
│   ├── cli.py              -   Entry point (python -m permon.monitor)
│   ├── video_reader.py     -   Threaded video capture with RTSP reconnect
│   ├── frame_pipeline.py   -   Per-frame crop, detect, track, stall filter (shared with tests)
│   ├── vehicle_detector.py -   YOLO wrapper
│   ├── stall_tracker.py    -   Parked vehicle detection
│   ├── clip_recorder.py    -   Clip and screenshot recording
│   └── telegram_alert.py   -   Telegram alerts
├── viewer/                 - Web gallery (python -m permon.viewer)
└── labeling/               - FiftyOne session, dataset helpers, model export
tools/                      - Standalone utilities (verify_setup.py, rtsp_viewer.py)
tests/                      - Unit and video regression tests
models/                     - Model files (not in git)
archive/                    - Obsolete code kept for reference
config_local.py             - Machine-specific settings (not in git)
```

Run all commands from the repo root, with the virtual environment activated.

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

3. **Install Python dependencies** (pinned; includes CPU-only PyTorch):
   ```bash
   pip install -r requirements.txt

   # Optional: labeling workflow and tests (FiftyOne, IPython, pytest).
   # --no-deps is required so fiftyone doesn't pull in opencv-python-headless,
   # which would break cv2.imshow.
   pip install --no-deps -r requirements-dev.txt
   ```

4. **Models**: with no `config_local.py`, the monitor uses the COCO
   `yolov8n.pt`, which downloads to `models/` on first run. Put custom models
   (`.pt` files or OpenVINO model folders) in `models/` and set `MODEL_PATH`
   and `CLASS_NAMES` in `config_local.py` (see `config_local_template.py`).

5. **Verify**:
   ```bash
   python tools/verify_setup.py
   ```

### Managing Dependencies

Dependencies are declared in `.in` files and compiled into pinned lock files
with [uv](https://docs.astral.sh/uv/):

| Source (edit this)    | Lock file (generated)  | Contents                             |
|-----------------------|------------------------|--------------------------------------|
| `requirements.in`     | `requirements.txt`     | Runtime: monitor, viewer, OpenVINO   |
| `requirements-dev.in` | `requirements-dev.txt` | Labeling (FiftyOne, IPython), pytest |

`uv.toml` sets `index-strategy = "unsafe-best-match"` so uv considers both PyPI
and the PyTorch CPU index (as pip does); run uv commands from the repo root.

Never edit the `.txt` files by hand. To add, remove, or upgrade a package:

```bash
# 1. Edit requirements.in / requirements-dev.in, then recompile:
uv pip compile requirements.in -o requirements.txt \
    --index-strategy unsafe-best-match --emit-index-url
uv pip compile requirements-dev.in -o requirements-dev.txt \
    --index-strategy unsafe-best-match --emit-index-url \
    --no-emit-package opencv-python-headless

# To upgrade: add -P <package> (one package) or -U (everything) to the commands above.

# 2. Make the venv match the lock files exactly (removes anything not listed):
uv pip sync --python venv/bin/python requirements.txt requirements-dev.txt
```

Commit the `.in` and `.txt` files together.

## Usage

### Monitor: RTSP Live Monitoring

```bash
# Basic usage with UDP transport
python -m permon.monitor -t udp "rtsp://user:pass@192.168.1.100:554/stream"

# With live display window
python -m permon.monitor -t udp -d "rtsp://user:pass@192.168.1.100:554/stream"

# TCP transport (more reliable, higher latency)
python -m permon.monitor -t tcp "rtsp://user:pass@192.168.1.100:554/stream"
```

### Monitor: Video File Processing

```bash
# Process a video file
python -m permon.monitor path/to/video.mp4

# With live display
python -m permon.monitor -d path/to/video.mp4
```

### Monitor: Command Line Options

```
positional arguments:
  video_source          RTSP URL or path to video file

optional arguments:
  -h, --help            Show help message
  -t {tcp,udp}          RTSP transport protocol (default: tcp)
  -d, --display         Show live video display window
  -o OUTPUT             Output video file (file mode only)
  -c CONFIDENCE         Detection confidence threshold (default: CONFIDENCE_THRESHOLD)
```

### Viewer

```bash
python -m permon.viewer            # all detections
python -m permon.viewer --days 7   # only the last 7 days
```

Then open `http://<host>:8081` (`VIEWER_PORT`). The viewer reads `CLIPS_DIR`
and picks up new screenshots as the monitor saves them.

### Labeling Workflow

Start an IPython session with the master FiftyOne dataset loaded and the app
running at `http://<host>:5151` (`FIFTYONE_PORT`):

```bash
ipython -i -m permon.labeling.session
```

The session defines `dataset`, `session` and `fo`, plus these helpers:

```python
# Add new screenshots for review
dataset.add_dir("/path/to/clips/2026-02-10", dataset_type=fo.types.YOLOv5Dataset, tags=["new_import"])

# Add an image with no vehicles as a background sample
add_background(dataset, "/path/to/image.jpg")

# Delete samples not tagged 'master_set' (asks first)
delete_non_master_samples(dataset)

# Export the master set with a fresh 80/20 split, zipped for Colab.
# new_sample_view (optional) is tagged 'master_set' first.
export_for_training(dataset, "/path/to/training_set_2026_02_11", new_sample_view=view)
```

The export uses `CLASS_NAMES` from the config for the class list. After
training, convert the model to OpenVINO for the monitor:

```bash
python -m permon.labeling.export_openvino models/permon_2026_02_11_model.pt
```

## Configuration

Settings live in `permon/config.py`. Override any of them for a machine in
`config_local.py` at the repo root (not in git; copy
`config_local_template.py`). Common ones:

- **Model**: `MODEL_PATH`, `CLASS_NAMES`, `VEHICLE_CLASSES`, `CONFIDENCE_THRESHOLD`
- **Output**: `CLIPS_DIR` (clips and screenshots; the viewer reads it too)
- **Detection region**: `CROP_X1, CROP_X2, CROP_Y1, CROP_Y2`
- **Recording duration**: `CLIP_BUFFER_SECONDS` (buffer after last detection)
- **Parked vehicles**: `STALL_SECONDS`, `STALL_MOVEMENT_THRESHOLD`
- **Alerts**: `TELEGRAM_BOT_TOKEN`, `TELEGRAM_CHAT_ID`, `ALERT_VEHICLE_CLASSES`
- **Ports**: `VIEWER_PORT`, `FIFTYONE_PORT`
- **Queue size**: `READER_QUEUE_SIZE` (frame buffer size)
- **Status update interval**: `STATUS_INTERVAL_SECONDS`

## Output

### RTSP Mode
Each time a vehicle appears, a clip is saved under `CLIPS_DIR`, in a folder
per day:
- `YYYY-MM-DD/<class>_<timestamp>.avi`: annotated video clip, ending
  `CLIP_BUFFER_SECONDS` after the last detection
- `YYYY-MM-DD/images/` and `labels/`: one screenshot and YOLO label file per
  clip frame, the layout FiftyOne imports

### File Mode
- `CLIPS_DIR/<name>_output.avi`: the full video with detection overlays
- `CLIPS_DIR/<name>_output_frames/`: a screenshot and label file for every frame

## Testing

Video regression tests run recorded clips through `FramePipeline`
(`permon/monitor/frame_pipeline.py`), the same per-frame detection, tracking,
and stall filtering that live monitoring uses:

```bash
pytest -v
```

Cases live in `tests/video_cases.yaml`. Each one names a video, an inclusive
0-based frame range (matching the `Frame N` lines the monitor prints), and
the vehicle class that must be the only active detection on every frame in
that range, the number of clips the live recorder must save (catches parked
vehicles being treated as new), and a maximum clip length (catches parked
vehicles being recorded the whole time they're parked). All checks are optional. Results depend on `MODEL_PATH` in
`config_local.py`.

`tests/test_stall_tracker.py` has fast unit tests for the stall logic that
need no video or model (`pytest tests/test_stall_tracker.py`).

The videos are not in git (`input_4K/` by default); cases whose video is
missing are skipped. `pytest` is installed from `requirements-dev.txt`.

## GPU Configuration

### Intel GPU (Iris Xe, UHD Graphics)

Export the model to OpenVINO format (see Labeling Workflow) and point
`MODEL_PATH` at the exported folder. `python tools/verify_setup.py` lists the
devices OpenVINO can use.

### NVIDIA GPU

If moving to a system with NVIDIA GPU, install the CUDA build of PyTorch and
use a `.pt` model:

```bash
pip uninstall torch torchvision
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

## Troubleshooting

### "Assertion fctx->async_lock failed"
This is a threading issue with older FFmpeg. Upgrade FFmpeg:
```bash
sudo apt-get update && sudo apt-get upgrade ffmpeg
```

### Poor Performance on Intel GPU
1. Verify Intel GPU is detected: `lspci | grep -i graphics`
2. Use an OpenVINO model (see GPU Configuration)
3. Reduce input resolution using camera's substream
4. Lower frame rate in camera settings

### RTSP Connection Issues
- Check network connectivity: `ping <camera-ip>`
- Verify RTSP URL with: `ffplay "rtsp://..."` or `python tools/rtsp_viewer.py "rtsp://..."`
- Try different transport: use `-t tcp` if UDP fails
- Check camera stream settings (resolution, framerate)

### Memory Usage
- Reduce `READER_QUEUE_SIZE` in `config_local.py`
- Lower camera resolution
- Use smaller YOLO model if needed

## License

MIT License - See LICENSE file for details

## Credits

- YOLOv8: Ultralytics (https://github.com/ultralytics/ultralytics)
- OpenVINO: Intel (https://github.com/openvinotoolkit/openvino)
- FiftyOne: Voxel51 (https://github.com/voxel51/fiftyone)

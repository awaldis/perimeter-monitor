#!/bin/bash
# Setup script for perimeter-monitor on Ubuntu systems
# Supports both CPU and Intel GPU configurations

set -e  # Exit on error

echo "=== Perimeter Monitor Setup ==="
echo

# Check Ubuntu version
echo "Checking system..."
if [ -f /etc/os-release ]; then
    . /etc/os-release
    echo "OS: $NAME $VERSION"
else
    echo "Warning: Cannot detect OS version"
fi

# Install system dependencies
echo
echo "Installing system dependencies..."
sudo apt-get update
sudo apt-get install -y \
    python3 \
    python3-pip \
    python3-venv \
    git \
    ffmpeg \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1

# Check for Intel GPU and offer OpenVINO
if lspci | grep -i "intel.*graphics" > /dev/null; then
    echo
    echo "Intel GPU detected!"
    echo "For better performance, consider using ONNX Runtime with OpenVINO."
    echo "See: https://onnxruntime.ai/docs/execution-providers/OpenVINO-ExecutionProvider.html"
fi

# Create virtual environment
echo
echo "Creating Python virtual environment..."
if [ ! -d "venv" ]; then
    python3 -m venv venv
    echo "Virtual environment created"
else
    echo "Virtual environment already exists"
fi

# Activate virtual environment
echo
echo "Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo
echo "Upgrading pip..."
pip install --upgrade pip

# Install PyTorch (CPU version - smaller and faster for inference)
echo
echo "Installing PyTorch (CPU version)..."
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# Install other dependencies
echo
echo "Installing other Python dependencies..."
pip install -r requirements.txt

# Download YOLO model if not present
echo
echo "Checking for YOLO model..."
if [ ! -f "yolov8n.onnx" ]; then
    echo "YOLO model not found. It will be downloaded on first run."
    echo "Or download manually: https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.onnx"
else
    echo "YOLO model found: yolov8n.onnx"
fi

# Create clips directory
echo
echo "Creating output directory..."
mkdir -p clips

echo
echo "=== Setup Complete ==="
echo
echo "To activate the virtual environment:"
echo "  source venv/bin/activate"
echo
echo "To run the monitor:"
echo "  python mon_perim.py -t udp \"rtsp://user:pass@ip:port/stream\""
echo
echo "For help:"
echo "  python mon_perim.py --help"

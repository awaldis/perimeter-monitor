#!/usr/bin/env python3
"""
Detection Viewer - Web-based real-time detection monitoring.

Displays detection thumbnails in a grid, updates in real-time when new
detections occur, and allows clicking to view all frames from a detection.

Usage:
  python detection_viewer.py
  Then open http://localhost:8080 in a browser
"""

import json
import os
import queue
import re
import threading
import time
from pathlib import Path

from flask import Flask, Response, jsonify, render_template, abort
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

# Import local configuration for clips directory (with fallback)
try:
  from config_local import CLIPS_DIR
except ImportError:
  CLIPS_DIR = "clips"

app = Flask(__name__)

# Queue for SSE events
event_queue = queue.Queue()

# Track known detections
known_detections = []
detections_lock = threading.Lock()


def parse_detection_name(dirname):
  """
  Parse detection directory name to extract vehicle type and timestamp.

  Args:
    dirname: Directory name like 'car_2026-01-08T14-30-45_frames'

  Returns:
    Dict with 'name', 'vehicle_type', 'timestamp', 'display_time' or None if invalid
  """
  # Remove _frames suffix
  if not dirname.endswith('_frames'):
    return None

  base_name = dirname[:-7]  # Remove '_frames'

  # Try to parse new format: vehicle_YYYY-MM-DDTHH-MM-SS
  match = re.match(r'^(\w+)_(\d{4}-\d{2}-\d{2})T(\d{2}-\d{2}-\d{2})$', base_name)
  if match:
    vehicle_type = match.group(1)
    date_part = match.group(2)
    time_part = match.group(3).replace('-', ':')
    return {
      'name': base_name,
      'vehicle_type': vehicle_type,
      'timestamp': f"{date_part}T{time_part}",
      'display_time': f"{date_part} {time_part}"
    }

  # Try to parse old format: clip_timestamp_count
  match = re.match(r'^clip_(\d+)_(\d+)$', base_name)
  if match:
    return {
      'name': base_name,
      'vehicle_type': 'vehicle',
      'timestamp': match.group(1),
      'display_time': f"clip #{match.group(2)}"
    }

  # Try other formats (like video file outputs)
  return {
    'name': base_name,
    'vehicle_type': 'unknown',
    'timestamp': '0',
    'display_time': base_name
  }


def get_thumbnail_path(detection_name):
  """Get the path to frame_0003.jpg for a detection."""
  return f"{detection_name}_frames/frame_0003.jpg"


def scan_existing_detections():
  """Scan CLIPS_DIR for existing detection directories."""
  global known_detections

  detections = []
  clips_path = Path(CLIPS_DIR)

  if not clips_path.exists():
    return

  for item in clips_path.iterdir():
    if item.is_dir() and item.name.endswith('_frames'):
      # Check if frame_0003.jpg exists
      thumbnail = item / 'frame_0003.jpg'
      if thumbnail.exists():
        info = parse_detection_name(item.name)
        if info:
          info['mtime'] = item.stat().st_mtime
          detections.append(info)

  # Sort by modification time, newest first
  detections.sort(key=lambda x: x['mtime'], reverse=True)

  with detections_lock:
    known_detections = detections


class DetectionHandler(FileSystemEventHandler):
  """Handle filesystem events for new detection directories."""

  def __init__(self):
    self.pending_dirs = {}

  def on_created(self, event):
    """Called when a file or directory is created."""
    if event.is_directory and event.src_path.endswith('_frames'):
      # New detection directory created
      dir_path = Path(event.src_path)
      dirname = dir_path.name

      # Start a thread to wait for frame_0003.jpg
      thread = threading.Thread(
        target=self._wait_for_thumbnail,
        args=(dir_path, dirname)
      )
      thread.daemon = True
      thread.start()

  def _wait_for_thumbnail(self, dir_path, dirname):
    """Wait for frame_0003.jpg to appear, then notify clients."""
    thumbnail_path = dir_path / 'frame_0003.jpg'

    # Wait up to 5 seconds for the thumbnail
    for _ in range(50):
      if thumbnail_path.exists():
        # Parse detection info
        info = parse_detection_name(dirname)
        if info:
          info['mtime'] = time.time()

          # Add to known detections
          with detections_lock:
            known_detections.insert(0, info)

          # Send SSE event
          event_queue.put(info)
        return
      time.sleep(0.1)


def start_watcher():
  """Start the directory watcher in a background thread."""
  observer = Observer()
  handler = DetectionHandler()
  observer.schedule(handler, CLIPS_DIR, recursive=False)
  observer.start()
  return observer


@app.route('/')
def index():
  """Serve the main viewer page."""
  return render_template('viewer.html')


@app.route('/events')
def events():
  """SSE endpoint for real-time detection updates."""
  def generate():
    # Send initial keepalive
    yield 'data: {"type": "connected"}\n\n'

    while True:
      try:
        # Wait for new events with timeout for keepalive
        info = event_queue.get(timeout=30)
        data = json.dumps({
          'type': 'new_detection',
          'detection': info
        })
        yield f'data: {data}\n\n'
      except queue.Empty:
        # Send keepalive
        yield 'data: {"type": "keepalive"}\n\n'

  return Response(
    generate(),
    mimetype='text/event-stream',
    headers={
      'Cache-Control': 'no-cache',
      'Connection': 'keep-alive',
      'X-Accel-Buffering': 'no'
    }
  )


@app.route('/detections')
def get_detections():
  """Return JSON list of all known detections."""
  with detections_lock:
    return jsonify(known_detections)


@app.route('/frames/<detection_name>')
def get_frames(detection_name):
  """Return JSON list of all frames for a detection."""
  frames_dir = Path(CLIPS_DIR) / f"{detection_name}_frames"

  if not frames_dir.exists():
    abort(404)

  # Security check - ensure we're still within CLIPS_DIR
  try:
    frames_dir.resolve().relative_to(Path(CLIPS_DIR).resolve())
  except ValueError:
    abort(403)

  frames = []
  for item in sorted(frames_dir.iterdir()):
    if item.suffix == '.jpg':
      frames.append({
        'name': item.stem,
        'path': f"{detection_name}_frames/{item.name}"
      })

  return jsonify(frames)


@app.route('/image/<path:filepath>')
def serve_image(filepath):
  """Serve images from CLIPS_DIR."""
  full_path = Path(CLIPS_DIR) / filepath

  # Security check - ensure we're still within CLIPS_DIR
  try:
    full_path.resolve().relative_to(Path(CLIPS_DIR).resolve())
  except ValueError:
    abort(403)

  if not full_path.exists():
    abort(404)

  # Read and serve the image
  with open(full_path, 'rb') as f:
    image_data = f.read()

  return Response(image_data, mimetype='image/jpeg')


def main():
  """Main entry point."""
  print(f"Detection Viewer")
  print(f"================")
  print(f"Monitoring: {CLIPS_DIR}")

  # Ensure clips directory exists
  os.makedirs(CLIPS_DIR, exist_ok=True)

  # Scan for existing detections
  print("Scanning for existing detections...")
  scan_existing_detections()
  print(f"Found {len(known_detections)} existing detections")

  # Start directory watcher
  print("Starting directory watcher...")
  observer = start_watcher()

  # Start Flask server
  print(f"\nStarting web server on http://0.0.0.0:8080")
  print("Press Ctrl+C to stop\n")

  try:
    app.run(host='0.0.0.0', port=8080, threaded=True)
  finally:
    observer.stop()
    observer.join()


if __name__ == '__main__':
  main()

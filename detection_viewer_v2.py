#!/usr/bin/env python3
"""
Detection Viewer V2 - Web-based detection viewer for FiftyOne-compatible structure.

Handles the new directory structure with images/ and labels/ subdirectories
within date-based folders (YYYY-MM-DD).

Directory structure:
  CLIPS_DIR/
  └── 2026-01-22/
      ├── images/
      │   ├── car_2026-01-22T14-30-45_0001.jpg
      │   └── car_2026-01-22T14-30-45_0002.jpg
      └── labels/
          ├── car_2026-01-22T14-30-45_0001.txt
          └── car_2026-01-22T14-30-45_0002.txt

Usage:
  python detection_viewer_v2.py
  Then open http://localhost:8081 in a browser
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

# Set of client queues for SSE broadcast
client_queues = set()
clients_lock = threading.Lock()

# Track known detections
known_detections = []
detections_lock = threading.Lock()


def broadcast_event(info):
  """Send an event to all connected clients."""
  with clients_lock:
    for q in client_queues:
      try:
        q.put_nowait(info)
      except queue.Full:
        pass  # Skip clients with full queues


def is_date_directory(name):
  """Check if directory name matches YYYY-MM-DD format."""
  return bool(re.match(r'^\d{4}-\d{2}-\d{2}$', name))


def parse_image_filename(filename):
  """
  Parse image filename to extract detection info.

  Args:
    filename: Filename like 'car_2026-01-22T14-30-45_0001.jpg'

  Returns:
    Dict with 'clip_name', 'vehicle_type', 'timestamp', 'frame_number' or None if invalid
  """
  # Remove extension
  stem = Path(filename).stem

  # Try to parse format: vehicle_YYYY-MM-DDTHH-MM-SS_NNNN
  match = re.match(r'^(\w+)_(\d{4}-\d{2}-\d{2}T\d{2}-\d{2}-\d{2})_(\d+)$', stem)
  if match:
    vehicle_type = match.group(1)
    timestamp = match.group(2)
    frame_number = int(match.group(3))
    clip_name = f"{vehicle_type}_{timestamp}"
    return {
      'clip_name': clip_name,
      'vehicle_type': vehicle_type,
      'timestamp': timestamp,
      'frame_number': frame_number
    }

  return None


def scan_existing_detections():
  """Scan CLIPS_DIR for existing detections in the new structure."""
  global known_detections

  detections_dict = {}  # clip_name -> detection info
  clips_path = Path(CLIPS_DIR)

  if not clips_path.exists():
    return

  for date_dir in clips_path.iterdir():
    if not date_dir.is_dir() or not is_date_directory(date_dir.name):
      continue

    images_dir = date_dir / 'images'
    if not images_dir.exists():
      continue

    for image_file in images_dir.iterdir():
      if image_file.suffix.lower() != '.jpg':
        continue

      parsed = parse_image_filename(image_file.name)
      if not parsed:
        continue

      clip_name = parsed['clip_name']

      if clip_name not in detections_dict:
        # Format display time from timestamp
        ts = parsed['timestamp']
        date_part = ts[:10]
        time_part = ts[11:].replace('-', ':')
        display_time = f"{date_part} {time_part}"

        detections_dict[clip_name] = {
          'name': clip_name,
          'vehicle_type': parsed['vehicle_type'],
          'timestamp': ts,
          'display_time': display_time,
          'date_dir': date_dir.name,
          'frame_count': 0,
          'mtime': image_file.stat().st_mtime
        }

      detections_dict[clip_name]['frame_count'] += 1
      # Update mtime to latest frame
      mtime = image_file.stat().st_mtime
      if mtime > detections_dict[clip_name]['mtime']:
        detections_dict[clip_name]['mtime'] = mtime

  # Convert to list and sort by modification time, newest first
  detections = list(detections_dict.values())
  detections.sort(key=lambda x: x['mtime'], reverse=True)

  with detections_lock:
    known_detections = detections


class DetectionHandler(FileSystemEventHandler):
  """Handle filesystem events for new detection images."""

  def __init__(self):
    self.pending_clips = {}
    self.pending_lock = threading.Lock()

  def on_created(self, event):
    """Called when a file is created."""
    if event.is_directory:
      return

    file_path = Path(event.src_path)

    # Only handle .jpg files in images/ directories
    if file_path.suffix.lower() != '.jpg':
      return
    if file_path.parent.name != 'images':
      return

    parsed = parse_image_filename(file_path.name)
    if not parsed:
      return

    clip_name = parsed['clip_name']
    date_dir = file_path.parent.parent.name

    # Check if we already know about this clip
    with detections_lock:
      if any(d['name'] == clip_name for d in known_detections):
        return

    # Check if we're already processing this clip
    with self.pending_lock:
      if clip_name in self.pending_clips:
        return
      self.pending_clips[clip_name] = True

    # Start a thread to wait for the thumbnail and then notify
    thread = threading.Thread(
      target=self._wait_for_thumbnail,
      args=(clip_name, parsed, date_dir, file_path.parent)
    )
    thread.daemon = True
    thread.start()

  def _wait_for_thumbnail(self, clip_name, parsed, date_dir, images_dir):
    """Wait for frame 3 (thumbnail) to appear, then notify clients."""
    thumbnail_path = images_dir / f"{clip_name}_0003.jpg"

    # Wait up to 10 seconds for the thumbnail
    for _ in range(100):
      if thumbnail_path.exists():
        self._handle_new_detection(clip_name, parsed, date_dir)
        break
      time.sleep(0.1)

    # Remove from pending
    with self.pending_lock:
      self.pending_clips.pop(clip_name, None)

  def _handle_new_detection(self, clip_name, parsed, date_dir):
    """Handle a new detection when thumbnail is ready."""
    # Format display time from timestamp
    ts = parsed['timestamp']
    date_part = ts[:10]
    time_part = ts[11:].replace('-', ':')
    display_time = f"{date_part} {time_part}"

    info = {
      'name': clip_name,
      'vehicle_type': parsed['vehicle_type'],
      'timestamp': ts,
      'display_time': display_time,
      'date_dir': date_dir,
      'frame_count': 3,
      'mtime': time.time()
    }

    # Add to known detections (skip if already exists)
    with detections_lock:
      if not any(d['name'] == clip_name for d in known_detections):
        known_detections.insert(0, info)
        # Broadcast to all connected clients
        broadcast_event(info)


def start_watcher():
  """Start the directory watcher in a background thread."""
  observer = Observer()
  handler = DetectionHandler()
  observer.schedule(handler, CLIPS_DIR, recursive=True)
  observer.start()
  return observer


def poll_for_new_detections():
  """Periodically scan for new detections (fallback for network filesystems)."""
  while True:
    time.sleep(5)  # Poll every 5 seconds
    try:
      clips_path = Path(CLIPS_DIR)
      if not clips_path.exists():
        continue

      for date_dir in clips_path.iterdir():
        if not date_dir.is_dir() or not is_date_directory(date_dir.name):
          continue

        images_dir = date_dir / 'images'
        if not images_dir.exists():
          continue

        # Group files by clip name
        clip_files = {}
        for image_file in images_dir.iterdir():
          if image_file.suffix.lower() != '.jpg':
            continue
          parsed = parse_image_filename(image_file.name)
          if not parsed:
            continue
          clip_name = parsed['clip_name']
          if clip_name not in clip_files:
            clip_files[clip_name] = {'parsed': parsed, 'count': 0, 'mtime': 0}
          clip_files[clip_name]['count'] += 1
          mtime = image_file.stat().st_mtime
          if mtime > clip_files[clip_name]['mtime']:
            clip_files[clip_name]['mtime'] = mtime

        # Check for new clips with at least 3 frames
        for clip_name, data in clip_files.items():
          if data['count'] < 3:
            continue

          # Check if thumbnail exists
          thumbnail = images_dir / f"{clip_name}_0003.jpg"
          if not thumbnail.exists():
            continue

          with detections_lock:
            if any(d['name'] == clip_name for d in known_detections):
              continue

            # New detection found
            parsed = data['parsed']
            ts = parsed['timestamp']
            date_part = ts[:10]
            time_part = ts[11:].replace('-', ':')
            display_time = f"{date_part} {time_part}"

            info = {
              'name': clip_name,
              'vehicle_type': parsed['vehicle_type'],
              'timestamp': ts,
              'display_time': display_time,
              'date_dir': date_dir.name,
              'frame_count': data['count'],
              'mtime': data['mtime']
            }

            known_detections.insert(0, info)
            broadcast_event(info)

    except Exception as e:
      print(f"Polling error: {e}")


def start_polling():
  """Start the polling thread for network filesystem support."""
  thread = threading.Thread(target=poll_for_new_detections, daemon=True)
  thread.start()
  return thread


@app.route('/')
def index():
  """Serve the main viewer page."""
  return render_template('viewer_v2.html')


@app.route('/events')
def events():
  """SSE endpoint for real-time detection updates."""
  # Create a queue for this client
  client_queue = queue.Queue(maxsize=100)

  # Register client
  with clients_lock:
    client_queues.add(client_queue)

  def generate():
    try:
      # Send initial keepalive
      yield 'data: {"type": "connected"}\n\n'

      while True:
        try:
          # Wait for new events with timeout for keepalive
          info = client_queue.get(timeout=30)
          data = json.dumps({
            'type': 'new_detection',
            'detection': info
          })
          yield f'data: {data}\n\n'
        except queue.Empty:
          # Send keepalive
          yield 'data: {"type": "keepalive"}\n\n'
    finally:
      # Unregister client on disconnect
      with clients_lock:
        client_queues.discard(client_queue)

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
  clips_path = Path(CLIPS_DIR)

  # Look up the detection to get date_dir
  date_dir = None
  with detections_lock:
    for det in known_detections:
      if det['name'] == detection_name:
        date_dir = det.get('date_dir')
        break

  if not date_dir:
    abort(404)

  images_dir = clips_path / date_dir / 'images'

  if not images_dir.exists():
    abort(404)

  # Security check - ensure we're still within CLIPS_DIR
  try:
    images_dir.resolve().relative_to(clips_path.resolve())
  except ValueError:
    abort(403)

  # Find all frames for this detection
  frames = []
  prefix = f"{detection_name}_"

  for item in sorted(images_dir.iterdir()):
    if item.suffix.lower() == '.jpg' and item.name.startswith(prefix):
      parsed = parse_image_filename(item.name)
      if parsed:
        frames.append({
          'name': item.stem,
          'path': f"{date_dir}/images/{item.name}",
          'frame_number': parsed['frame_number']
        })

  # Sort by frame number
  frames.sort(key=lambda x: x['frame_number'])

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
  print(f"Detection Viewer V2 (FiftyOne-compatible structure)")
  print(f"===================================================")
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

  # Start polling thread (fallback for network filesystems)
  print("Starting polling thread...")
  start_polling()

  # Start Flask server on port 8081 (different from v1)
  print(f"\nStarting web server on http://0.0.0.0:8081")
  print("Press Ctrl+C to stop\n")

  try:
    app.run(host='0.0.0.0', port=8081, threaded=True)
  finally:
    observer.stop()
    observer.join()


if __name__ == '__main__':
  main()

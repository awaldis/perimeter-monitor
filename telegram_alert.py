"""
Telegram alert functionality for vehicle detection notifications.
"""

import requests
import cv2
import tempfile
import os
from datetime import datetime


class TelegramAlerter:
  """
  Sends Telegram alerts when specific vehicle types are detected.
  Tracks which vehicles have already triggered alerts to avoid duplicates.
  """

  def __init__(self, bot_token, chat_id, alert_classes, class_names):
    """
    Initialize the Telegram alerter.

    Args:
      bot_token: Telegram bot token
      chat_id: Telegram chat ID to send messages to
      alert_classes: List of class IDs that should trigger alerts
      class_names: Dict mapping class ID to name
    """
    self.bot_token = bot_token
    self.chat_id = chat_id
    self.alert_classes = set(alert_classes)
    self.class_names = class_names

    # Track which (track_id, class_name) pairs have already sent alerts
    # This allows re-alerting if the same track_id changes to a different alert class
    self.alerted_tracks = set()

  def is_enabled(self):
    """Check if Telegram alerting is enabled."""
    return self.bot_token is not None and self.chat_id is not None

  def send_startup_message(self):
    """Send a message indicating the monitor has started."""
    if not self.is_enabled():
      return

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    alert_names = [self.class_names.get(c, f"class_{c}") for c in self.alert_classes]
    message = f"🚗 Perimeter Monitor Started\n⏰ {timestamp}\n📋 Alerting on: {', '.join(alert_names)}"

    self._send_message(message)

  def check_and_alert(self, detections_by_class, frame, class_ids_by_track):
    """
    Check detections and send alerts for new alertable vehicles.

    Args:
      detections_by_class: Dict mapping class_name -> list of track_ids
      frame: The current frame (will be saved as image for alert)
      class_ids_by_track: Dict mapping track_id -> class_id

    Returns:
      List of (class_name, track_id) tuples that triggered alerts
    """
    if not self.is_enabled():
      return []

    alerts_sent = []

    for class_name, track_ids in detections_by_class.items():
      for track_id in track_ids:
        # Get the class_id for this track
        class_id = class_ids_by_track.get(track_id)
        if class_id is None:
          continue

        # Check if this class should trigger alerts
        if class_id not in self.alert_classes:
          continue

        # Check if we've already alerted for this (track_id, class_name) pair
        alert_key = (track_id, class_name)
        if alert_key in self.alerted_tracks:
          continue

        # Send alert
        self._send_vehicle_alert(class_name, track_id, frame)
        self.alerted_tracks.add(alert_key)
        alerts_sent.append((class_name, track_id))

    return alerts_sent

  def _send_vehicle_alert(self, class_name, track_id, frame):
    """Send an alert for a detected vehicle with image."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    message = f"🚚 {class_name.upper()} Detected!\n⏰ {timestamp}\n🔢 Track ID: {track_id}"

    # Save frame to temporary file
    with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp:
      tmp_path = tmp.name
      cv2.imwrite(tmp_path, frame)

    try:
      self._send_photo(message, tmp_path)
    finally:
      # Clean up temp file
      if os.path.exists(tmp_path):
        os.remove(tmp_path)

  def _send_message(self, text):
    """Send a text-only message to Telegram."""
    try:
      url = f"https://api.telegram.org/bot{self.bot_token}/sendMessage"
      data = {"chat_id": self.chat_id, "text": text}
      response = requests.post(url, data=data, timeout=10)
      response.raise_for_status()
      print(f"Telegram alert sent: {text[:50]}...")
    except Exception as e:
      print(f"Failed to send Telegram message: {e}")

  def _send_photo(self, caption, image_path):
    """Send a photo with caption to Telegram."""
    try:
      url = f"https://api.telegram.org/bot{self.bot_token}/sendPhoto"
      with open(image_path, "rb") as image_file:
        files = {"photo": image_file}
        data = {"chat_id": self.chat_id, "caption": caption}
        response = requests.post(url, files=files, data=data, timeout=30)
      response.raise_for_status()
      print(f"Telegram photo alert sent: {caption[:50]}...")
    except Exception as e:
      print(f"Failed to send Telegram photo: {e}")

  def clear_stale_tracks(self, active_track_ids):
    """
    Remove alerts for tracks that are no longer being tracked.
    Call periodically to prevent memory buildup.

    Args:
      active_track_ids: Set of currently active track IDs
    """
    self.alerted_tracks = {
      (track_id, class_name)
      for track_id, class_name in self.alerted_tracks
      if track_id in active_track_ids
    }

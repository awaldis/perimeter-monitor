import cv2
import argparse
import time

def main():
  # Parse command-line arguments
  parser = argparse.ArgumentParser(description='Connect to and display an RTSP stream')
  parser.add_argument('rtsp_url', help='RTSP URL (e.g., rtsp://username:password@ip:port/stream)')
  parser.add_argument('-t', '--transport', choices=['tcp', 'udp'], default='tcp',
                      help='Transport protocol (default: tcp)')
  args = parser.parse_args()

  rtsp_url = args.rtsp_url

  print(f"Connecting to RTSP stream: {rtsp_url}")
  print(f"Transport protocol: {args.transport}")

  # Configure OpenCV to use the specified transport protocol
  if args.transport == 'tcp':
    # Force TCP transport (more reliable, especially over network)
    cap = cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Minimize latency
    # Set RTSP transport to TCP
    import os
    os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"
  else:
    # Use UDP transport (default, lower latency but less reliable)
    cap = cv2.VideoCapture(rtsp_url)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

  if not cap.isOpened():
    print(f"Error: Could not connect to RTSP stream {rtsp_url}")
    print("Please check:")
    print("  - The URL is correct")
    print("  - The camera/server is reachable")
    print("  - Username and password are correct (if required)")
    print("  - Firewall settings allow RTSP traffic")
    return

  # Get stream properties
  width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
  height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
  fps = int(cap.get(cv2.CAP_PROP_FPS))

  print(f"Connected successfully!")
  print(f"Stream resolution: {width}x{height}")
  print(f"Stream FPS: {fps if fps > 0 else 'Unknown'}")
  print("\nPress 'q' to quit, 's' to save a snapshot")

  frame_count = 0
  fps_start_time = time.time()
  fps_counter = 0
  current_fps = 0.0

  try:
    while True:
      ret, frame = cap.read()

      if not ret:
        print("Warning: Failed to read frame from stream")
        # Try to reconnect
        time.sleep(1)
        continue

      frame_count += 1
      fps_counter += 1

      # Calculate actual FPS every 30 frames
      if fps_counter >= 30:
        fps_end_time = time.time()
        current_fps = fps_counter / (fps_end_time - fps_start_time)
        fps_counter = 0
        fps_start_time = fps_end_time

      # Add FPS overlay to the frame
      if current_fps > 0:
        cv2.putText(frame, f"FPS: {current_fps:.1f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

      # Display the frame
      cv2.imshow('RTSP Stream Viewer', frame)

      # Handle keyboard input (1ms wait)
      key = cv2.waitKey(1) & 0xFF
      if key == ord('q'):
        print("\nQuitting...")
        break
      elif key == ord('s'):
        # Save snapshot
        snapshot_filename = f"snapshot_{int(time.time())}.jpg"
        cv2.imwrite(snapshot_filename, frame)
        print(f"\nSnapshot saved: {snapshot_filename}")

  except KeyboardInterrupt:
    print("\n\nInterrupted by user")

  finally:
    cap.release()
    cv2.destroyAllWindows()
    print(f"Total frames received: {frame_count}")

if __name__ == "__main__":
  main()

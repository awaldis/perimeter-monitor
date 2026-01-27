#!/usr/bin/env python3
"""Export a YOLOv8 .pt model to OpenVINO format."""

import argparse
import sys
from ultralytics import YOLO


def main():
  parser = argparse.ArgumentParser(description="Export a YOLOv8 .pt model to OpenVINO format")
  parser.add_argument("model", help="Path to the .pt model file")
  parser.add_argument("--imgsz", type=int, nargs="+", default=[480, 1600],
                      help="Input image size [height, width] (default: 480 1600)")
  parser.add_argument("--half", action="store_true", help="Export with FP16 precision")
  args = parser.parse_args()

  if not args.model.endswith(".pt"):
    print(f"Error: Expected a .pt file, got '{args.model}'")
    sys.exit(1)

  print(f"Loading model: {args.model}")
  model = YOLO(args.model)

  print(f"Exporting to OpenVINO format (imgsz={args.imgsz}, half={args.half})...")
  output_path = model.export(format="openvino", imgsz=args.imgsz, half=args.half)

  print(f"Done! OpenVINO model saved to: {output_path}")


if __name__ == "__main__":
  main()

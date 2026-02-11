"""
Add a background image (no annotations) to the FiftyOne dataset.

Usage from IPython:
  %run -i add_background.py /path/to/image.jpg
"""

import sys

if 'fo' not in dir():
    import fiftyone as fo

if len(sys.argv) < 2:
    raise SystemExit("Usage: %run -i add_background.py /path/to/image.jpg")

image_path = sys.argv[1]
sample = fo.Sample(filepath=image_path)
sample["ground_truth"] = fo.Detections()
sample.tags = ["background", "master_set"]
dataset.add_sample(sample)
print(f"Added {image_path} with tags: background, master_set")

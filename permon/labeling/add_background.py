"""
Add a background image (no annotations) to the FiftyOne dataset.

Usage from the labeling session:
  add_background(dataset, "/path/to/image.jpg")
"""

import fiftyone as fo


def add_background(dataset, image_path):
  """Add image_path to dataset as a master-set sample with no detections."""
  sample = fo.Sample(filepath=image_path)
  sample["ground_truth"] = fo.Detections()
  sample.tags = ["background", "master_set"]
  dataset.add_sample(sample)
  print(f"Added {image_path} with tags: background, master_set")

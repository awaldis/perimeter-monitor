"""
Interactive FiftyOne session for the Perimeter Monitor project.

Run with: ipython -i start_51.py

After startup, you'll have access to:
  - dataset: The master dataset with all labeled samples
  - session: The FiftyOne app session (accessible at http://localhost:8081)

To add new samples:
  dataset.add_dir("/path/to/new/data", dataset_type=fo.types.YOLOv5Dataset, tags=["new_import"])
"""

import fiftyone as fo

DATASET_NAME = "perim_mon_master"

# Load the master dataset
if fo.dataset_exists(DATASET_NAME):
  dataset = fo.load_dataset(DATASET_NAME)
  print(f"Loaded dataset '{DATASET_NAME}': {len(dataset)} samples")
else:
  print(f"Error: Dataset '{DATASET_NAME}' does not exist.")
  print("Available datasets:", fo.list_datasets())
  raise SystemExit(1)

# Launch FiftyOne app (accessible from any machine on the network)
session = fo.launch_app(dataset, address="0.0.0.0", port=8081)
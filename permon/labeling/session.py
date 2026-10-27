"""
Interactive FiftyOne session for the Perimeter Monitor project.

Run from the repo root with:
  ipython -i -m permon.labeling.session

After startup, you'll have access to:
  - dataset: The master dataset with all labeled samples
  - session: The FiftyOne app session (accessible at http://localhost:5151, FIFTYONE_PORT)
  - fo: the fiftyone module
  - add_background(dataset, image_path)
  - delete_non_master_samples(dataset)
  - export_for_training(dataset, export_dir, new_sample_view=None)

To add new samples:
  dataset.add_dir("/path/to/new/data", dataset_type=fo.types.YOLOv5Dataset, tags=["new_import"])
"""

import fiftyone as fo

from permon.config import FIFTYONE_PORT
from permon.labeling.add_background import add_background
from permon.labeling.delete_non_master_samples import delete_non_master_samples
from permon.labeling.export_for_training import export_for_training

DATASET_NAME = "perim_mon_master"


def load_dataset(name=DATASET_NAME):
  """Load the master dataset, exiting if it doesn't exist."""
  if not fo.dataset_exists(name):
    print(f"Error: Dataset '{name}' does not exist.")
    print("Available datasets:", fo.list_datasets())
    raise SystemExit(1)
  dataset = fo.load_dataset(name)
  print(f"Loaded dataset '{name}': {len(dataset)} samples")
  return dataset


def launch_app(dataset, port=FIFTYONE_PORT):
  """Launch the FiftyOne app (accessible from any machine on the network)."""
  return fo.launch_app(dataset, address="0.0.0.0", port=port)


if __name__ == "__main__":
  dataset = load_dataset()
  session = launch_app(dataset)

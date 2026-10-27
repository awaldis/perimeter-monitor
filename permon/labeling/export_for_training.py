"""
Export FiftyOne dataset for YOLO training.

Usage from the labeling session:
  export_for_training(dataset, "/path/to/training_set_2026_02_01")
  export_for_training(dataset, "/path/to/training_set_2026_02_01", new_sample_view=view)
"""

import os
import shutil

import fiftyone as fo
import fiftyone.utils.random as four
import yaml

from permon.config import CLASS_NAMES


def training_classes(class_names=CLASS_NAMES):
  """
  Class names in model output order, from CLASS_NAMES (class_id -> name).

  YOLO export numbers classes by list position, so the IDs must be 0..N-1.
  """
  ids = sorted(class_names)
  if ids != list(range(len(ids))):
    raise ValueError(
      f"CLASS_NAMES IDs must be 0..{len(ids) - 1} for a training export, got {ids}. "
      "Set CLASS_NAMES for the custom model in config_local.py."
    )
  return [class_names[i] for i in ids]


def export_for_training(dataset, export_dir, new_sample_view=None, classes=None):
  """
  Export the master set as a YOLOv5 dataset with a fresh 80/20 train/val
  split, then zip it for Colab upload.

  Args:
    dataset: FiftyOne dataset
    export_dir: Directory to export into; the zip is written next to it
    new_sample_view: Optional view of new samples to tag 'master_set' (and
      untag 'keep') before exporting
    classes: Class names in model output order (default: from CLASS_NAMES)

  Returns:
    Path to the zip file
  """
  if classes is None:
    classes = training_classes()

  # Tag new samples for inclusion in master set
  if new_sample_view is not None:
    new_sample_view.tag_samples("master_set")
    new_sample_view.untag_samples("keep")

  # Get all samples with ground_truth labels tagged as master_set
  master_set_view = dataset.exists("ground_truth").match_tags("master_set")

  # Clear existing splits and create new 80/20 train/val split
  master_set_view.untag_samples(["train", "val"])
  four.random_split(master_set_view, {"train": 0.8, "val": 0.2})

  train_view = dataset.match_tags("train")
  val_view = dataset.match_tags("val")

  print(f"Training: {len(train_view)}, Validation: {len(val_view)}")

  # Export train and val splits
  print(f"Exporting to: {export_dir}")

  for split, view in (("train", train_view), ("val", val_view)):
    view.export(
      export_dir=export_dir,
      dataset_type=fo.types.YOLOv5Dataset,
      label_field="ground_truth",
      split=split,
      classes=classes
    )

  # Fix dataset.yaml path for Colab compatibility
  yaml_path = os.path.join(export_dir, "dataset.yaml")
  with open(yaml_path, 'r') as f:
    config = yaml.safe_load(f)

  config['path'] = '.'

  with open(yaml_path, 'w') as f:
    yaml.dump(config, f, default_flow_style=False)

  print(f"Fixed {yaml_path}: set path to '.'")

  # Create zip file for Colab upload
  zip_path = shutil.make_archive(export_dir, 'zip', export_dir)
  print(f"Created zip: {zip_path}")
  print("Export complete!")
  return zip_path

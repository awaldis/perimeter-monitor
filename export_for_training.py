"""
Export FiftyOne dataset for YOLO training.

Before running, set EXPORT_DIR in your IPython session:
  EXPORT_DIR = "/path/to/training_set_2026_02_01"
  %run -i export_for_training.py

Requires: dataset, new_sample_view (from your FiftyOne session)
"""

if 'fo' not in dir():
  import fiftyone as fo
if 'four' not in dir():
  import fiftyone.utils.random as four

# Check that EXPORT_DIR is set
if 'EXPORT_DIR' not in dir():
  raise NameError("Set EXPORT_DIR before running: EXPORT_DIR = '/path/to/export'")

# Tag new samples for inclusion in master set
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

output_classes = ["car", "suv", "pickup_truck", "truck", "usmail_truck", "fedex_truck", "ups_truck", "amazon_truck"]

# Export train and val splits
print(f"Exporting to: {EXPORT_DIR}")

train_view.export(
  export_dir=EXPORT_DIR,
  dataset_type=fo.types.YOLOv5Dataset,
  label_field="ground_truth",
  split="train",
  classes=output_classes
)

val_view.export(
  export_dir=EXPORT_DIR,
  dataset_type=fo.types.YOLOv5Dataset,
  label_field="ground_truth",
  split="val",
  classes=output_classes
)

# Fix dataset.yaml path for Colab compatibility
import os
import yaml

yaml_path = os.path.join(EXPORT_DIR, "dataset.yaml")
with open(yaml_path, 'r') as f:
  config = yaml.safe_load(f)

config['path'] = '.'

with open(yaml_path, 'w') as f:
  yaml.dump(config, f, default_flow_style=False)

print(f"Fixed {yaml_path}: set path to '.'")

# Create zip file for Colab upload
import shutil
zip_path = shutil.make_archive(EXPORT_DIR, 'zip', EXPORT_DIR)
print(f"Created zip: {zip_path}")
print("Export complete!")

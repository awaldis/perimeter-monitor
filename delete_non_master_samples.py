"""
Delete all samples not tagged 'master_set' from a FiftyOne dataset.

Usage from IPython:
  %run -i delete_non_master_samples.py
"""


def delete_non_master_samples(dataset):
    samples_to_remove = dataset.match_tags("master_set", bool=False)
    count = len(samples_to_remove)
    if count == 0:
        print("No samples to delete.")
        return
    response = input(f"About to delete {count} samples not tagged 'master_set'. Continue? [y/N] ")
    if response.lower() == 'y':
        dataset.delete_samples(samples_to_remove)
        print(f"Deleted {count} samples.")
    else:
        print("Aborted.")


delete_non_master_samples(dataset)

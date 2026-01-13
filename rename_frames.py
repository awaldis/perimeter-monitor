#!/usr/bin/env python3
"""
Bulk rename frames in a detection directory.

Renames files starting with "frame" to use a custom label and the
date extracted from the directory name.

Usage:
  python rename_frames.py <directory_path> <label_string>

Example:
  python rename_frames.py /path/to/truck_2026-01-11T14-06-18_frames blue_suv

  This renames:
    frame_0021.jpg -> blue_suv_2026-01-11T14-06-18_0021.jpg
    frame_0021.txt -> blue_suv_2026-01-11T14-06-18_0021.txt
"""

import argparse
import os
import sys


def extract_date_string(dirname):
  """
  Extract the date string from a directory name.

  The date string is the part between the first and second underscores.
  Example: "truck_2026-01-11T14-06-18_frames" -> "2026-01-11T14-06-18"
  """
  parts = dirname.split('_')
  if len(parts) < 2:
    return None
  return parts[1]


def rename_frames(directory_path, label_string, dry_run=False):
  """
  Rename all files starting with 'frame' in the given directory.

  Args:
    directory_path: Path to the directory containing frames
    label_string: The label to use as prefix
    dry_run: If True, print what would be renamed without actually renaming

  Returns:
    Number of files renamed
  """
  # Validate directory exists
  if not os.path.isdir(directory_path):
    print(f"Error: Directory does not exist: {directory_path}")
    return 0

  # Extract directory name and date string
  dirname = os.path.basename(directory_path.rstrip('/'))
  date_string = extract_date_string(dirname)

  if not date_string:
    print(f"Error: Could not extract date string from directory name: {dirname}")
    print("Expected format: <type>_<date>_frames (e.g., truck_2026-01-11T14-06-18_frames)")
    return 0

  # Build the new prefix
  new_prefix = f"{label_string}_{date_string}"

  print(f"Directory: {directory_path}")
  print(f"Date string: {date_string}")
  print(f"New prefix: {new_prefix}")
  print()

  # Find and rename files
  renamed_count = 0
  for filename in sorted(os.listdir(directory_path)):
    if filename.startswith('frame'):
      # Replace 'frame' with the new prefix
      new_filename = filename.replace('frame', new_prefix, 1)

      old_path = os.path.join(directory_path, filename)
      new_path = os.path.join(directory_path, new_filename)

      if dry_run:
        print(f"  {filename} -> {new_filename}")
      else:
        os.rename(old_path, new_path)
        print(f"  Renamed: {filename} -> {new_filename}")

      renamed_count += 1

  return renamed_count


def main():
  parser = argparse.ArgumentParser(
    description='Bulk rename frames in a detection directory',
    formatter_class=argparse.RawDescriptionHelpFormatter,
    epilog="""
Example:
  python rename_frames.py /path/to/truck_2026-01-11T14-06-18_frames blue_suv

  This renames:
    frame_0021.jpg -> blue_suv_2026-01-11T14-06-18_0021.jpg
    frame_0021.txt -> blue_suv_2026-01-11T14-06-18_0021.txt
    """
  )
  parser.add_argument('directory', help='Path to the frames directory')
  parser.add_argument('label', help='Label string to use as prefix')
  parser.add_argument(
    '--dry-run', '-n',
    action='store_true',
    help='Show what would be renamed without actually renaming'
  )

  args = parser.parse_args()

  if args.dry_run:
    print("DRY RUN - no files will be renamed\n")

  count = rename_frames(args.directory, args.label, dry_run=args.dry_run)

  if count == 0:
    print("No files were renamed")
    sys.exit(1)
  else:
    action = "would be renamed" if args.dry_run else "renamed"
    print(f"\n{count} file(s) {action}")


if __name__ == '__main__':
  main()

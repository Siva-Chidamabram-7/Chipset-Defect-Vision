"""
generate_regions.py — DEPRECATED
──────────────────────────────────
The SAM (Segment Anything Model) pipeline has been removed from this project.

WHY
───
The dataset is now sourced directly from Roboflow, which provides:
  • Pre-annotated images in YOLOv8 format
  • Consistent train / val splits
  • Version-controlled dataset exports

There is no longer a need to run SAM for automatic region proposals or to
manually annotate regions with scripts/annotate.py.

HOW TO GET YOUR DATASET
────────────────────────
1.  Open your Roboflow project at https://roboflow.com
2.  Click  Versions → Export Dataset
3.  Choose format:  YOLOv8
4.  Download and unzip into:

        data/
        ├── images/
        │   ├── train/
        │   └── val/
        └── labels/
            ├── train/
            └── val/

5.  Train:
        python !training/train.py --data !training/data.yaml

DO NOT run this file.  It will exit immediately with an error.
"""

import sys

print(
    "[generate_regions] ERROR: This script is deprecated.\n"
    "                   SAM has been removed from the pipeline.\n"
    "\n"
    "                   Export your annotated dataset from Roboflow\n"
    "                   in YOLOv8 format and place the files under data/.\n"
    "\n"
    "                   Then run:\n"
    "                     python !training/train.py --data !training/data.yaml",
    file=sys.stderr,
)
sys.exit(1)

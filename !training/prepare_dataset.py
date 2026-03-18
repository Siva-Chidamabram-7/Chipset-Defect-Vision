"""
prepare_dataset.py — DEPRECATED
────────────────────────────────
This script has been removed from the active pipeline.

WHY
───
The dataset is now sourced directly from Roboflow, which handles:
  • Image collection and versioning
  • Annotation in YOLO format
  • Train / val / test splitting
  • Dataset augmentation configuration

There is no longer a need to manually split a flat raw_data/ directory.

HOW TO GET YOUR DATASET
────────────────────────
1.  Go to https://roboflow.com and open your PCB defect project.
2.  Click  Versions → Export Dataset.
3.  Select format:  YOLOv8
4.  Download the zip and extract it so the layout matches:

        data/
        ├── images/
        │   ├── train/
        │   └── val/
        └── labels/
            ├── train/
            └── val/

5.  Run training:
        python !training/train.py --data !training/data.yaml

DO NOT run this file.  It will exit immediately with an error.
"""

import sys

print(
    "[prepare_dataset] ERROR: This script is deprecated.\n"
    "                  The dataset is now sourced from Roboflow.\n"
    "\n"
    "                  Export your dataset from Roboflow in YOLOv8 format\n"
    "                  and place the images/labels under data/images/ and\n"
    "                  data/labels/ (train/ and val/ subdirectories).\n"
    "\n"
    "                  Then run:\n"
    "                    python !training/train.py --data !training/data.yaml",
    file=sys.stderr,
)
sys.exit(1)

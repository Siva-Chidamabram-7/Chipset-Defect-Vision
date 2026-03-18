"""
prepare_dataset.py — split a flat annotated dataset into YOLO train/val splits
────────────────────────────────────────────────────────────────────────────────
OFFLINE-FIRST: this script works entirely on local files.

Workflow
────────
1. Label images with LabelImg (see README § Dataset Preparation).
   LabelImg saves annotations into raw_data/labels/ as YOLO .txt files.

2. Run this script to split raw_data/ into the canonical data/ structure
   that training/data.yaml and train.py expect:

       python training/prepare_dataset.py

   Or with explicit options:

       python training/prepare_dataset.py \\
           --src raw_data \\
           --dst data \\
           --train 0.80 --val 0.20

3. Train:
       python training/train.py --epochs 30 --imgsz 640

Expected input (--src):
    raw_data/
    ├── images/   ← .jpg / .png files
    └── labels/   ← matching YOLO .txt annotation files (same stem as image)

Output (--dst):
    data/
    ├── images/
    │   ├── train/
    │   └── val/
    └── labels/
        ├── train/
        └── val/

YOLO label format (one object per line):
    <class_id> <x_center> <y_center> <width> <height>   (normalised 0–1)

Classes:
    0 → good
    1 → defect
"""

import argparse
import random
import shutil
import sys
from pathlib import Path


# ── Project root (two levels up from training/prepare_dataset.py) ─────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent


def parse_args():
    p = argparse.ArgumentParser(
        description="Split a flat annotated dataset into YOLO train/val splits.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--src",
        default="raw_data",
        help="Source directory (contains images/ and labels/ sub-dirs).",
    )
    p.add_argument(
        "--dst",
        default="data",
        help="Destination directory (will be created if absent).",
    )
    p.add_argument("--train", type=float, default=0.80, help="Fraction for training.")
    p.add_argument("--val",   type=float, default=0.20, help="Fraction for validation.")
    p.add_argument("--seed",  type=int,   default=42,   help="Random seed.")
    return p.parse_args()


def split_dataset(args):
    if abs(args.train + args.val - 1.0) > 1e-6:
        print(
            f"[Prepare] ERROR: --train ({args.train}) + --val ({args.val}) "
            f"must equal 1.0",
            file=sys.stderr,
        )
        sys.exit(1)

    src = Path(args.src)
    dst = Path(args.dst)

    # Resolve relative paths against project root
    if not src.is_absolute():
        src = PROJECT_ROOT / src
    if not dst.is_absolute():
        dst = PROJECT_ROOT / dst

    img_src = src / "images"
    lbl_src = src / "labels"

    # ── Validate source ───────────────────────────────────────────────────────
    if not img_src.exists():
        print(
            f"[Prepare] ERROR: images directory not found: {img_src}\n"
            f"          Label your images with LabelImg first (see README).",
            file=sys.stderr,
        )
        sys.exit(1)

    img_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    images = sorted([p for p in img_src.iterdir() if p.suffix.lower() in img_exts])

    if not images:
        print(
            f"[Prepare] ERROR: No images found in {img_src}",
            file=sys.stderr,
        )
        sys.exit(1)

    # ── Split ─────────────────────────────────────────────────────────────────
    random.seed(args.seed)
    random.shuffle(images)

    n       = len(images)
    n_val   = max(1, int(n * args.val))
    n_train = n - n_val

    splits = {
        "train": images[:n_train],
        "val":   images[n_train:],
    }

    # ── Copy files ────────────────────────────────────────────────────────────
    unlabelled = 0
    for split, files in splits.items():
        img_dst = dst / "images" / split
        lbl_dst = dst / "labels" / split
        img_dst.mkdir(parents=True, exist_ok=True)
        lbl_dst.mkdir(parents=True, exist_ok=True)

        for img_path in files:
            shutil.copy2(img_path, img_dst / img_path.name)

            lbl_path = lbl_src / (img_path.stem + ".txt")
            if lbl_path.exists():
                shutil.copy2(lbl_path, lbl_dst / lbl_path.name)
            else:
                # Create empty label file so YOLO doesn't error on missing files.
                # An empty label means this image has no annotated objects
                # (treated as a background / negative example).
                (lbl_dst / (img_path.stem + ".txt")).touch()
                unlabelled += 1

        print(f"[Prepare] {split:5s}: {len(files):4d} images → {img_dst}")

    print(
        f"\n[Prepare] Total: {n} images  "
        f"train={n_train}  val={n_val}"
    )
    if unlabelled:
        print(
            f"[Prepare] WARNING: {unlabelled} image(s) had no matching label file "
            f"and were created as empty (background) examples."
        )

    print(
        f"\n[Prepare] Dataset ready at: {dst}\n"
        f"[Prepare] Next step: python training/train.py --epochs 30 --imgsz 640"
    )


if __name__ == "__main__":
    split_dataset(parse_args())

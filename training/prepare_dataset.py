"""
prepare_dataset.py
───────────────────
Helper to split a flat annotated dataset into YOLO train/val/test splits.

Expected input structure:
    raw_data/
        images/   ← .jpg / .png files
        labels/   ← matching .txt YOLO-format annotation files

Output (written to datasets/pcb_solder/):
    images/train/, images/val/, images/test/
    labels/train/, labels/val/, labels/test/

Usage:
    python training/prepare_dataset.py \
        --src raw_data \
        --dst datasets/pcb_solder \
        --train 0.80 --val 0.15 --test 0.05
"""

import argparse
import random
import shutil
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--src",   default="raw_data",           help="Source directory")
    p.add_argument("--dst",   default="datasets/pcb_solder",help="Destination directory")
    p.add_argument("--train", type=float, default=0.80)
    p.add_argument("--val",   type=float, default=0.15)
    p.add_argument("--test",  type=float, default=0.05)
    p.add_argument("--seed",  type=int,   default=42)
    return p.parse_args()


def split_dataset(args):
    assert abs(args.train + args.val + args.test - 1.0) < 1e-6, "Splits must sum to 1.0"

    src      = Path(args.src)
    dst      = Path(args.dst)
    img_src  = src / "images"
    lbl_src  = src / "labels"

    extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    images = sorted([p for p in img_src.iterdir() if p.suffix.lower() in extensions])

    if not images:
        print(f"[Prepare] No images found in {img_src}")
        return

    random.seed(args.seed)
    random.shuffle(images)

    n      = len(images)
    n_val  = int(n * args.val)
    n_test = int(n * args.test)
    n_train= n - n_val - n_test

    splits = {
        "train": images[:n_train],
        "val":   images[n_train: n_train + n_val],
        "test":  images[n_train + n_val:],
    }

    for split, files in splits.items():
        (dst / "images" / split).mkdir(parents=True, exist_ok=True)
        (dst / "labels" / split).mkdir(parents=True, exist_ok=True)

        for img_path in files:
            lbl_path = lbl_src / (img_path.stem + ".txt")
            shutil.copy2(img_path, dst / "images" / split / img_path.name)
            if lbl_path.exists():
                shutil.copy2(lbl_path, dst / "labels" / split / lbl_path.name)
            else:
                # create empty label (background image)
                (dst / "labels" / split / (img_path.stem + ".txt")).touch()

        print(f"[Prepare] {split:5s}: {len(files):4d} images → {dst}/images/{split}/")

    print(f"\n[Prepare] Total: {n} images  |  train={n_train}  val={n_val}  test={n_test}")


if __name__ == "__main__":
    split_dataset(parse_args())

#!/usr/bin/env python3
"""
auto_label_with_yolo.py — Step 3 of the SAM → YOLO bootstrap pipeline
───────────────────────────────────────────────────────────────────────
After an initial training round, this script replaces the SAM-generated
labels with higher-quality YOLO predictions.  Re-training on the improved
labels typically yields a significantly better model.

Bootstrap loop (zero manual annotation required):
    Step 1  python scripts/sam_auto_annotate.py       # SAM labels → data/
    Step 2  python !training/train.py                 # train YOLO on SAM labels
    Step 3  python scripts/auto_label_with_yolo.py    # replace labels with YOLO
    Step 4  python !training/train.py                 # retrain on improved labels
    Step 5  (repeat Steps 3–4 as many times as desired)

How it works:
    • Scans data/images/train/ and data/images/val/ for all images.
    • Runs trained YOLO inference on each image.
    • Keeps only detections above --conf threshold.
    • Writes new YOLO-format labels to the matching data/labels/<split>/ path,
      overwriting whatever was there before (SAM or a previous YOLO pass).
    • Optional: --keep-undetected preserves the previous label when YOLO
      finds nothing (useful early in training when the model is immature).

Usage (from project root):
    python scripts/auto_label_with_yolo.py
    python scripts/auto_label_with_yolo.py --weights weights/best.pt --conf 0.3
    python scripts/auto_label_with_yolo.py --data data/ --keep-undetected

Requirements:
    pip install -r requirements.txt        (ultralytics is included)
    weights/best.pt must exist             (run !training/train.py first)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# ── Project root ──────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


# ═════════════════════════════════════════════════════════════════════════════
# CLI
# ═════════════════════════════════════════════════════════════════════════════

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Step 3 — Replace SAM labels with YOLO-predicted labels.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--weights", default="weights/best.pt",
        help="Path to trained YOLO weights (.pt).",
    )
    p.add_argument(
        "--data", default="data",
        help=(
            "YOLO dataset root.  Must contain images/train/ and/or images/val/."
        ),
    )
    p.add_argument(
        "--conf", type=float, default=0.30,
        help="Minimum confidence to keep a detection.",
    )
    p.add_argument(
        "--device", default="cpu",
        help="Inference device: 'cpu', 'cuda', 'cuda:0', 'mps'.",
    )
    p.add_argument(
        "--keep-undetected", action="store_true",
        help=(
            "If YOLO finds no detections for an image, keep the existing label "
            "instead of overwriting with an empty file.  Useful when the model "
            "is still immature (early training iterations)."
        ),
    )
    return p.parse_args()


# ═════════════════════════════════════════════════════════════════════════════
# Path helpers
# ═════════════════════════════════════════════════════════════════════════════

def image_to_label_path(img_path: Path, data_dir: Path) -> Path:
    """
    Map a data/images/<split>/foo.jpg path to data/labels/<split>/foo.txt.

    Works by computing the relative path from data_dir, swapping the leading
    'images' segment to 'labels', then appending .txt suffix.
    """
    try:
        rel = img_path.relative_to(data_dir)   # e.g. images/train/foo.jpg
    except ValueError:
        # img_path is not under data_dir — shouldn't happen, but be safe
        return img_path.with_suffix(".txt")

    parts = list(rel.parts)
    if parts and parts[0] == "images":
        parts[0] = "labels"

    return (data_dir / Path(*parts)).with_suffix(".txt")


# ═════════════════════════════════════════════════════════════════════════════
# Label writing
# ═════════════════════════════════════════════════════════════════════════════

def write_labels(label_path: Path, detections: list[dict]) -> None:
    """
    Write YOLO-format label file.  An empty detections list → empty file
    (treated as background by YOLO trainer).
    """
    label_path.parent.mkdir(parents=True, exist_ok=True)
    with open(label_path, "w") as f:
        for det in detections:
            f.write(
                f"{det['cls']} "
                f"{det['cx']:.6f} {det['cy']:.6f} "
                f"{det['w']:.6f}  {det['h']:.6f}\n"
            )


# ═════════════════════════════════════════════════════════════════════════════
# Main
# ═════════════════════════════════════════════════════════════════════════════

def main() -> None:
    args = parse_args()

    weights_path = Path(args.weights)
    data_dir     = Path(args.data)

    if not weights_path.is_absolute():
        weights_path = PROJECT_ROOT / weights_path
    if not data_dir.is_absolute():
        data_dir = PROJECT_ROOT / data_dir

    # ── Validate inputs ───────────────────────────────────────────────────────
    if not weights_path.exists():
        print(
            f"[YOLO-Label] ERROR: Weights not found: {weights_path}\n"
            "             Run !training/train.py first to produce weights/best.pt.",
            file=sys.stderr,
        )
        sys.exit(1)

    # ── Collect all images from train + val splits ────────────────────────────
    img_paths: list[Path] = []
    for split in ("train", "val"):
        split_dir = data_dir / "images" / split
        if split_dir.exists():
            img_paths.extend(
                p for p in sorted(split_dir.iterdir())
                if p.suffix.lower() in IMAGE_EXTS
            )

    if not img_paths:
        print(
            f"[YOLO-Label] ERROR: No images found under {data_dir / 'images'}\n"
            "             Run scripts/sam_auto_annotate.py first.",
            file=sys.stderr,
        )
        sys.exit(1)

    n_train = sum(1 for p in img_paths if "train" in p.parts)
    n_val   = sum(1 for p in img_paths if "val"   in p.parts)

    print(f"[YOLO-Label] Weights            : {weights_path}")
    print(f"[YOLO-Label] Dataset            : {data_dir}")
    print(f"[YOLO-Label] Images             : {len(img_paths)}  "
          f"(train={n_train}, val={n_val})")
    print(f"[YOLO-Label] Conf threshold     : {args.conf}")
    print(f"[YOLO-Label] Device             : {args.device}")
    print(f"[YOLO-Label] Keep-undetected    : {args.keep_undetected}\n")

    # ── Load YOLO model ───────────────────────────────────────────────────────
    try:
        from ultralytics import YOLO
    except ImportError:
        print(
            "[YOLO-Label] ERROR: ultralytics not installed.\n"
            "             Install:  pip install -r requirements.txt",
            file=sys.stderr,
        )
        sys.exit(1)

    model = YOLO(str(weights_path))
    print(f"[YOLO-Label] Model loaded.  Running inference …\n")

    # ── Inference loop ────────────────────────────────────────────────────────
    replaced     = 0
    kept_sam     = 0
    total_dets   = 0
    empty_labels = 0

    for img_path in img_paths:
        label_path = image_to_label_path(img_path, data_dir)

        results = model.predict(
            source  = str(img_path),
            conf    = args.conf,
            device  = args.device,
            save    = False,
            verbose = False,
        )[0]

        # ── Extract detections above confidence threshold ─────────────────────
        detections: list[dict] = []
        if results.boxes is not None:
            img_h, img_w = results.orig_shape
            for box in results.boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                conf = float(box.conf[0])
                cls  = int(box.cls[0])

                # Convert pixel xyxy → normalised YOLO cx, cy, w, h
                bw = x2 - x1
                bh = y2 - y1
                cx = (x1 + bw / 2) / img_w
                cy = (y1 + bh / 2) / img_h
                nw = bw / img_w
                nh = bh / img_h

                detections.append({
                    "cls": cls,
                    "cx": cx, "cy": cy,
                    "w":  nw, "h":  nh,
                    "conf": conf,
                })

        # ── Handle zero-detection case ────────────────────────────────────────
        if not detections:
            if args.keep_undetected and label_path.exists():
                kept_sam += 1
                continue   # leave the existing label unchanged
            # Overwrite with empty label → background / negative example
            empty_labels += 1

        write_labels(label_path, detections)
        replaced   += 1
        total_dets += len(detections)

    # ── Summary ───────────────────────────────────────────────────────────────
    avg = total_dets / replaced if replaced > 0 else 0.0

    print(f"\n{'─'*60}")
    print(f"[YOLO-Label] Relabelling complete.")
    print(f"             Labels written      : {replaced}")
    print(f"             Total detections    : {total_dets}")
    print(f"             Avg detections/img  : {avg:.1f}")
    if empty_labels:
        print(f"             Empty (background)  : {empty_labels}")
    if kept_sam:
        print(f"             SAM labels kept     : {kept_sam}  (--keep-undetected)")
    print(f"\n[YOLO-Label] Next step:")
    print(f"             python !training/train.py --data !training/data.yaml")
    print(f"{'─'*60}\n")


if __name__ == "__main__":
    main()

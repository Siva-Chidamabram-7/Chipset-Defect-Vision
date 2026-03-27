#!/usr/bin/env python3
"""
sam_to_yolo.py — Convert raw PCB images into a YOLO training dataset using SAM.
────────────────────────────────────────────────────────────────────────────────
Reads class-labelled folders from raw_data/, runs SAM automatic mask generation
on each image, converts masks to YOLO-format bounding boxes, and writes the
final dataset to data/images/ and data/labels/.

INPUT:
    raw_data/
    ├── Missing_hole/
    ├── Mouse_bite/
    ├── Open_circuit/
    ├── Short/
    ├── Spur/
    └── Spurious_copper/

OUTPUT:
    data/
    ├── images/     ← copied images, prefixed with class name
    └── labels/     ← YOLO .txt label files

Usage:
    python scripts/sam_to_yolo.py
    python scripts/sam_to_yolo.py --input raw_data --output data
    python scripts/sam_to_yolo.py --checkpoint weights/sam_vit_b.pth
    python scripts/sam_to_yolo.py --input raw_data --output data \\
        --checkpoint weights/sam_vit_b.pth --min-px 10 --max-cover 0.9
"""

from __future__ import annotations

import argparse
import shutil
import sys
import traceback
from pathlib import Path

import cv2
import numpy as np

# ── Project root ──────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# ── Class map ─────────────────────────────────────────────────────────────────
CLASS_MAP: dict[str, int] = {
<<<<<<< HEAD
    "Short":           0,
=======
    "Missing_hole":    0,
    "Mouse_bite":      1,
    "Open_circuit":    2,
    "Short":           3,
    "Spur":            4,
    "Spurious_copper": 5,
    "Good":            6,
    "Solder_Defect":   7,
>>>>>>> b3b38bdc8568e3830d194c147d70e12a2d46a9e2
}

IMAGE_EXTS: set[str] = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


# ═════════════════════════════════════════════════════════════════════════════
# CLI
# ═════════════════════════════════════════════════════════════════════════════

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Convert raw PCB images to a YOLO dataset via SAM auto-annotation.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--input", default="raw_data",
        help="Root directory containing one sub-folder per defect class.",
    )
    p.add_argument(
        "--output", default="data",
        help="Output root directory.  images/ and labels/ will be created inside it.",
    )
    p.add_argument(
        "--checkpoint", default="weights/sam_vit_b.pth",
        help="Path to SAM ViT-B checkpoint (.pth).",
    )
    p.add_argument(
        "--min-px", type=int, default=10,
        help="Minimum bounding-box side length in pixels.  Smaller boxes are dropped.",
    )
    p.add_argument(
        "--max-cover", type=float, default=0.90,
        help="Maximum allowed box area as a fraction of total image area (0–1).",
    )
    p.add_argument(
        "--log-every", type=int, default=10,
        help="Print a progress line every N images.",
    )
    return p.parse_args()


# ═════════════════════════════════════════════════════════════════════════════
# SAM loader
# ═════════════════════════════════════════════════════════════════════════════

def load_sam(checkpoint: Path):
    """
    Load a SAM ViT-B model from a local checkpoint.
    Returns a SamAutomaticMaskGenerator ready for inference.
    Exits with a clear message if segment_anything is not installed.
    """
    try:
        from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
    except ImportError:
        print(
            "[ERROR] 'segment_anything' package not installed.\n"
            "        Install it:  pip install segment-anything",
            file=sys.stderr,
        )
        sys.exit(1)

    if not checkpoint.exists():
        print(
            f"[ERROR] SAM checkpoint not found: {checkpoint}\n"
            "\n"
            "  Download ViT-B (~375 MB) from:\n"
            "    https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth\n"
            "  Save it as:  weights/sam_vit_b.pth",
            file=sys.stderr,
        )
        sys.exit(1)

    print(f"[SAM] Loading ViT-B checkpoint: {checkpoint}")
    sam = sam_model_registry["vit_b"](checkpoint=str(checkpoint))
    sam.eval()

    mask_generator = SamAutomaticMaskGenerator(
        sam,
        points_per_side        = 16,
        pred_iou_thresh        = 0.88,
        stability_score_thresh = 0.92,
        crop_n_layers          = 0,
        min_mask_region_area   = 100,
    )
    print("[SAM] Model loaded successfully.\n")
    return mask_generator


# ═════════════════════════════════════════════════════════════════════════════
# Mask → YOLO conversion helpers
# ═════════════════════════════════════════════════════════════════════════════

def masks_to_yolo_boxes(
    masks: list[dict],
    img_h: int,
    img_w: int,
    min_px: int,
    max_cover: float,
) -> list[tuple[float, float, float, float]]:
    """
    Convert SAM mask dicts to normalised YOLO [cx, cy, w, h] tuples.

    Filtering rules applied per mask:
      1. Drop masks whose bounding box is narrower OR shorter than min_px pixels.
      2. Drop masks whose bounding box covers more than max_cover of the image.

    Returns a list of (cx, cy, w, h) tuples, all values in [0, 1].
    """
    img_area = img_h * img_w
    boxes: list[tuple[float, float, float, float]] = []

    for mask in masks:
        # SAM bbox: [x_topleft, y_topleft, width, height]  (pixel coordinates)
        x, y, bw, bh = mask["bbox"]

        # ── Filter 1: minimum side length ─────────────────────────────────────
        if bw < min_px or bh < min_px:
            continue

        # ── Filter 2: maximum coverage ────────────────────────────────────────
        if (bw * bh) / img_area > max_cover:
            continue

        # ── Normalise to YOLO format ───────────────────────────────────────────
        cx = (x + bw / 2.0) / img_w
        cy = (y + bh / 2.0) / img_h
        nw = bw / img_w
        nh = bh / img_h

        # Clamp to guard against sub-pixel rounding
        cx = min(max(cx, 0.0), 1.0)
        cy = min(max(cy, 0.0), 1.0)
        nw = min(max(nw, 0.0), 1.0)
        nh = min(max(nh, 0.0), 1.0)

        boxes.append((cx, cy, nw, nh))

    return boxes


def fallback_box() -> list[tuple[float, float, float, float]]:
    """Return a full-image bounding box in YOLO format."""
    return [(0.5, 0.5, 1.0, 1.0)]


# ═════════════════════════════════════════════════════════════════════════════
# Label writer
# ═════════════════════════════════════════════════════════════════════════════

def write_label(
    label_path: Path,
    class_id: int,
    boxes: list[tuple[float, float, float, float]],
) -> None:
    """Write a YOLO-format .txt label file.  One detection per line."""
    label_path.parent.mkdir(parents=True, exist_ok=True)
    with open(label_path, "w") as f:
        for cx, cy, w, h in boxes:
            f.write(f"{class_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n")


# ═════════════════════════════════════════════════════════════════════════════
# Main pipeline
# ═════════════════════════════════════════════════════════════════════════════

def main() -> None:
    args = parse_args()

    # ── Resolve paths ─────────────────────────────────────────────────────────
    input_dir  = Path(args.input)
    output_dir = Path(args.output)
    checkpoint = Path(args.checkpoint)

    if not input_dir.is_absolute():
        input_dir = PROJECT_ROOT / input_dir
    if not output_dir.is_absolute():
        output_dir = PROJECT_ROOT / output_dir
    if not checkpoint.is_absolute():
        checkpoint = PROJECT_ROOT / checkpoint

    images_out = output_dir / "images"
    labels_out = output_dir / "labels"
    images_out.mkdir(parents=True, exist_ok=True)
    labels_out.mkdir(parents=True, exist_ok=True)

    # ── Validate input directory ───────────────────────────────────────────────
    if not input_dir.exists():
        print(f"[ERROR] Input directory not found: {input_dir}", file=sys.stderr)
        sys.exit(1)

    # ── Discover class folders ────────────────────────────────────────────────
    class_dirs: dict[str, Path] = {}
    for d in sorted(input_dir.iterdir()):
        if d.is_dir() and d.name in CLASS_MAP:
            class_dirs[d.name] = d

    if not class_dirs:
        print(
            f"[ERROR] No recognised class folders found in: {input_dir}\n"
            f"        Expected one or more of: {list(CLASS_MAP.keys())}\n"
            f"        Found: {[d.name for d in input_dir.iterdir() if d.is_dir()]}",
            file=sys.stderr,
        )
        sys.exit(1)

    skipped_dirs = [
        d.name for d in input_dir.iterdir()
        if d.is_dir() and d.name not in CLASS_MAP
    ]
    if skipped_dirs:
        print(f"[WARN] Skipping unrecognised folders: {skipped_dirs}")

    print("=" * 60)
    print("[sam_to_yolo] Starting SAM annotation pipeline")
    print(f"  Input        : {input_dir}")
    print(f"  Output       : {output_dir}")
    print(f"  Checkpoint   : {checkpoint}")
    print(f"  Min px       : {args.min_px}")
    print(f"  Max coverage : {args.max_cover:.0%}")
    print(f"  Classes      : {list(class_dirs.keys())}")
    print("=" * 60)

    # ── Load SAM ──────────────────────────────────────────────────────────────
    mask_generator = load_sam(checkpoint)

    # ── Per-class processing ──────────────────────────────────────────────────
    total_images    = 0
    total_labels    = 0
    total_fallbacks = 0
    total_skipped   = 0

    for class_name, class_dir in class_dirs.items():
        class_id = CLASS_MAP[class_name]

        image_paths = sorted(
            p for p in class_dir.iterdir()
            if p.suffix.lower() in IMAGE_EXTS
        )

        if not image_paths:
            print(f"\n[{class_name}] No images found — skipping class.")
            continue

        print(f"\n[{class_name}] class_id={class_id}  images={len(image_paths)}")

        class_images_done  = 0
        class_label_count  = 0
        class_fallbacks    = 0
        class_skipped      = 0

        for idx, img_path in enumerate(image_paths, 1):

            # ── Progress log every N images ───────────────────────────────────
            if idx % args.log_every == 0 or idx == len(image_paths):
                print(
                    f"  [{class_name}] Progress: {idx}/{len(image_paths)} images"
                    f"  (labels={class_label_count}, fallbacks={class_fallbacks},"
                    f" skipped={class_skipped})"
                )

            # ── Build output file names (prefixed with class name) ─────────────
            stem        = img_path.stem
            suffix      = img_path.suffix.lower()
            out_name    = f"{class_name}_{stem}"
            dst_image   = images_out / f"{out_name}{suffix}"
            dst_label   = labels_out / f"{out_name}.txt"

            # ── Load image ────────────────────────────────────────────────────
            img_bgr = cv2.imread(str(img_path))
            if img_bgr is None:
                print(f"  [WARN] Cannot read image — skipping: {img_path.name}")
                class_skipped += 1
                total_skipped += 1
                continue

            img_h, img_w = img_bgr.shape[:2]

            # ── Resize to max width 640 (aspect-ratio preserved) ──────────────
            MAX_WIDTH = 640
            if img_w > MAX_WIDTH:
                scale     = MAX_WIDTH / img_w
                new_w     = MAX_WIDTH
                new_h     = int(img_h * scale)
                img_bgr   = cv2.resize(img_bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)
                img_h, img_w = new_h, new_w

            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

            # ── Run SAM ───────────────────────────────────────────────────────
            try:
                masks = mask_generator.generate(img_rgb)
            except Exception as exc:
                print(
                    f"  [WARN] SAM failed on {img_path.name}: "
                    f"{type(exc).__name__}: {exc} — using fallback box"
                )
                traceback.print_exc(file=sys.stderr)
                masks = []

            # ── Convert masks to YOLO boxes ───────────────────────────────────
            boxes = masks_to_yolo_boxes(
                masks    = masks,
                img_h    = img_h,
                img_w    = img_w,
                min_px   = args.min_px,
                max_cover= args.max_cover,
            )

            # ── Keep only top 5 boxes by area (w × h), descending ────────────
            if len(boxes) > 5:
                boxes = sorted(boxes, key=lambda b: b[2] * b[3], reverse=True)[:5]

            # ── Fallback: full-image box if SAM yields nothing valid ───────────
            used_fallback = False
            if not boxes:
                boxes         = fallback_box()
                used_fallback = True
                class_fallbacks += 1
                total_fallbacks += 1

            # ── Write label file ──────────────────────────────────────────────
            write_label(dst_label, class_id, boxes)

            # ── Copy image to output directory ────────────────────────────────
            shutil.copy2(img_path, dst_image)

            class_images_done += 1
            class_label_count += len(boxes)
            total_images      += 1
            total_labels      += len(boxes)

        print(
            f"  [{class_name}] Done — "
            f"images={class_images_done}, "
            f"labels={class_label_count}, "
            f"fallbacks={class_fallbacks}, "
            f"skipped={class_skipped}"
        )

    # ── Final summary ─────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("[sam_to_yolo] Pipeline complete.")
    print(f"  Images processed : {total_images}")
    print(f"  Total labels     : {total_labels}")
    print(f"  Fallback boxes   : {total_fallbacks}  (full-image box — no valid SAM mask)")
    print(f"  Skipped (errors) : {total_skipped}")
    print(f"  Output images    : {images_out}")
    print(f"  Output labels    : {labels_out}")
    print("=" * 60)

    if total_images == 0:
        print(
            "\n[ERROR] No images were processed.  Check that your raw_data/ "
            "folders contain image files.",
            file=sys.stderr,
        )
        sys.exit(1)

    print("\n[sam_to_yolo] Next step:")
    print("  python training/train.py --data training/data.yaml --epochs 50")


if __name__ == "__main__":
    main()

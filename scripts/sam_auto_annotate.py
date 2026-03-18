#!/usr/bin/env python3
"""
sam_auto_annotate.py — Step 1 of the SAM → YOLO bootstrap pipeline
────────────────────────────────────────────────────────────────────
Uses SAM (Segment Anything Model) to generate bounding-box annotations
automatically from a class-labelled folder structure.  Labels are written
in YOLO format, ready for training with !training/train.py.

No manual annotation is required.

INPUT layout expected (--input):
    dataset/
    ├── Missing_hole/        ← all images of that defect type
    ├── Mouse_bite/
    ├── Open_circuit/
    ├── Short/
    ├── Spur/
    └── Spurious_copper/

OUTPUT layout produced (--output):
    data/
    ├── images/
    │   ├── train/
    │   └── val/
    └── labels/
        ├── train/
        └── val/

Usage (from project root):
    python scripts/sam_auto_annotate.py
    python scripts/sam_auto_annotate.py --input dataset/ --output data/
    python scripts/sam_auto_annotate.py --sam-weights weights/sam_vit_b.pth
    python scripts/sam_auto_annotate.py --val-split 0.2 --overwrite

Requirements:
    pip install -r requirements-sam.txt
    # SAM checkpoint in weights/ (see requirements-sam.txt for download URLs)

Bootstrap pipeline:
    Step 1  python scripts/sam_auto_annotate.py       # SAM labels → data/
    Step 2  python !training/train.py                 # YOLO → weights/best.pt
    Step 3  python scripts/auto_label_with_yolo.py    # replace labels
    Step 4  python !training/train.py                 # retrain → improved model
    Step 5  (repeat Steps 3–4 as desired)
"""

from __future__ import annotations

import argparse
import random
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np

# ── Project root ──────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# ── Class map — must stay in sync with !training/data.yaml ───────────────────
CLASS_MAP: dict[str, int] = {
    "Missing_hole":    0,
    "Mouse_bite":      1,
    "Open_circuit":    2,
    "Short":           3,
    "Spur":            4,
    "Spurious_copper": 5,
}

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# ── SAM filtering defaults ────────────────────────────────────────────────────
_MIN_AREA_PX      = 500     # ignore regions smaller than this (pixels²)
_MAX_AREA_RATIO   = 0.60    # ignore regions covering > 60 % of image area
_MIN_ASPECT       = 0.10    # ignore very thin/tall slivers (w/h ratio)
_MAX_ASPECT       = 10.0    # ignore very wide/flat regions
_BORDER_MARGIN_PX = 3       # ignore boxes whose edge touches the image border
_NMS_IOU_THRESH   = 0.50    # greedy NMS IoU threshold
_MIN_STABILITY    = 0.75    # SAM stability score minimum


# ═════════════════════════════════════════════════════════════════════════════
# CLI
# ═════════════════════════════════════════════════════════════════════════════

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Step 1 — SAM automatic annotation from class-labelled folder dataset.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--input", default="dataset",
        help="Root folder containing one sub-directory per defect class.",
    )
    p.add_argument(
        "--output", default="data",
        help="Output root for the YOLO train/val split.",
    )
    p.add_argument(
        "--sam-weights", default="weights/sam_vit_h.pth",
        help="Path to SAM checkpoint (.pth).  Filename must contain vit_b, vit_l, or vit_h.",
    )
    p.add_argument(
        "--device", default="cpu",
        help="Device for SAM inference: 'cpu', 'cuda', 'cuda:0', 'mps'.",
    )
    p.add_argument(
        "--val-split", type=float, default=0.20,
        help="Fraction of images per class to allocate to the validation set.",
    )
    p.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducible train/val split.",
    )
    p.add_argument(
        "--min-area", type=int, default=_MIN_AREA_PX,
        help="Minimum SAM mask area in pixels² to accept.",
    )
    p.add_argument(
        "--max-area-ratio", type=float, default=_MAX_AREA_RATIO,
        help="Maximum SAM mask area as a fraction of total image area.",
    )
    p.add_argument(
        "--overwrite", action="store_true",
        help="Re-annotate images that already have a label file.",
    )
    return p.parse_args()


# ═════════════════════════════════════════════════════════════════════════════
# SAM helpers
# ═════════════════════════════════════════════════════════════════════════════

def detect_model_type(weights_path: Path) -> str:
    """Infer SAM model type from checkpoint filename (vit_b / vit_l / vit_h)."""
    name = weights_path.stem.lower()
    for variant in ("vit_h", "vit_l", "vit_b"):
        if variant in name:
            return variant
    print(
        f"[SAM] WARNING: Cannot infer model type from '{weights_path.name}'.\n"
        "              Defaulting to vit_h.  Rename the file to include\n"
        "              'vit_b', 'vit_l', or 'vit_h' to silence this warning.",
        file=sys.stderr,
    )
    return "vit_h"


# ═════════════════════════════════════════════════════════════════════════════
# Mask filtering
# ═════════════════════════════════════════════════════════════════════════════

def _iou(box_a: list[int], box_b: list[int]) -> float:
    """IoU between two [x1, y1, x2, y2] boxes."""
    xa1 = max(box_a[0], box_b[0])
    ya1 = max(box_a[1], box_b[1])
    xa2 = min(box_a[2], box_b[2])
    ya2 = min(box_a[3], box_b[3])
    inter = max(0, xa2 - xa1) * max(0, ya2 - ya1)
    if inter == 0:
        return 0.0
    area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
    union  = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def _greedy_nms(
    boxes_scores: list[tuple[list[int], float]],
    iou_thresh: float,
) -> list[int]:
    """
    Greedy NMS.
    boxes_scores — list of ([x1, y1, x2, y2], score)
    Returns the indices to keep, ordered by descending score.
    """
    if not boxes_scores:
        return []
    order = sorted(
        range(len(boxes_scores)),
        key=lambda i: boxes_scores[i][1],
        reverse=True,
    )
    keep: list[int] = []
    while order:
        best = order.pop(0)
        keep.append(best)
        order = [
            j for j in order
            if _iou(boxes_scores[best][0], boxes_scores[j][0]) < iou_thresh
        ]
    return keep


def filter_masks(
    masks: list[dict],
    img_h: int,
    img_w: int,
    min_area: int,
    max_area_ratio: float,
) -> list[dict]:
    """
    Five-stage filter applied to raw SAM masks:
      1. Minimum area (absolute pixel count)
      2. Maximum area (fraction of image)
      3. Aspect ratio bounds
      4. Border proximity
      5. SAM stability score
    Followed by greedy IoU-NMS to remove duplicates.
    """
    img_area = img_h * img_w
    valid: list[dict] = []

    for m in masks:
        # SAM bbox format: [x, y, width, height]  (x,y = top-left corner)
        x, y, bw, bh = m["bbox"]
        area = m["area"]

        # ── 1. Area (absolute) ───────────────────────────────────────────────
        if area < min_area:
            continue

        # ── 2. Area (relative) ───────────────────────────────────────────────
        if area > img_area * max_area_ratio:
            continue

        # ── 3. Aspect ratio ──────────────────────────────────────────────────
        if bh <= 0:
            continue
        aspect = bw / bh
        if not (_MIN_ASPECT <= aspect <= _MAX_ASPECT):
            continue

        # ── 4. Border proximity ──────────────────────────────────────────────
        x2, y2 = x + bw, y + bh
        if (
            x  <= _BORDER_MARGIN_PX
            or y  <= _BORDER_MARGIN_PX
            or x2 >= img_w - _BORDER_MARGIN_PX
            or y2 >= img_h - _BORDER_MARGIN_PX
        ):
            continue

        # ── 5. SAM stability score ───────────────────────────────────────────
        if m.get("stability_score", 1.0) < _MIN_STABILITY:
            continue

        valid.append(m)

    if not valid:
        return valid

    # ── NMS ───────────────────────────────────────────────────────────────────
    boxes_scores = [
        ([m["bbox"][0],
          m["bbox"][1],
          m["bbox"][0] + m["bbox"][2],
          m["bbox"][1] + m["bbox"][3]],
         m.get("stability_score", 1.0))
        for m in valid
    ]
    keep_idx = _greedy_nms(boxes_scores, _NMS_IOU_THRESH)
    return [valid[i] for i in keep_idx]


# ═════════════════════════════════════════════════════════════════════════════
# YOLO format helpers
# ═════════════════════════════════════════════════════════════════════════════

def mask_to_yolo(
    mask: dict,
    img_h: int,
    img_w: int,
) -> tuple[float, float, float, float]:
    """
    Convert a SAM mask's [x, y, w, h] bbox (pixel, top-left origin) to
    YOLO normalised [cx, cy, w, h] (all values 0–1).
    """
    x, y, bw, bh = mask["bbox"]
    cx = (x + bw / 2) / img_w
    cy = (y + bh / 2) / img_h
    nw = bw / img_w
    nh = bh / img_h
    # Clamp to [0, 1] to guard against sub-pixel rounding
    cx = min(max(cx, 0.0), 1.0)
    cy = min(max(cy, 0.0), 1.0)
    nw = min(max(nw, 0.0), 1.0)
    nh = min(max(nh, 0.0), 1.0)
    return cx, cy, nw, nh


def write_labels(
    label_path: Path,
    class_id: int,
    yolo_boxes: list[tuple[float, float, float, float]],
) -> None:
    """Write YOLO-format .txt label file.  One object per line."""
    label_path.parent.mkdir(parents=True, exist_ok=True)
    with open(label_path, "w") as f:
        for cx, cy, w, h in yolo_boxes:
            f.write(f"{class_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n")


# ═════════════════════════════════════════════════════════════════════════════
# Main
# ═════════════════════════════════════════════════════════════════════════════

def main() -> None:
    args = parse_args()
    random.seed(args.seed)

    # ── Resolve paths ─────────────────────────────────────────────────────────
    input_dir   = Path(args.input)
    output_dir  = Path(args.output)
    sam_weights = Path(args.sam_weights)

    if not input_dir.is_absolute():
        input_dir   = PROJECT_ROOT / input_dir
    if not output_dir.is_absolute():
        output_dir  = PROJECT_ROOT / output_dir
    if not sam_weights.is_absolute():
        sam_weights = PROJECT_ROOT / sam_weights

    # ── Validate inputs ───────────────────────────────────────────────────────
    if not input_dir.exists():
        print(f"[SAM] ERROR: Input directory not found: {input_dir}", file=sys.stderr)
        sys.exit(1)

    if not sam_weights.exists():
        print(
            f"[SAM] ERROR: SAM checkpoint not found: {sam_weights}\n"
            "\n"
            "  Download one of the following on a connected machine and transfer\n"
            "  it to the weights/ directory:\n"
            "\n"
            "    ViT-H (~2.4 GB — best quality):\n"
            "      https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth\n"
            "      → save as weights/sam_vit_h.pth\n"
            "\n"
            "    ViT-L (~1.2 GB):\n"
            "      https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth\n"
            "      → save as weights/sam_vit_l.pth\n"
            "\n"
            "    ViT-B (~375 MB — fastest):\n"
            "      https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth\n"
            "      → save as weights/sam_vit_b.pth",
            file=sys.stderr,
        )
        sys.exit(1)

    # ── Discover class folders ────────────────────────────────────────────────
    class_dirs: dict[str, Path] = {
        d.name: d
        for d in sorted(input_dir.iterdir())
        if d.is_dir() and d.name in CLASS_MAP
    }

    if not class_dirs:
        print(
            f"[SAM] ERROR: No recognised class folders found in {input_dir}\n"
            f"             Expected one or more of: {list(CLASS_MAP.keys())}\n"
            f"             Got: {[d.name for d in input_dir.iterdir() if d.is_dir()]}",
            file=sys.stderr,
        )
        sys.exit(1)

    unknown = [
        d.name for d in input_dir.iterdir()
        if d.is_dir() and d.name not in CLASS_MAP
    ]
    if unknown:
        print(
            f"[SAM] WARNING: Ignoring unrecognised folder(s): {unknown}",
            file=sys.stderr,
        )

    print(f"[SAM] Input        : {input_dir}")
    print(f"[SAM] Output       : {output_dir}")
    print(f"[SAM] SAM weights  : {sam_weights}")
    print(f"[SAM] Device       : {args.device}")
    print(f"[SAM] Val split    : {args.val_split:.0%}")
    print(f"[SAM] Classes      : {list(class_dirs.keys())}\n")

    # ── Load SAM ──────────────────────────────────────────────────────────────
    try:
        from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
    except ImportError:
        print(
            "[SAM] ERROR: segment_anything package not installed.\n"
            "             Install it:  pip install -r requirements-sam.txt",
            file=sys.stderr,
        )
        sys.exit(1)

    model_type = detect_model_type(sam_weights)
    print(f"[SAM] Loading {model_type} checkpoint …")

    sam = sam_model_registry[model_type](checkpoint=str(sam_weights))
    sam.to(device=args.device)
    sam.eval()

    mask_generator = SamAutomaticMaskGenerator(
        model                  = sam,
        points_per_side        = 32,
        pred_iou_thresh        = 0.88,
        stability_score_thresh = _MIN_STABILITY,
        box_nms_thresh         = _NMS_IOU_THRESH,
        min_mask_region_area   = args.min_area,
    )

    print(f"[SAM] Model loaded.  Starting annotation …\n")

    # ── Create output directory structure ─────────────────────────────────────
    for split in ("train", "val"):
        (output_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (output_dir / "labels" / split).mkdir(parents=True, exist_ok=True)

    # ── Annotate ──────────────────────────────────────────────────────────────
    total_images    = 0
    total_fallbacks = 0
    total_labels    = 0

    for class_name, class_dir in class_dirs.items():
        class_id = CLASS_MAP[class_name]
        images   = sorted(
            p for p in class_dir.iterdir()
            if p.suffix.lower() in IMAGE_EXTS
        )

        if not images:
            print(f"  [{class_name}] No images found — skipping.")
            continue

        # ── Per-class train/val split ──────────────────────────────────────────
        random.shuffle(images)
        n_val   = max(1, int(len(images) * args.val_split))
        val_set = {p.name for p in images[:n_val]}
        n_train = len(images) - n_val

        print(
            f"  [{class_name}]  class_id={class_id}  "
            f"total={len(images)}  train={n_train}  val={n_val}"
        )

        for img_path in images:
            split     = "val" if img_path.name in val_set else "train"
            dst_img   = output_dir / "images" / split / img_path.name
            dst_label = output_dir / "labels" / split / (img_path.stem + ".txt")

            # ── Skip already processed ────────────────────────────────────────
            if dst_label.exists() and not args.overwrite:
                continue

            # ── Copy image into output tree ───────────────────────────────────
            shutil.copy2(img_path, dst_img)

            # ── Load image ────────────────────────────────────────────────────
            img_bgr = cv2.imread(str(img_path))
            if img_bgr is None:
                print(
                    f"    [warn] Cannot read {img_path.name} — skipping",
                    file=sys.stderr,
                )
                continue

            img_rgb       = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            img_h, img_w  = img_bgr.shape[:2]

            # ── Run SAM ───────────────────────────────────────────────────────
            try:
                masks = mask_generator.generate(img_rgb)
            except Exception as exc:
                print(
                    f"    [warn] SAM failed on {img_path.name}: {exc} — using fallback",
                    file=sys.stderr,
                )
                masks = []

            # ── Filter masks ──────────────────────────────────────────────────
            valid_masks = filter_masks(
                masks, img_h, img_w,
                min_area       = args.min_area,
                max_area_ratio = args.max_area_ratio,
            )

            # ── Fallback: full-image bounding box ─────────────────────────────
            # If SAM finds no valid region, record the entire image as one bbox.
            # This ensures every image has at least one label, which YOLO needs.
            if not valid_masks:
                yolo_boxes = [(0.5, 0.5, 1.0, 1.0)]
                total_fallbacks += 1
            else:
                yolo_boxes = [
                    mask_to_yolo(m, img_h, img_w) for m in valid_masks
                ]

            write_labels(dst_label, class_id, yolo_boxes)
            total_images += 1
            total_labels += len(yolo_boxes)

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n{'─'*60}")
    print(f"[SAM] Annotation complete.")
    print(f"      Images processed  : {total_images}")
    print(f"      Total labels      : {total_labels}")
    if total_fallbacks:
        print(
            f"      Fallback boxes    : {total_fallbacks}  "
            "(full-image box — SAM found no valid regions)"
        )
    print(f"      Output            : {output_dir}")
    print(f"\n[SAM] Next step:")
    print(f"      python !training/train.py --data !training/data.yaml")
    print(f"{'─'*60}\n")


if __name__ == "__main__":
    main()

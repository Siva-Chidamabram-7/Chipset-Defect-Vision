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

    GCS is also supported:
    --input gs://my-bucket/dataset/

OUTPUT layout produced (--output):
    data/
    ├── images/
    │   ├── train/
    │   └── val/
    └── labels/
        ├── train/
        └── val/

    GCS output: --output gs://my-bucket/data/

Usage (from project root):
    # Local
    python scripts/sam_auto_annotate.py
    python scripts/sam_auto_annotate.py --input dataset/ --output data/
    python scripts/sam_auto_annotate.py --sam-weights weights/sam_vit_b.pth
    python scripts/sam_auto_annotate.py --val-split 0.2 --overwrite

    # GCS (Vertex AI / Cloud)
    python scripts/sam_auto_annotate.py \
        --input  gs://my-bucket/dataset/ \
        --output gs://my-bucket/data/ \
        --sam-weights gs://my-bucket/weights/sam_vit_b.pth \
        --device cuda

Requirements:
    pip install -r requirements-training.txt
    # SAM checkpoint in weights/ (see requirements-training.txt for download URLs)

Bootstrap pipeline:
    Step 1  python scripts/sam_auto_annotate.py       # SAM labels → data/
    Step 2  python !training/train.py                 # YOLO → weights/best.pt
    Step 3  python scripts/auto_label_with_yolo.py    # replace labels
    Step 4  python !training/train.py                 # retrain → improved model
    Step 5  (repeat Steps 3–4 as desired)
"""

from __future__ import annotations

import argparse
import logging
import random
import shutil
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np

# ── Project root ──────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# ── Logging ───────────────────────────────────────────────────────────────────
# Configured in main() via gcs_utils.setup_logging() so that log level can be
# controlled from the CLI and output is captured by Vertex AI / Docker.
log = logging.getLogger("sam_annotate")

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
        "--input", default=None,
        help=(
            "Root folder containing one sub-directory per defect class.  "
            "Accepts local path or gs://bucket/prefix."
        ),
    )
    p.add_argument(
        "--output", default=None,
        help=(
            "Output root for the YOLO train/val split.  "
            "Accepts local path or gs://bucket/prefix."
        ),
    )
    p.add_argument(
        "--sam-weights", default=None,
        help=(
            "Path to SAM checkpoint (.pth).  Filename must contain vit_b, vit_l, "
            "or vit_h.  Accepts local path or gs://bucket/blob."
        ),
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
    p.add_argument(
        "--log-level", default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity (passed to Python logging module).",
    )
    # ── Environment-variable defaults (set by Vertex AI or CI) ────────────────
    # The script reads these so it can be driven entirely by env vars,
    # which is the standard Vertex AI custom-training-job pattern.
    import os
    p.set_defaults(
        input       = os.environ.get("SAM_INPUT_PATH",   "dataset"),
        output      = os.environ.get("SAM_OUTPUT_PATH",  "data"),
        sam_weights = os.environ.get("SAM_WEIGHTS_PATH", "weights/sam_vit_h.pth"),
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
    log.warning(
        "[SAM] Cannot infer model type from '%s'. "
        "Defaulting to vit_h.  Rename the file to include "
        "'vit_b', 'vit_l', or 'vit_h' to silence this warning.",
        weights_path.name,
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

    # ── Logging setup (Vertex AI + Docker compatible) ─────────────────────────
    from scripts.gcs_utils import setup_logging, is_gcs_path, resolve_input_path, \
        resolve_input_file, upload_gcs_dir
    setup_logging(level=getattr(logging, args.log_level.upper(), logging.INFO))

    log.info("=" * 60)
    log.info("[SAM] Step 1 — SAM Auto-Annotation starting")
    log.info("[SAM] Input path    : %s", args.input)
    log.info("[SAM] Output path   : %s", args.output)
    log.info("[SAM] SAM weights   : %s", args.sam_weights)
    log.info("[SAM] Device        : %s", args.device)
    log.info("[SAM] Val split     : %.0f%%", args.val_split * 100)
    log.info("[SAM] Seed          : %d", args.seed)
    log.info("=" * 60)

    random.seed(args.seed)

    # ── GCS / local path resolution ───────────────────────────────────────────
    # Create a single temp dir for all downloads in this run.
    _tmpdir = tempfile.mkdtemp(prefix="pcb_sam_")
    tmp     = Path(_tmpdir)

    try:
        # -- Input dataset: download from GCS if needed -----------------------
        input_dir = resolve_input_path(args.input, tmp, "input")
        if not input_dir.is_absolute():
            input_dir = PROJECT_ROOT / input_dir

        # -- SAM weights: download from GCS if needed -------------------------
        sam_weights = resolve_input_file(args.sam_weights, tmp, Path(args.sam_weights).name)
        if not sam_weights.is_absolute() and not is_gcs_path(args.sam_weights):
            sam_weights = PROJECT_ROOT / sam_weights

        # -- Output dir: local staging (will upload to GCS at the end) --------
        output_is_gcs = is_gcs_path(args.output)
        if output_is_gcs:
            output_dir = tmp / "output"
            output_dir.mkdir(parents=True, exist_ok=True)
        else:
            output_dir = Path(args.output)
            if not output_dir.is_absolute():
                output_dir = PROJECT_ROOT / output_dir

        # ── Validate inputs ───────────────────────────────────────────────────
        if not input_dir.exists():
            log.error("[SAM] Input directory not found: %s", input_dir)
            sys.exit(1)

        if not sam_weights.exists():
            log.error(
                "[SAM] SAM checkpoint not found: %s\n\n"
                "  Download one of the following and place in weights/:\n"
                "    ViT-H (~2.4 GB — best quality):\n"
                "      https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth\n"
                "      → save as weights/sam_vit_h.pth\n"
                "    ViT-L (~1.2 GB):\n"
                "      https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth\n"
                "      → save as weights/sam_vit_l.pth\n"
                "    ViT-B (~375 MB — fastest):\n"
                "      https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth\n"
                "      → save as weights/sam_vit_b.pth",
                sam_weights,
            )
            sys.exit(1)

        # ── Discover class folders ────────────────────────────────────────────
        class_dirs: dict[str, Path] = {
            d.name: d
            for d in sorted(input_dir.iterdir())
            if d.is_dir() and d.name in CLASS_MAP
        }

        if not class_dirs:
            log.error(
                "[SAM] No recognised class folders found in %s\n"
                "      Expected one or more of: %s\n"
                "      Got: %s",
                input_dir,
                list(CLASS_MAP.keys()),
                [d.name for d in input_dir.iterdir() if d.is_dir()],
            )
            sys.exit(1)

        unknown = [
            d.name for d in input_dir.iterdir()
            if d.is_dir() and d.name not in CLASS_MAP
        ]
        if unknown:
            log.warning("[SAM] Ignoring unrecognised folder(s): %s", unknown)

        log.info("[SAM] Classes found : %s", list(class_dirs.keys()))

        # ── Load SAM ──────────────────────────────────────────────────────────
        try:
            from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
        except ImportError:
            log.error(
                "[SAM] segment_anything package not installed.\n"
                "      Install it:  pip install -r requirements-training.txt"
            )
            sys.exit(1)

        model_type = detect_model_type(sam_weights)
        log.info("[SAM] Loading SAM checkpoint: %s (type=%s)", sam_weights.name, model_type)

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

        log.info("[SAM] Model loaded. Starting annotation ...")

        # ── Create output directory structure ─────────────────────────────────
        for split in ("train", "val"):
            (output_dir / "images" / split).mkdir(parents=True, exist_ok=True)
            (output_dir / "labels" / split).mkdir(parents=True, exist_ok=True)

        # ── Annotate ──────────────────────────────────────────────────────────
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
                log.warning("[SAM] [%s] No images found — skipping.", class_name)
                continue

            # ── Per-class train/val split ──────────────────────────────────────
            random.shuffle(images)
            n_val   = max(1, int(len(images) * args.val_split))
            val_set = {p.name for p in images[:n_val]}
            n_train = len(images) - n_val

            log.info(
                "[SAM] [%s]  class_id=%d  total=%d  train=%d  val=%d",
                class_name, class_id, len(images), n_train, n_val,
            )

            for img_path in images:
                split     = "val" if img_path.name in val_set else "train"
                dst_img   = output_dir / "images" / split / img_path.name
                dst_label = output_dir / "labels" / split / (img_path.stem + ".txt")

                # ── Skip already processed ────────────────────────────────────
                if dst_label.exists() and not args.overwrite:
                    continue

                # ── Copy image into output tree ───────────────────────────────
                shutil.copy2(img_path, dst_img)

                # ── Load image ────────────────────────────────────────────────
                img_bgr = cv2.imread(str(img_path))
                if img_bgr is None:
                    log.warning("[SAM] Cannot read %s — skipping", img_path.name)
                    continue

                img_rgb       = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
                img_h, img_w  = img_bgr.shape[:2]

                # ── Run SAM ───────────────────────────────────────────────────
                try:
                    masks = mask_generator.generate(img_rgb)
                except Exception as exc:
                    log.warning(
                        "[SAM] SAM failed on %s: %s — using fallback", img_path.name, exc
                    )
                    masks = []

                # ── Filter masks ──────────────────────────────────────────────
                valid_masks = filter_masks(
                    masks, img_h, img_w,
                    min_area       = args.min_area,
                    max_area_ratio = args.max_area_ratio,
                )

                # ── Fallback: full-image bounding box ─────────────────────────
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

        # ── Summary ───────────────────────────────────────────────────────────
        log.info("─" * 60)
        log.info("[SAM] Annotation complete.")
        log.info("[SAM] Images processed  : %d", total_images)
        log.info("[SAM] Total labels      : %d", total_labels)
        if total_fallbacks:
            log.info(
                "[SAM] Fallback boxes    : %d  (full-image box — SAM found no valid regions)",
                total_fallbacks,
            )
        log.info("[SAM] Local output      : %s", output_dir)

        # ── Upload to GCS (if output was a GCS path) ──────────────────────────
        if output_is_gcs:
            log.info("[SAM] Uploading output to GCS: %s", args.output)
            upload_gcs_dir(output_dir, args.output)
            log.info("[SAM] GCS upload complete.")

        log.info("[SAM] Next step:")
        log.info("[SAM]   python !training/train.py --data !training/data.yaml")
        log.info("─" * 60)

    finally:
        # Clean up temp dir
        shutil.rmtree(_tmpdir, ignore_errors=True)


if __name__ == "__main__":
    main()

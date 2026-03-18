#!/usr/bin/env python3
"""
generate_regions.py — Phase 1 of the Hybrid SAM + YOLO Pipeline
────────────────────────────────────────────────────────────────
OFFLINE-FIRST: loads SAM from a local checkpoint; no internet, no download.

What this script does
─────────────────────
1. Loads SAM (Segment Anything Model) from a local .pth checkpoint.
2. Runs automatic mask generation on every PCB image in --images dir.
3. Converts segmentation masks → bounding boxes.
4. Filters regions by area, aspect ratio, and border proximity.
5. Deduplicates overlapping boxes with IoU-based NMS.
6. Saves one JSON file per image to --output dir.

The JSON files are the input to  scripts/annotate.py  (Phase 2).

Usage (run from project root):
    python scripts/generate_regions.py

    python scripts/generate_regions.py \\
        --images      raw_data/images \\
        --output      raw_data/regions \\
        --checkpoint  weights/sam_vit_b.pth \\
        --model-type  vit_b \\
        --device      cpu

Typical workflow:
    1.  python scripts/generate_regions.py   # Phase 1 — SAM region proposals
    2.  python scripts/annotate.py           # Phase 2 — human labeling
    3.  python training/prepare_dataset.py   # Phase 3 — train/val split
    4.  python training/train.py             # Phase 4 — YOLO fine-tuning

SAM checkpoint download (do once on a connected machine, then transfer):
    # ViT-B (~375 MB, recommended for CPU)
    wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth \\
         -O weights/sam_vit_b.pth

    # ViT-L (~1.2 GB, better accuracy)
    wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth \\
         -O weights/sam_vit_l.pth

    # ViT-H (~2.4 GB, best accuracy)
    wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth \\
         -O weights/sam_vit_h.pth
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np

# ── Project root ──────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
WEIGHTS_DIR  = PROJECT_ROOT / "weights"

# ── SAM import guard ───────────────────────────────────────────────────────────
try:
    from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
    _SAM_AVAILABLE = True
except ImportError:
    _SAM_AVAILABLE = False


# ═════════════════════════════════════════════════════════════════════════════
# CLI
# ═════════════════════════════════════════════════════════════════════════════

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Phase 1 — SAM automatic region proposal for PCB solder joints.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ── Paths ─────────────────────────────────────────────────────────────────
    p.add_argument("--images",     default="raw_data/images",
                   help="Directory containing PCB images.")
    p.add_argument("--output",     default="raw_data/regions",
                   help="Output directory for JSON region files.")
    p.add_argument("--checkpoint", default="weights/sam_vit_b.pth",
                   help="SAM checkpoint (.pth) — must exist locally.")
    p.add_argument("--model-type", default="vit_b",
                   choices=["vit_b", "vit_l", "vit_h"],
                   help="SAM model variant matching the checkpoint.")
    p.add_argument("--device",     default="cpu",
                   help="Torch device: 'cpu', '0' (CUDA), 'mps'.")

    # ── SAM generation parameters ─────────────────────────────────────────────
    p.add_argument("--points-per-side", type=int, default=32,
                   help="Grid density for automatic point prompts (higher = more masks).")
    p.add_argument("--pred-iou-thresh", type=float, default=0.88,
                   help="Minimum predicted IoU quality score to keep a mask.")
    p.add_argument("--stability-thresh", type=float, default=0.85,
                   help="Minimum stability score to keep a mask.")

    # ── Region filtering parameters ───────────────────────────────────────────
    p.add_argument("--min-area",   type=int,   default=400,
                   help="Minimum region area in pixels (removes noise/dust).")
    p.add_argument("--max-area",   type=int,   default=40_000,
                   help="Maximum region area in pixels (removes large background).")
    p.add_argument("--min-aspect", type=float, default=0.25,
                   help="Minimum bbox aspect ratio w/h (removes thin lines/traces).")
    p.add_argument("--max-aspect", type=float, default=4.0,
                   help="Maximum bbox aspect ratio w/h (removes elongated traces).")
    p.add_argument("--border-pad", type=int,   default=5,
                   help="Pixel margin — reject regions touching the image border.")
    p.add_argument("--nms-thresh", type=float, default=0.5,
                   help="IoU threshold for NMS to remove duplicate regions.")
    p.add_argument("--overwrite",  action="store_true",
                   help="Re-process images that already have a regions JSON.")

    return p.parse_args()


# ═════════════════════════════════════════════════════════════════════════════
# SAM loading
# ═════════════════════════════════════════════════════════════════════════════

def load_sam_generator(checkpoint: Path, model_type: str,
                        device: str, args: argparse.Namespace
                        ) -> "SamAutomaticMaskGenerator":
    """
    Load SAM from a local checkpoint and return an automatic mask generator.
    Never attempts a network download — exits loudly if checkpoint is missing.
    """
    if not _SAM_AVAILABLE:
        print(
            "[SAM] ERROR: 'segment-anything' package is not installed.\n"
            "      Install the dataset-creation environment:\n"
            "        pip install -r requirements-sam.txt\n"
            "      (torch must be installed first — see README)",
            file=sys.stderr,
        )
        sys.exit(1)

    if not checkpoint.exists():
        print(
            f"[SAM] ERROR: Checkpoint not found: {checkpoint}\n"
            f"      Download on a connected machine then transfer:\n"
            f"        wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth \\\n"
            f"             -O {checkpoint}\n"
            f"      Available variants: vit_b (~375 MB), vit_l (~1.2 GB), vit_h (~2.4 GB)",
            file=sys.stderr,
        )
        sys.exit(1)

    print(f"[SAM] Loading {model_type} from {checkpoint}  (device={device})")

    sam = sam_model_registry[model_type](checkpoint=str(checkpoint))

    # Map device string — CUDA index (e.g. "0") → "cuda:0"
    torch_device = device
    if device.isdigit():
        torch_device = f"cuda:{device}"
    sam.to(torch_device)

    mask_gen = SamAutomaticMaskGenerator(
        model                       = sam,
        points_per_side             = args.points_per_side,
        points_per_batch            = 64,
        pred_iou_thresh             = args.pred_iou_thresh,
        stability_score_thresh      = args.stability_thresh,
        stability_score_offset      = 1.0,
        box_nms_thresh              = 0.7,
        crop_n_layers               = 1,
        crop_nms_thresh             = 0.7,
        crop_overlap_ratio          = 512 / 1500,
        crop_n_points_downscale_factor = 2,
        min_mask_region_area        = 100,   # pre-filter inside SAM
        output_mode                 = "binary_mask",
    )

    print(f"[SAM] Model ready.\n")
    return mask_gen


# ═════════════════════════════════════════════════════════════════════════════
# Bounding-box extraction & filtering
# ═════════════════════════════════════════════════════════════════════════════

def iou(a: list[int], b: list[int]) -> float:
    """Compute IoU between two [x1, y1, x2, y2] boxes."""
    ix1 = max(a[0], b[0])
    iy1 = max(a[1], b[1])
    ix2 = min(a[2], b[2])
    iy2 = min(a[3], b[3])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    if inter == 0:
        return 0.0
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    return inter / (area_a + area_b - inter)


def nms(regions: list[dict], thresh: float) -> list[dict]:
    """
    Non-maximum suppression on regions sorted by stability_score (desc).
    Removes any region whose IoU with a higher-scored region exceeds thresh.
    """
    if not regions:
        return regions

    regions = sorted(regions, key=lambda r: r["stability_score"], reverse=True)
    kept: list[dict] = []

    for candidate in regions:
        bbox = candidate["bbox_abs"]
        if all(iou(bbox, k["bbox_abs"]) < thresh for k in kept):
            kept.append(candidate)

    return kept


def masks_to_regions(masks: list[dict], img_h: int, img_w: int,
                     args: argparse.Namespace) -> list[dict]:
    """
    Convert SAM mask dicts → filtered, normalised region dicts.

    SAM bbox format: [x, y, w, h] (absolute pixels, top-left origin)
    Output format: {"bbox_abs": [x1,y1,x2,y2], "bbox_yolo": [cx,cy,w,h], ...}
    """
    regions: list[dict] = []

    for i, m in enumerate(masks):
        x, y, bw, bh = m["bbox"]          # SAM: [x, y, w, h]
        x1, y1 = int(x), int(y)
        x2, y2 = int(x + bw), int(y + bh)
        area        = int(m["area"])
        stability   = float(m["stability_score"])
        pred_iou    = float(m.get("predicted_iou", 0.0))

        # ── Area filter ───────────────────────────────────────────────────────
        if area < args.min_area or area > args.max_area:
            continue

        # ── Aspect ratio filter ───────────────────────────────────────────────
        w_px = x2 - x1
        h_px = y2 - y1
        if h_px == 0:
            continue
        aspect = w_px / h_px
        if aspect < args.min_aspect or aspect > args.max_aspect:
            continue

        # ── Border proximity filter ───────────────────────────────────────────
        pad = args.border_pad
        if x1 < pad or y1 < pad or x2 > img_w - pad or y2 > img_h - pad:
            continue

        # ── YOLO normalised bbox ──────────────────────────────────────────────
        cx = (x1 + x2) / 2 / img_w
        cy = (y1 + y2) / 2 / img_h
        nw = w_px / img_w
        nh = h_px / img_h

        regions.append({
            "id":              i,
            "bbox_abs":        [x1, y1, x2, y2],
            "bbox_yolo":       [round(cx, 6), round(cy, 6),
                                round(nw, 6), round(nh, 6)],
            "area":            area,
            "stability_score": round(stability, 4),
            "pred_iou":        round(pred_iou, 4),
            "label":           None,   # filled in by annotate.py
        })

    return regions


# ═════════════════════════════════════════════════════════════════════════════
# Per-image processing
# ═════════════════════════════════════════════════════════════════════════════

def process_image(img_path: Path, mask_gen: "SamAutomaticMaskGenerator",
                  output_dir: Path, args: argparse.Namespace) -> dict[str, Any]:
    """
    Run SAM on a single image, filter regions, write JSON, return summary.
    """
    json_path = output_dir / (img_path.stem + ".json")

    if json_path.exists() and not args.overwrite:
        print(f"  [skip] {img_path.name} — regions file exists (use --overwrite to redo)")
        with open(json_path) as f:
            data = json.load(f)
        return {"file": img_path.name, "regions": len(data["regions"]), "status": "cached"}

    # ── Load image ────────────────────────────────────────────────────────────
    img_bgr = cv2.imread(str(img_path))
    if img_bgr is None:
        print(f"  [warn] Cannot read {img_path.name} — skipping.", file=sys.stderr)
        return {"file": img_path.name, "regions": 0, "status": "unreadable"}

    img_h, img_w = img_bgr.shape[:2]
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    # ── Run SAM ───────────────────────────────────────────────────────────────
    print(f"  [SAM]  {img_path.name} ({img_w}×{img_h}) ...", end=" ", flush=True)
    masks = mask_gen.generate(img_rgb)
    print(f"{len(masks)} masks", end=" → ", flush=True)

    # ── Filter + NMS ──────────────────────────────────────────────────────────
    regions = masks_to_regions(masks, img_h, img_w, args)
    regions = nms(regions, args.nms_thresh)

    # Re-index after NMS
    for idx, r in enumerate(regions):
        r["id"] = idx

    print(f"{len(regions)} regions after filtering")

    # ── Save JSON ─────────────────────────────────────────────────────────────
    payload: dict[str, Any] = {
        "image":   str(img_path),
        "width":   img_w,
        "height":  img_h,
        "regions": regions,
    }
    with open(json_path, "w") as f:
        json.dump(payload, f, indent=2)

    return {"file": img_path.name, "regions": len(regions), "status": "ok"}


# ═════════════════════════════════════════════════════════════════════════════
# Main
# ═════════════════════════════════════════════════════════════════════════════

def main() -> None:
    args = parse_args()

    # ── Resolve paths ─────────────────────────────────────────────────────────
    images_dir  = Path(args.images)
    output_dir  = Path(args.output)
    checkpoint  = Path(args.checkpoint)

    if not images_dir.is_absolute():
        images_dir = PROJECT_ROOT / images_dir
    if not output_dir.is_absolute():
        output_dir = PROJECT_ROOT / output_dir
    if not checkpoint.is_absolute():
        checkpoint = PROJECT_ROOT / checkpoint

    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Gather images ─────────────────────────────────────────────────────────
    img_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    images   = sorted([p for p in images_dir.iterdir()
                       if p.suffix.lower() in img_exts])

    if not images:
        print(
            f"[SAM] ERROR: No images found in {images_dir}\n"
            f"      Place PCB images in raw_data/images/ first.",
            file=sys.stderr,
        )
        sys.exit(1)

    print(f"[SAM] Found {len(images)} image(s) in {images_dir}")
    print(f"[SAM] Output → {output_dir}\n")
    print(f"[SAM] Filter settings:")
    print(f"      area     : {args.min_area} – {args.max_area} px")
    print(f"      aspect   : {args.min_aspect} – {args.max_aspect} (w/h)")
    print(f"      border   : {args.border_pad} px margin")
    print(f"      NMS IoU  : {args.nms_thresh}\n")

    # ── Load SAM ──────────────────────────────────────────────────────────────
    mask_gen = load_sam_generator(checkpoint, args.model_type, args.device, args)

    # ── Process images ────────────────────────────────────────────────────────
    summaries: list[dict] = []
    for i, img_path in enumerate(images, 1):
        print(f"[{i:3d}/{len(images)}] {img_path.name}")
        summary = process_image(img_path, mask_gen, output_dir, args)
        summaries.append(summary)

    # ── Final summary ─────────────────────────────────────────────────────────
    total_regions = sum(s["regions"] for s in summaries)
    processed     = sum(1 for s in summaries if s["status"] == "ok")
    skipped       = sum(1 for s in summaries if s["status"] == "cached")

    print(f"\n{'─'*60}")
    print(f"[SAM] Done.")
    print(f"      Processed : {processed} image(s)")
    print(f"      Cached    : {skipped} image(s) (already had regions)")
    print(f"      Regions   : {total_regions} total across all images")
    print(f"\n[SAM] Next step:")
    print(f"      python scripts/annotate.py")
    print(f"{'─'*60}\n")


if __name__ == "__main__":
    main()

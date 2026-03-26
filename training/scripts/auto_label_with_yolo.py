#!/usr/bin/env python3
"""
auto_label_with_yolo.py — Step 3 of the SAM → YOLO bootstrap pipeline
───────────────────────────────────────────────────────────────────────
After an initial training round, this script replaces the SAM-generated
labels with higher-quality YOLO predictions.  Re-training on the improved
labels typically yields a significantly better model.

Bootstrap loop (zero manual annotation required):
    Step 1  python scripts/sam_auto_annotate.py       # SAM labels → data/
    Step 2  python training/train.py                  # train YOLO on SAM labels
    Step 3  python scripts/auto_label_with_yolo.py    # replace labels with YOLO
    Step 4  python training/train.py                  # retrain on improved labels
    Step 5  (repeat Steps 3–4 as many times as desired)

How it works:
    • Scans data/images/train/ and data/images/val/ for all images.
    • Runs trained YOLO inference on each image.
    • Keeps only detections above --conf threshold.
    • Writes new YOLO-format labels to the matching data/labels/<split>/ path,
      overwriting whatever was there before (SAM or a previous YOLO pass).
    • Optional: --keep-undetected preserves the previous label when YOLO
      finds nothing (useful early in training when the model is immature).

GCS support:
    --weights gs://my-bucket/models/best.pt
    --data    gs://my-bucket/data/
    (labels are written back to GCS automatically)

Usage (from project root):
    python scripts/auto_label_with_yolo.py
    python scripts/auto_label_with_yolo.py --weights weights/best.pt --conf 0.3
    python scripts/auto_label_with_yolo.py --data data/ --keep-undetected

    # GCS
    python scripts/auto_label_with_yolo.py \
        --weights gs://my-bucket/models/best.pt \
        --data    gs://my-bucket/data/

Requirements:
    pip install -r requirements-training.txt    (ultralytics + GCS support)
    weights/best.pt must exist                  (run training/train.py first)
"""

from __future__ import annotations

import argparse
import logging
import os
import shutil
import sys
import tempfile
from pathlib import Path

# ── Project root ──────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# ── Logging ───────────────────────────────────────────────────────────────────
log = logging.getLogger("yolo_label")

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
        "--weights", default=None,
        help=(
            "Path to trained YOLO weights (.pt).  "
            "Accepts local path or gs://bucket/blob."
        ),
    )
    p.add_argument(
        "--data", default=None,
        help=(
            "YOLO dataset root.  Must contain images/train/ and/or images/val/.  "
            "Accepts local path or gs://bucket/prefix."
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
    p.add_argument(
        "--log-level", default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity.",
    )
    # ── Environment-variable defaults (Vertex AI / CI) ────────────────────────
    p.set_defaults(
        weights = os.environ.get("YOLO_WEIGHTS_PATH", "weights/best.pt"),
        data    = os.environ.get("YOLO_DATA_PATH",    "data"),
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

    # ── Logging setup (Vertex AI + Docker compatible) ─────────────────────────
    from training.scripts.gcs_utils import setup_logging, is_gcs_path, resolve_input_path, \
        resolve_input_file, upload_gcs_dir, download_gcs_dir
    setup_logging(level=getattr(logging, args.log_level.upper(), logging.INFO))

    log.info("=" * 60)
    log.info("[YOLO-Label] Step 3 — YOLO Re-Labelling starting")
    log.info("[YOLO-Label] Weights          : %s", args.weights)
    log.info("[YOLO-Label] Dataset          : %s", args.data)
    log.info("[YOLO-Label] Conf threshold   : %s", args.conf)
    log.info("[YOLO-Label] Device           : %s", args.device)
    log.info("[YOLO-Label] Keep-undetected  : %s", args.keep_undetected)
    log.info("=" * 60)

    # ── GCS / local path resolution ───────────────────────────────────────────
    _tmpdir = tempfile.mkdtemp(prefix="pcb_yolabel_")
    tmp     = Path(_tmpdir)

    try:
        # -- Weights: download from GCS if needed -----------------------------
        weights_path = resolve_input_file(args.weights, tmp, Path(args.weights).name)
        if not weights_path.is_absolute() and not is_gcs_path(args.weights):
            weights_path = PROJECT_ROOT / weights_path

        # -- Data dir: download from GCS if needed ----------------------------
        data_is_gcs = is_gcs_path(args.data)
        if data_is_gcs:
            data_dir = tmp / "data"
            data_dir.mkdir()
            download_gcs_dir(args.data, data_dir)
        else:
            data_dir = Path(args.data)
            if not data_dir.is_absolute():
                data_dir = PROJECT_ROOT / data_dir

        # ── Validate inputs ───────────────────────────────────────────────────
        if not weights_path.exists():
            log.error(
                "[YOLO-Label] Weights not found: %s\n"
                "             Run training/train.py first to produce weights/best.pt.",
                weights_path,
            )
            sys.exit(1)

        # ── Collect all images from train + val splits ────────────────────────
        img_paths: list[Path] = []
        for split in ("train", "val"):
            split_dir = data_dir / "images" / split
            if split_dir.exists():
                img_paths.extend(
                    p for p in sorted(split_dir.iterdir())
                    if p.suffix.lower() in IMAGE_EXTS
                )

        if not img_paths:
            log.error(
                "[YOLO-Label] No images found under %s\n"
                "             Run scripts/sam_auto_annotate.py first.",
                data_dir / "images",
            )
            sys.exit(1)

        n_train = sum(1 for p in img_paths if "train" in p.parts)
        n_val   = sum(1 for p in img_paths if "val"   in p.parts)
        log.info(
            "[YOLO-Label] Images found: %d  (train=%d, val=%d)",
            len(img_paths), n_train, n_val,
        )

        # ── Load YOLO model ───────────────────────────────────────────────────
        try:
            from ultralytics import YOLO
        except ImportError:
            log.error(
                "[YOLO-Label] ultralytics not installed.\n"
                "             Install:  pip install -r requirements-training.txt"
            )
            sys.exit(1)

        log.info("[YOLO-Label] Loading model: %s", weights_path)
        model = YOLO(str(weights_path))
        log.info("[YOLO-Label] Model loaded. Running inference on %d images ...", len(img_paths))

        # ── Inference loop ────────────────────────────────────────────────────
        replaced     = 0
        kept_sam     = 0
        total_dets   = 0
        empty_labels = 0

        for i, img_path in enumerate(img_paths, 1):
            if i % 50 == 0 or i == len(img_paths):
                log.info("[YOLO-Label] Progress: %d / %d images", i, len(img_paths))

            label_path = image_to_label_path(img_path, data_dir)

            results = model.predict(
                source  = str(img_path),
                conf    = args.conf,
                device  = args.device,
                save    = False,
                verbose = False,
            )[0]

            # ── Extract detections above confidence threshold ─────────────────
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

            # ── Handle zero-detection case ────────────────────────────────────
            if not detections:
                if args.keep_undetected and label_path.exists():
                    kept_sam += 1
                    continue   # leave the existing label unchanged
                # Overwrite with empty label → background / negative example
                empty_labels += 1

            write_labels(label_path, detections)
            replaced   += 1
            total_dets += len(detections)

        # ── Summary ───────────────────────────────────────────────────────────
        avg = total_dets / replaced if replaced > 0 else 0.0

        log.info("─" * 60)
        log.info("[YOLO-Label] Relabelling complete.")
        log.info("[YOLO-Label] Labels written      : %d", replaced)
        log.info("[YOLO-Label] Total detections    : %d", total_dets)
        log.info("[YOLO-Label] Avg detections/img  : %.1f", avg)
        if empty_labels:
            log.info("[YOLO-Label] Empty (background)  : %d", empty_labels)
        if kept_sam:
            log.info("[YOLO-Label] SAM labels kept     : %d  (--keep-undetected)", kept_sam)

        # ── Upload labels back to GCS (if data was a GCS path) ────────────────
        if data_is_gcs:
            labels_local = data_dir / "labels"
            labels_gcs   = args.data.rstrip("/") + "/labels"
            log.info("[YOLO-Label] Uploading updated labels to GCS: %s", labels_gcs)
            upload_gcs_dir(labels_local, labels_gcs)
            log.info("[YOLO-Label] GCS upload complete.")

        log.info("[YOLO-Label] Next step:")
        log.info("[YOLO-Label]   python training/train.py --data training/data.yaml")
        log.info("─" * 60)

    finally:
        shutil.rmtree(_tmpdir, ignore_errors=True)


if __name__ == "__main__":
    main()

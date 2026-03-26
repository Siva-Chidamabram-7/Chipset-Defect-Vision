#!/usr/bin/env python3
"""
check_pipeline.py — Pipeline health checker
─────────────────────────────────────────────
Validates every phase of the SAM + YOLO dataset pipeline and reports
exactly which step has been completed, which is broken, and what to
run next to unblock the pipeline.

Phases checked:
    0  SAM checkpoint      — weights/sam_vit_b.pth (or vit_l / vit_h)
    1  Raw images          — raw_data/images/ has ≥1 image
    2  SAM region JSON     — raw_data/regions/ has JSON files
    3  Raw labels          — raw_data/labels/ has .txt files
    4  Train/val split     — data/images/train|val populated
    5  YOLO base weights   — weights/yolov8n.pt
    6  Fine-tuned weights  — weights/best.pt

Additional cross-checks:
    • Region JSON ↔ image alignment (missing counterparts)
    • Label ↔ image alignment in raw_data/
    • Label ↔ image alignment in data/train+val
    • Images with empty label files (background-only warning)

Usage (from project root):
    python training/scripts/check_pipeline.py
    python training/scripts/check_pipeline.py --verbose
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# ── Project root ──────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[2]

IMAGE_EXTS  = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
SAM_WEIGHTS = ["sam_vit_b.pth", "sam_vit_l.pth", "sam_vit_h.pth"]


# ═════════════════════════════════════════════════════════════════════════════
# CLI
# ═════════════════════════════════════════════════════════════════════════════

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Phase-by-phase health check for the SAM + YOLO pipeline.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--verbose", "-v", action="store_true",
        help="Print per-file details for mismatches.",
    )
    return p.parse_args()


# ═════════════════════════════════════════════════════════════════════════════
# Utility helpers
# ═════════════════════════════════════════════════════════════════════════════

PASS = "  [OK]  "
WARN = "  [WARN]"
FAIL = "  [FAIL]"
INFO = "         "


def image_stems(directory: Path) -> set[str]:
    """Return the set of stems of all image files in directory (non-recursive)."""
    return {p.stem for p in directory.iterdir()
            if p.is_file() and p.suffix.lower() in IMAGE_EXTS} \
        if directory.exists() else set()


def image_stems_recursive(directory: Path) -> set[str]:
    """Return the set of stems of all image files under directory (recursive)."""
    return {p.stem for p in directory.rglob("*")
            if p.is_file() and p.suffix.lower() in IMAGE_EXTS} \
        if directory.exists() else set()


def label_stems(directory: Path) -> set[str]:
    """Return the set of stems of all .txt label files in directory (non-recursive)."""
    return {p.stem for p in directory.iterdir()
            if p.is_file() and p.suffix == ".txt" and p.name != ".gitkeep"} \
        if directory.exists() else set()


def count_images(directory: Path, recursive: bool = False) -> int:
    if not directory.exists():
        return 0
    if recursive:
        return sum(1 for p in directory.rglob("*")
                   if p.is_file() and p.suffix.lower() in IMAGE_EXTS)
    return sum(1 for p in directory.iterdir()
               if p.is_file() and p.suffix.lower() in IMAGE_EXTS)


def empty_label_count(directory: Path) -> int:
    """Count label .txt files that are zero bytes (background / skipped images)."""
    if not directory.exists():
        return 0
    return sum(1 for p in directory.iterdir()
               if p.is_file() and p.suffix == ".txt" and p.stat().st_size == 0)


# ═════════════════════════════════════════════════════════════════════════════
# Phase checks
# ═════════════════════════════════════════════════════════════════════════════

class PipelineChecker:
    def __init__(self, verbose: bool = False) -> None:
        self.verbose  = verbose
        self.warnings = 0
        self.failures = 0
        self.results:  list[str] = []

    # ── Output helpers ────────────────────────────────────────────────────────

    def ok(self, msg: str) -> None:
        self.results.append(f"{PASS}{msg}")

    def warn(self, msg: str, detail: list[str] | None = None) -> None:
        self.warnings += 1
        self.results.append(f"{WARN}{msg}")
        if detail and self.verbose:
            for line in detail:
                self.results.append(f"{INFO}  {line}")

    def fail(self, msg: str, fix: str | None = None) -> None:
        self.failures += 1
        self.results.append(f"{FAIL}{msg}")
        if fix:
            self.results.append(f"{INFO}Fix: {fix}")

    def header(self, title: str) -> None:
        self.results.append("")
        self.results.append(f"  {'─'*50}")
        self.results.append(f"  Phase {title}")
        self.results.append(f"  {'─'*50}")

    # ── Phase 0: SAM checkpoint ───────────────────────────────────────────────

    def check_sam_checkpoint(self) -> None:
        self.header("0 — SAM checkpoint  (weights/sam_vit_*.pth)")
        weights_dir = PROJECT_ROOT / "weights"
        found = [w for w in SAM_WEIGHTS if (weights_dir / w).exists()]
        if found:
            for w in found:
                size_mb = (weights_dir / w).stat().st_size // 1_048_576
                self.ok(f"{w}  ({size_mb} MB)")
        else:
            self.fail(
                "No SAM checkpoint found in weights/",
                fix=(
                    "Download one and place it in weights/:\n"
                    f"{INFO}  wget https://dl.fbaipublicfiles.com/segment_anything/"
                    "sam_vit_b_01ec64.pth -O weights/sam_vit_b.pth"
                ),
            )

    # ── Phase 1: Raw images ───────────────────────────────────────────────────

    def check_raw_images(self) -> None:
        self.header("1 — Raw images  (raw_data/images/)")
        raw_images_dir = PROJECT_ROOT / "raw_data" / "images"
        n = count_images(raw_images_dir)
        if n == 0:
            self.fail(
                "raw_data/images/ is empty — no source images found.",
                fix="Copy your PCB images into raw_data/images/\n"
                    f"{INFO}  If they are in data/images/train|val by mistake, run:\n"
                    f"{INFO}    python scripts/setup_data.py --execute",
            )
        else:
            self.ok(f"{n} image(s) in raw_data/images/")

    # ── Phase 2: SAM region JSONs ─────────────────────────────────────────────

    def check_regions(self) -> None:
        self.header("2 — SAM region proposals  (raw_data/regions/*.json)")
        regions_dir    = PROJECT_ROOT / "raw_data" / "regions"
        raw_images_dir = PROJECT_ROOT / "raw_data" / "images"

        json_files = sorted(regions_dir.glob("*.json")) if regions_dir.exists() else []

        if not json_files:
            self.fail(
                "No region JSON files found in raw_data/regions/.",
                fix="python scripts/generate_regions.py",
            )
            return

        self.ok(f"{len(json_files)} region JSON file(s) found.")

        # Cross-check: JSON ↔ image
        json_stems  = {j.stem for j in json_files}
        img_stems   = image_stems(raw_images_dir)
        orphan_json = sorted(json_stems - img_stems)   # JSON with no image
        missing_json = sorted(img_stems - json_stems)  # image with no JSON

        if orphan_json:
            self.warn(
                f"{len(orphan_json)} JSON file(s) have no matching image in raw_data/images/.",
                detail=orphan_json[:10],
            )
        if missing_json:
            self.warn(
                f"{len(missing_json)} image(s) in raw_data/images/ have no region JSON yet.",
                detail=missing_json[:10],
            )
            self.results.append(f"{INFO}Fix: python scripts/generate_regions.py "
                                 "(new images will be processed)")

        # Sanity: total regions across all JSON
        total_regions = 0
        for jf in json_files:
            try:
                data = json.loads(jf.read_text())
                total_regions += len(data.get("regions", []))
            except Exception:
                self.warn(f"Could not parse {jf.name}")
        self.ok(f"{total_regions} total region proposals across all JSON files.")

    # ── Phase 3: Raw labels ───────────────────────────────────────────────────

    def check_raw_labels(self) -> None:
        self.header("3 — Annotation labels  (raw_data/labels/*.txt)")
        labels_dir     = PROJECT_ROOT / "raw_data" / "labels"
        raw_images_dir = PROJECT_ROOT / "raw_data" / "images"
        regions_dir    = PROJECT_ROOT / "raw_data" / "regions"

        lbl_stems = label_stems(labels_dir)
        if not lbl_stems:
            self.fail(
                "No label .txt files found in raw_data/labels/.",
                fix="python scripts/annotate.py",
            )
            return

        self.ok(f"{len(lbl_stems)} label file(s) in raw_data/labels/.")

        # Cross-check: label ↔ image
        img_stems    = image_stems(raw_images_dir)
        orphan_lbls  = sorted(lbl_stems - img_stems)
        missing_lbls = sorted(img_stems - lbl_stems)

        if orphan_lbls:
            self.warn(
                f"{len(orphan_lbls)} label file(s) have no matching image.",
                detail=orphan_lbls[:10],
            )
        if missing_lbls:
            self.warn(
                f"{len(missing_lbls)} image(s) have no label file yet.",
                detail=missing_lbls[:10],
            )
            self.results.append(f"{INFO}Fix: python scripts/annotate.py "
                                 "(unlabeled images will be shown)")

        # Empty label files (background-only images are OK but worth noting)
        n_empty = empty_label_count(labels_dir)
        if n_empty:
            self.warn(
                f"{n_empty} label file(s) are empty "
                "(background-only / fully-skipped images — this is fine).",
            )

        # Coverage: labeled vs total
        n_regions_labeled = sum(
            1 for lf in (labels_dir.iterdir() if labels_dir.exists() else [])
            if lf.is_file() and lf.suffix == ".txt" and lf.stat().st_size > 0
        )
        self.ok(f"{n_regions_labeled} image(s) have at least one labeled region.")

    # ── Phase 4: Train / val split ────────────────────────────────────────────

    def check_split(self) -> None:
        self.header("4 — Train/val split  (data/images/train|val)")
        train_img = PROJECT_ROOT / "data" / "images" / "train"
        val_img   = PROJECT_ROOT / "data" / "images" / "val"
        train_lbl = PROJECT_ROOT / "data" / "labels" / "train"
        val_lbl   = PROJECT_ROOT / "data" / "labels" / "val"

        n_train = count_images(train_img)
        n_val   = count_images(val_img)

        if n_train == 0 and n_val == 0:
            self.fail(
                "data/images/train/ and data/images/val/ are both empty.",
                fix="python training/prepare_dataset.py",
            )
            return

        if n_train == 0:
            self.fail(
                "data/images/train/ is empty — no training images.",
                fix="python training/prepare_dataset.py",
            )
        else:
            self.ok(f"{n_train} image(s) in data/images/train/")

        if n_val == 0:
            self.warn(
                "data/images/val/ is empty — validation set missing.",
            )
            self.results.append(f"{INFO}Fix: python training/prepare_dataset.py")
        else:
            self.ok(f"{n_val} image(s) in data/images/val/")

        # Label alignment in split dirs
        for split, img_dir, lbl_dir in [
            ("train", train_img, train_lbl),
            ("val",   val_img,   val_lbl),
        ]:
            img_s = image_stems(img_dir)
            lbl_s = label_stems(lbl_dir)
            missing = sorted(img_s - lbl_s)
            orphan  = sorted(lbl_s - img_s)
            if missing:
                self.warn(
                    f"{len(missing)} image(s) in {split}/ have no label file.",
                    detail=missing[:10],
                )
            if orphan:
                self.warn(
                    f"{len(orphan)} label file(s) in {split}/ have no matching image.",
                    detail=orphan[:10],
                )

    # ── Phase 5: YOLO base weights ────────────────────────────────────────────

    def check_yolo_base(self) -> None:
        self.header("5 — YOLO base weights  (weights/yolov8n.pt)")
        path = PROJECT_ROOT / "weights" / "yolov8n.pt"
        if path.exists():
            size_mb = path.stat().st_size // 1_048_576
            self.ok(f"yolov8n.pt  ({size_mb} MB)")
        else:
            self.fail(
                "weights/yolov8n.pt not found.",
                fix=(
                    "Download once on a connected machine:\n"
                    f"{INFO}  python -c \"from ultralytics import YOLO; YOLO('yolov8n.pt')\"\n"
                    f"{INFO}  # then copy yolov8n.pt into weights/"
                ),
            )

    # ── Phase 6: Fine-tuned weights ───────────────────────────────────────────

    def check_finetuned(self) -> None:
        self.header("6 — Fine-tuned weights  (weights/best.pt)")
        path = PROJECT_ROOT / "weights" / "best.pt"
        if path.exists():
            size_mb = path.stat().st_size // 1_048_576
            self.ok(f"best.pt  ({size_mb} MB)  — fine-tuned model ready for inference.")
        else:
            self.warn(
                "weights/best.pt not found — inference cannot start until best.pt is available.",
            )
            self.results.append(f"{INFO}To train: python training/train.py")

    # ── Run all ───────────────────────────────────────────────────────────────

    def run(self) -> None:
        self.check_sam_checkpoint()
        self.check_raw_images()
        self.check_regions()
        self.check_raw_labels()
        self.check_split()
        self.check_yolo_base()
        self.check_finetuned()


# ═════════════════════════════════════════════════════════════════════════════
# Entry point
# ═════════════════════════════════════════════════════════════════════════════

def main() -> None:
    args    = parse_args()
    checker = PipelineChecker(verbose=args.verbose)

    print()
    print("═" * 60)
    print("  Chipset Defect Vision — Pipeline Health Check")
    print("═" * 60)

    checker.run()

    for line in checker.results:
        print(line)

    print()
    print("═" * 60)

    if checker.failures == 0 and checker.warnings == 0:
        print("  All checks passed.")
    else:
        if checker.failures:
            print(f"  Failures : {checker.failures}  (pipeline blocked — fix these first)")
        if checker.warnings:
            print(f"  Warnings : {checker.warnings}  (non-blocking — review recommended)")

    if not args.verbose and (checker.failures or checker.warnings):
        print("  Tip: run with --verbose for per-file details.")

    print("═" * 60)
    print()

    sys.exit(1 if checker.failures else 0)


if __name__ == "__main__":
    main()

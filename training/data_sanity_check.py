"""
training/data_sanity_check.py
─────────────────────────────
Run BEFORE training to find corrupted images and broken labels that cause
OpenCV / DataLoader crashes.

Usage
-----
    python -m training.data_sanity_check              # default: data/
    python -m training.data_sanity_check --root path/to/dataset

Exit code
---------
    0  →  dataset is clean
    1  →  at least one problem was found
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
SPLITS = ("train", "val")

# ── helpers ──────────────────────────────────────────────────────────────────

def _bold(text: str) -> str:
    return f"\033[1m{text}\033[0m"

def _red(text: str) -> str:
    return f"\033[91m{text}\033[0m"

def _yellow(text: str) -> str:
    return f"\033[93m{text}\033[0m"

def _green(text: str) -> str:
    return f"\033[92m{text}\033[0m"

def _sep(char: str = "─", width: int = 72) -> str:
    return char * width


# ── image checks ─────────────────────────────────────────────────────────────

def check_image(img_path: Path) -> str | None:
    """Return an error string, or None if the image is readable."""
    try:
        import cv2  # type: ignore
    except ImportError:
        return None  # cv2 not installed — skip pixel-level checks

    img = cv2.imread(str(img_path))
    if img is None:
        return "cv2.imread() returned None — file is missing, empty, or corrupt"

    h, w = img.shape[:2]
    if h == 0 or w == 0:
        return f"zero-dimension image ({w}×{h})"

    return None


# ── label checks ─────────────────────────────────────────────────────────────

def check_label(label_path: Path) -> list[str]:
    """Return a list of problems found in the label file (empty = clean)."""
    problems: list[str] = []

    raw = label_path.read_text(encoding="utf-8", errors="replace").strip()
    if not raw:
        problems.append("file is empty — no annotations")
        return problems

    for line_no, line in enumerate(raw.splitlines(), start=1):
        parts = line.strip().split()
        if len(parts) != 5:
            problems.append(
                f"line {line_no}: expected 5 values, got {len(parts)} → {line!r}"
            )
            continue

        cls_str, *coords_str = parts

        # Class ID
        try:
            cls_id = int(cls_str)
            if cls_id < 0:
                problems.append(f"line {line_no}: negative class ID ({cls_id})")
        except ValueError:
            problems.append(f"line {line_no}: class ID is not an integer ({cls_str!r})")

        # Coordinates: x_centre, y_centre, width, height  (all in [0, 1])
        for i, coord_str in enumerate(coords_str):
            name = ("x_c", "y_c", "w", "h")[i]
            try:
                val = float(coord_str)
            except ValueError:
                problems.append(
                    f"line {line_no}: {name} is not a float ({coord_str!r})"
                )
                continue

            if math.isnan(val) or math.isinf(val):
                problems.append(f"line {line_no}: {name} is NaN/Inf ({coord_str})")
            elif not (0.0 <= val <= 1.0):
                problems.append(
                    f"line {line_no}: {name}={val:.6f} is outside [0, 1]"
                )

    return problems


# ── per-split scan ────────────────────────────────────────────────────────────

def scan_split(dataset_root: Path, split: str) -> tuple[int, int]:
    """Scan one split (train or val).  Returns (issue_count, total_images)."""
    image_dir = dataset_root / "images" / split
    label_dir = dataset_root / "labels" / split

    print(f"\n{_bold(f'[ {split.upper()} ]')}")
    print(_sep())

    if not image_dir.exists():
        print(_red(f"  MISSING image directory: {image_dir}"))
        return 1, 0
    if not label_dir.exists():
        print(_red(f"  MISSING label directory: {label_dir}"))
        return 1, 0

    image_files = sorted(
        p for p in image_dir.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS
    )
    label_files = {p.stem: p for p in label_dir.iterdir() if p.suffix == ".txt"}

    if not image_files:
        print(_yellow(f"  WARNING: no images found in {image_dir}"))
        return 1, 0

    issues = 0

    for img_path in image_files:
        stem = img_path.stem

        # ── image readability ────────────────────────────────────────────
        err = check_image(img_path)
        if err:
            print(_red(f"  BROKEN IMAGE : {img_path}"))
            print(_red(f"    reason     : {err}"))
            issues += 1

        # ── label existence ──────────────────────────────────────────────
        if stem not in label_files:
            print(_yellow(f"  MISSING LABEL: {img_path.name}  →  expected {label_dir / stem}.txt"))
            issues += 1
            continue

        label_path = label_files[stem]

        # ── label content ────────────────────────────────────────────────
        label_problems = check_label(label_path)
        if label_problems:
            print(_red(f"  INVALID LABEL: {label_path}"))
            for prob in label_problems:
                print(_red(f"    ↳ {prob}"))
            issues += 1

    # ── orphan labels (label exists but no matching image) ───────────────
    image_stems = {p.stem for p in image_files}
    for stem, label_path in sorted(label_files.items()):
        if stem not in image_stems:
            print(_yellow(f"  ORPHAN LABEL : {label_path}  (no matching image)"))
            issues += 1

    total = len(image_files)
    if issues == 0:
        print(_green(f"  ✓ All {total} image/label pairs are clean."))
    else:
        print(f"\n  {_red(str(issues))} problem(s) found across {total} images.")

    return issues, total


# ── entry point ───────────────────────────────────────────────────────────────

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Sanity-check dataset images and YOLO labels before training."
    )
    parser.add_argument(
        "--root",
        default=str(ROOT / "data"),
        help="Dataset root directory (must contain images/{train,val} and labels/{train,val}).",
    )
    args = parser.parse_args()

    dataset_root = Path(args.root).resolve()

    print(_sep("═"))
    print(_bold("  CHIPSET DEFECT — DATASET SANITY CHECK"))
    print(f"  Root: {dataset_root}")
    print(_sep("═"))

    total_issues = 0
    total_images = 0

    for split in SPLITS:
        issues, images = scan_split(dataset_root, split)
        total_issues += issues
        total_images += images

    print(f"\n{_sep('═')}")
    if total_issues == 0:
        print(_green(f"  RESULT: CLEAN — {total_images} images checked, no problems found."))
        print(_green("  Safe to run training."))
    else:
        print(_red(f"  RESULT: {total_issues} PROBLEM(S) across {total_images} images."))
        print(_red("  Fix or remove the flagged files before running training."))
    print(_sep("═"))

    return 0 if total_issues == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())

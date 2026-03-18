#!/usr/bin/env python3
"""
annotate.py — Phase 2 of the Hybrid SAM + YOLO Pipeline
─────────────────────────────────────────────────────────
Interactive OpenCV annotation tool.

Reads SAM-generated region proposals (from scripts/generate_regions.py),
lets the user assign a class label to each region, and saves labels in
YOLO format (.txt) directly to raw_data/labels/.

Usage (run from project root):
    python scripts/annotate.py

    python scripts/annotate.py \\
        --regions  raw_data/regions \\
        --images   raw_data/images \\
        --output   raw_data/labels \\
        --zoom     3

Keyboard controls
─────────────────
    G  →  label as "good"   (class 0)
    D  →  label as "defect" (class 1)
    S  →  skip this region  (not saved to label file)
    B  →  undo last label in current image
    Q  →  save current image and quit
    Esc→  same as Q

After labeling all regions in an image the labels are saved automatically
and the tool moves to the next image.

Resume behaviour
────────────────
If a .txt label file already exists for an image it is skipped unless
--overwrite is passed.  This lets you safely interrupt and resume a session.

Output (raw_data/labels/<stem>.txt)
────────────────────────────────────
Each line is one labeled object in YOLO format:
    <class_id> <x_center> <y_center> <width> <height>

All coordinates are normalised 0–1 relative to image dimensions.

Classes:
    0 → good
    1 → defect
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

# ── Project root ──────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# ── Class config ──────────────────────────────────────────────────────────────
CLASSES = {0: "good", 1: "defect"}

# Colors in BGR
COLOR_GOOD    = (34,  197,  94)   # green
COLOR_DEFECT  = (239,  68,  68)   # red
COLOR_SKIP    = (120, 120, 120)   # grey
COLOR_CURRENT = (0,   220, 220)   # yellow-cyan — current unlabeled region
COLOR_TEXT    = (255, 255, 255)   # white text

# Key map
KEY_GOOD   = ord('g')
KEY_DEFECT = ord('d')
KEY_SKIP   = ord('s')
KEY_BACK   = ord('b')
KEY_QUIT   = ord('q')
KEY_ESC    = 27


# ═════════════════════════════════════════════════════════════════════════════
# CLI
# ═════════════════════════════════════════════════════════════════════════════

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Phase 2 — Interactive region labeling for PCB solder joints.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--regions",   default="raw_data/regions",
                   help="Directory containing SAM region JSON files.")
    p.add_argument("--images",    default="raw_data/images",
                   help="Directory containing the original PCB images.")
    p.add_argument("--output",    default="raw_data/labels",
                   help="Output directory for YOLO .txt label files.")
    p.add_argument("--zoom",      type=int, default=3,
                   help="Zoom multiplier for the region crop preview (1–8).")
    p.add_argument("--pad",       type=int, default=24,
                   help="Pixel padding around each region in the crop view.")
    p.add_argument("--overwrite", action="store_true",
                   help="Re-annotate images that already have a label file.")
    return p.parse_args()


# ═════════════════════════════════════════════════════════════════════════════
# Display helpers
# ═════════════════════════════════════════════════════════════════════════════

def _label_color(label: int | None) -> tuple[int, int, int]:
    if label == 0:
        return COLOR_GOOD
    if label == 1:
        return COLOR_DEFECT
    if label == -1:
        return COLOR_SKIP
    return COLOR_CURRENT


def draw_overview(base: np.ndarray, regions: list[dict],
                  current_idx: int, labeled: list[dict | None],
                  img_name: str) -> np.ndarray:
    """
    Build the left-panel overview:  full PCB image with all boxes drawn.
    Already-labeled boxes use their class colour; the current box is
    highlighted in bright cyan with a thick border.
    """
    img = base.copy()
    h, w = img.shape[:2]

    # Draw all previously labeled / skipped boxes
    for i, region in enumerate(regions):
        if i == current_idx or labeled[i] is None:
            continue
        x1, y1, x2, y2 = region["bbox_abs"]
        label = labeled[i]["label"] if labeled[i] else None
        color = _label_color(label)
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        name  = CLASSES.get(label, "skip") if label is not None else "skip"
        cv2.putText(img, name, (x1 + 2, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1, cv2.LINE_AA)

    # Draw current box — thick cyan
    if current_idx < len(regions):
        x1, y1, x2, y2 = regions[current_idx]["bbox_abs"]
        cv2.rectangle(img, (x1, y1), (x2, y2), COLOR_CURRENT, 3)
        cv2.putText(img, "?", (x1 + 2, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, COLOR_CURRENT, 1, cv2.LINE_AA)

    # Status bar at bottom
    done    = sum(1 for lbl in labeled if lbl is not None)
    n_good  = sum(1 for lbl in labeled
                  if lbl is not None and lbl.get("label") == 0)
    n_def   = sum(1 for lbl in labeled
                  if lbl is not None and lbl.get("label") == 1)
    n_skip  = sum(1 for lbl in labeled
                  if lbl is not None and lbl.get("label") == -1)
    total   = len(regions)

    bar_h = 44
    bar = np.zeros((bar_h, w, 3), dtype=np.uint8)
    bar[:] = (30, 30, 30)

    info1 = f"  {img_name}   Region {min(current_idx+1, total)}/{total}"
    info2 = (f"  good={n_good}  defect={n_def}  skip={n_skip}  "
             f"done={done}/{total}")

    cv2.putText(bar, info1, (4, 14), cv2.FONT_HERSHEY_SIMPLEX,
                0.45, (200, 200, 200), 1, cv2.LINE_AA)
    cv2.putText(bar, info2, (4, 32), cv2.FONT_HERSHEY_SIMPLEX,
                0.42, (150, 200, 150), 1, cv2.LINE_AA)

    return np.vstack([img, bar])


def draw_crop(base: np.ndarray, region: dict, zoom: int, pad: int) -> np.ndarray:
    """
    Build the right-panel crop: zoomed view of the current region with
    the bounding box drawn in cyan.
    """
    h, w = base.shape[:2]
    x1, y1, x2, y2 = region["bbox_abs"]

    # Add padding — clamp to image bounds
    cx1 = max(0, x1 - pad)
    cy1 = max(0, y1 - pad)
    cx2 = min(w, x2 + pad)
    cy2 = min(h, y2 + pad)

    crop = base[cy1:cy2, cx1:cx2].copy()
    if crop.size == 0:
        crop = np.zeros((64, 64, 3), dtype=np.uint8)

    # Draw bbox on the crop (offset from top-left of crop)
    bx1, by1 = x1 - cx1, y1 - cy1
    bx2, by2 = x2 - cx1, y2 - cy1
    cv2.rectangle(crop, (bx1, by1), (bx2, by2), COLOR_CURRENT, 2)

    # Zoom
    new_h = max(crop.shape[0] * zoom, 64)
    new_w = max(crop.shape[1] * zoom, 64)
    zoomed = cv2.resize(crop, (new_w, new_h), interpolation=cv2.INTER_CUBIC)

    # Info overlay on zoomed crop
    area   = region.get("area", 0)
    stab   = region.get("stability_score", 0)
    idx    = region.get("id", 0)
    info   = f"region {idx}  area={area}  stab={stab:.2f}"
    cv2.putText(zoomed, info, (4, 16),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, COLOR_TEXT, 1, cv2.LINE_AA)

    return zoomed


def draw_legend(width: int) -> np.ndarray:
    """Bottom legend strip shown under the crop panel."""
    bar = np.zeros((60, width, 3), dtype=np.uint8)
    bar[:] = (20, 20, 20)
    lines = [
        "  G = good   D = defect   S = skip",
        "  B = undo   Q / Esc = save & quit",
    ]
    for i, line in enumerate(lines):
        cv2.putText(bar, line, (4, 18 + i * 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 210, 180), 1, cv2.LINE_AA)
    return bar


# ═════════════════════════════════════════════════════════════════════════════
# Label saving
# ═════════════════════════════════════════════════════════════════════════════

def save_labels(output_path: Path, labeled: list[dict | None]) -> int:
    """
    Write YOLO-format labels for all non-skipped, labeled regions.
    Returns the number of lines written.
    """
    lines = []
    for entry in labeled:
        if entry is None:
            continue
        label = entry.get("label")
        if label not in (0, 1):   # skip (-1) or unlabeled (None)
            continue
        cx, cy, bw, bh = entry["bbox_yolo"]
        lines.append(f"{label} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        f.write("\n".join(lines) + ("\n" if lines else ""))

    return len(lines)


# ═════════════════════════════════════════════════════════════════════════════
# Per-image annotation session
# ═════════════════════════════════════════════════════════════════════════════

def annotate_image(img_path: Path, regions: list[dict],
                   output_path: Path, args: argparse.Namespace) -> dict:
    """
    Interactive annotation loop for a single image.
    Returns a summary dict.
    """
    img_bgr = cv2.imread(str(img_path))
    if img_bgr is None:
        print(f"  [warn] Cannot read image: {img_path.name}", file=sys.stderr)
        return {"file": img_path.name, "labeled": 0, "status": "unreadable"}

    img_h, img_w = img_bgr.shape[:2]

    # Scale overview to at most 700 px tall
    max_h    = 700
    scale    = min(1.0, max_h / img_h)
    disp_h   = int(img_h * scale)
    disp_w   = int(img_w * scale)
    overview_base = cv2.resize(img_bgr, (disp_w, disp_h))

    # Scale region bboxes to display coordinates
    disp_regions: list[dict] = []
    for r in regions:
        x1, y1, x2, y2 = r["bbox_abs"]
        disp_regions.append({
            **r,
            "bbox_abs": [int(x1 * scale), int(y1 * scale),
                         int(x2 * scale), int(y2 * scale)],
        })

    labeled: list[dict | None] = [None] * len(regions)
    current = 0

    cv2.namedWindow("PCB Annotator",    cv2.WINDOW_NORMAL)
    cv2.namedWindow("Current Region",   cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Current Region", 320, 320)

    quit_requested = False

    while current < len(regions):
        # ── Render overview ───────────────────────────────────────────────────
        overview = draw_overview(
            overview_base, disp_regions, current, labeled, img_path.name
        )
        cv2.imshow("PCB Annotator", overview)

        # ── Render crop ───────────────────────────────────────────────────────
        crop_panel = draw_crop(overview_base, disp_regions[current], args.zoom, args.pad)
        legend     = draw_legend(crop_panel.shape[1])
        crop_full  = np.vstack([crop_panel, legend])
        cv2.imshow("Current Region", crop_full)

        # ── Keypress ──────────────────────────────────────────────────────────
        key = cv2.waitKey(0) & 0xFF

        if key in (KEY_QUIT, KEY_ESC):
            quit_requested = True
            break
        elif key == KEY_GOOD:
            labeled[current] = {"label": 0, "bbox_yolo": regions[current]["bbox_yolo"]}
            print(f"    [{current+1:3d}/{len(regions)}] good    ✓")
            current += 1
        elif key == KEY_DEFECT:
            labeled[current] = {"label": 1, "bbox_yolo": regions[current]["bbox_yolo"]}
            print(f"    [{current+1:3d}/{len(regions)}] defect  ✓")
            current += 1
        elif key == KEY_SKIP:
            labeled[current] = {"label": -1, "bbox_yolo": regions[current]["bbox_yolo"]}
            print(f"    [{current+1:3d}/{len(regions)}] skipped")
            current += 1
        elif key == KEY_BACK:
            if current > 0:
                current -= 1
                labeled[current] = None
                print(f"    [{current+1:3d}/{len(regions)}] undo — relabeling")

    cv2.destroyAllWindows()

    # ── Save labels ───────────────────────────────────────────────────────────
    n_saved = save_labels(output_path, labeled)
    n_good  = sum(1 for e in labeled if e and e["label"] == 0)
    n_def   = sum(1 for e in labeled if e and e["label"] == 1)
    n_skip  = sum(1 for e in labeled if e and e["label"] == -1)

    status = "quit" if quit_requested else "done"
    print(f"  → Saved {n_saved} labels  (good={n_good}, defect={n_def}, skip={n_skip})")

    return {
        "file":    img_path.name,
        "labeled": n_saved,
        "status":  status,
    }


# ═════════════════════════════════════════════════════════════════════════════
# Main
# ═════════════════════════════════════════════════════════════════════════════

def main() -> None:
    args = parse_args()

    # ── Resolve paths ─────────────────────────────────────────────────────────
    regions_dir = Path(args.regions)
    images_dir  = Path(args.images)
    output_dir  = Path(args.output)

    if not regions_dir.is_absolute():
        regions_dir = PROJECT_ROOT / regions_dir
    if not images_dir.is_absolute():
        images_dir  = PROJECT_ROOT / images_dir
    if not output_dir.is_absolute():
        output_dir  = PROJECT_ROOT / output_dir

    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Check OpenCV display ──────────────────────────────────────────────────
    # cv2.imshow requires a display (X11/Wayland on Linux, native on Win/macOS).
    # On headless servers, use VNC or run annotate.py on a local workstation.
    try:
        test = np.zeros((2, 2, 3), dtype=np.uint8)
        cv2.imshow("__test__", test)
        cv2.waitKey(1)
        cv2.destroyWindow("__test__")
    except cv2.error:
        print(
            "[Annotator] ERROR: No display detected.\n"
            "            annotate.py requires a graphical desktop (X11/Wayland/macOS/Windows).\n"
            "            Run this script on your local workstation, not inside a headless container.",
            file=sys.stderr,
        )
        sys.exit(1)

    # ── Gather region JSON files ───────────────────────────────────────────────
    json_files = sorted(regions_dir.glob("*.json"))
    if not json_files:
        print(
            f"[Annotator] ERROR: No region JSON files found in {regions_dir}\n"
            f"            Run Phase 1 first:\n"
            f"              python scripts/generate_regions.py",
            file=sys.stderr,
        )
        sys.exit(1)

    print(f"[Annotator] Found {len(json_files)} region file(s) in {regions_dir}")
    print(f"[Annotator] Labels → {output_dir}\n")
    print("  Keys:  G=good  D=defect  S=skip  B=undo  Q/Esc=quit\n")

    # ── Annotate each image ────────────────────────────────────────────────────
    total_labeled = 0
    for i, json_path in enumerate(json_files, 1):
        # Locate the source image
        img_path = images_dir / json_path.stem
        found    = None
        for ext in (".jpg", ".jpeg", ".png", ".bmp", ".webp"):
            candidate = img_path.with_suffix(ext)
            if candidate.exists():
                found = candidate
                break

        if found is None:
            print(f"[{i}/{len(json_files)}] {json_path.stem}: image not found — skipping")
            continue

        label_path = output_dir / (json_path.stem + ".txt")
        if label_path.exists() and not args.overwrite:
            print(f"[{i}/{len(json_files)}] {json_path.stem}: labels exist — skipping "
                  f"(use --overwrite to redo)")
            continue

        # Load regions
        with open(json_path) as f:
            data = json.load(f)
        regions = data.get("regions", [])

        if not regions:
            print(f"[{i}/{len(json_files)}] {json_path.stem}: 0 regions — skipping")
            # Write an empty label file so training doesn't error
            label_path.touch()
            continue

        print(f"[{i}/{len(json_files)}] {json_path.stem}  ({len(regions)} regions)")

        summary = annotate_image(found, regions, label_path, args)
        total_labeled += summary["labeled"]

        if summary["status"] == "quit":
            print("\n[Annotator] Session saved.  Resume by running annotate.py again.")
            break

    # ── Final summary ─────────────────────────────────────────────────────────
    done    = sum(1 for jf in json_files
                  if (output_dir / (jf.stem + ".txt")).exists())
    pending = len(json_files) - done

    print(f"\n{'─'*60}")
    print(f"[Annotator] Done.")
    print(f"            Labeled objects : {total_labeled}")
    print(f"            Images finished : {done}/{len(json_files)}")
    if pending:
        print(f"            Still pending   : {pending} (run again to continue)")
    print(f"\n[Annotator] Next step:")
    print(f"            python training/prepare_dataset.py")
    print(f"{'─'*60}\n")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
setup_data.py — Dataset migration utility
──────────────────────────────────────────
Moves PCB images that were accidentally placed inside the YOLO split
directories (data/images/train/, data/images/val/) back to the correct
raw-data intake directory (raw_data/images/).

Run this ONCE if you added images directly to data/ instead of raw_data/.
After migration, run the full pipeline from Phase 1:

    python scripts/generate_regions.py   # SAM → region proposals
    python scripts/annotate.py           # label regions
    python training/prepare_dataset.py   # train/val split

Usage (from project root):
    python scripts/setup_data.py              # dry-run (shows what would move)
    python scripts/setup_data.py --execute    # actually moves the files
    python scripts/setup_data.py --src data/images --dst raw_data/images --execute
"""

from __future__ import annotations

import argparse
import shutil
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
        description="Migrate images from data/images/train|val → raw_data/images/.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--src", default="data/images",
        help="Source directory to scan (recursively) for images.",
    )
    p.add_argument(
        "--dst", default="raw_data/images",
        help="Destination directory to move images into (flat, no subdirs).",
    )
    p.add_argument(
        "--execute", action="store_true",
        help="Actually move files.  Without this flag the script is a dry-run.",
    )
    return p.parse_args()


# ═════════════════════════════════════════════════════════════════════════════
# Helpers
# ═════════════════════════════════════════════════════════════════════════════

def collect_images(src_dir: Path) -> list[Path]:
    """Return all image files under src_dir, excluding .gitkeep placeholders."""
    images = []
    for path in sorted(src_dir.rglob("*")):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTS:
            images.append(path)
    return images


def unique_dst(dst_dir: Path, filename: str) -> Path:
    """
    Return a destination path that does not already exist.
    If dst_dir/filename.jpg clashes, appends _2, _3, … until unique.
    """
    candidate = dst_dir / filename
    if not candidate.exists():
        return candidate

    stem   = Path(filename).stem
    suffix = Path(filename).suffix
    counter = 2
    while True:
        candidate = dst_dir / f"{stem}_{counter}{suffix}"
        if not candidate.exists():
            return candidate
        counter += 1


# ═════════════════════════════════════════════════════════════════════════════
# Main
# ═════════════════════════════════════════════════════════════════════════════

def main() -> None:
    args = parse_args()

    src_dir = Path(args.src)
    dst_dir = Path(args.dst)

    if not src_dir.is_absolute():
        src_dir = PROJECT_ROOT / src_dir
    if not dst_dir.is_absolute():
        dst_dir = PROJECT_ROOT / dst_dir

    dry_run = not args.execute

    # ── Banner ────────────────────────────────────────────────────────────────
    print("─" * 60)
    print("  Dataset Migration Utility")
    if dry_run:
        print("  MODE: DRY-RUN  (pass --execute to apply changes)")
    else:
        print("  MODE: EXECUTE  (files will be moved)")
    print(f"  src  : {src_dir}")
    print(f"  dst  : {dst_dir}")
    print("─" * 60)

    # ── Validate source ───────────────────────────────────────────────────────
    if not src_dir.exists():
        print(f"\n[Migration] Source directory not found: {src_dir}")
        print("  Nothing to migrate.")
        sys.exit(0)

    images = collect_images(src_dir)
    if not images:
        print(f"\n[Migration] No images found under {src_dir}")
        print("  Nothing to migrate.")
        sys.exit(0)

    print(f"\n[Migration] Found {len(images)} image(s) to migrate:\n")

    # ── Plan moves ────────────────────────────────────────────────────────────
    plan: list[tuple[Path, Path]] = []
    seen_names: set[str] = set()

    for src_path in images:
        filename = src_path.name
        dst_path = unique_dst(dst_dir, filename)
        plan.append((src_path, dst_path))

        tag = " (renamed)" if dst_path.name != filename else ""
        rel_src = src_path.relative_to(PROJECT_ROOT)
        rel_dst = dst_path.relative_to(PROJECT_ROOT)
        print(f"  {rel_src}  →  {rel_dst}{tag}")
        seen_names.add(filename)

    print()

    # ── Execute ───────────────────────────────────────────────────────────────
    if dry_run:
        print(f"[Migration] Dry-run complete.  {len(plan)} file(s) would be moved.")
        print("            Run with --execute to apply.\n")
        sys.exit(0)

    dst_dir.mkdir(parents=True, exist_ok=True)

    moved   = 0
    skipped = 0
    errors  = 0

    for src_path, dst_path in plan:
        try:
            shutil.move(str(src_path), str(dst_path))
            moved += 1
        except Exception as exc:
            print(f"  [error] {src_path.name}: {exc}", file=sys.stderr)
            errors += 1

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"{'─'*60}")
    print(f"[Migration] Done.")
    print(f"            Moved   : {moved}")
    if skipped:
        print(f"            Skipped : {skipped}")
    if errors:
        print(f"            Errors  : {errors}")
    print()
    print("[Migration] Next steps:")
    print("  1. python scripts/generate_regions.py   # SAM → region proposals")
    print("  2. python scripts/annotate.py           # label each region")
    print("  3. python training/prepare_dataset.py   # train/val split → data/")
    print("  4. python training/train.py             # fine-tune YOLO")
    print(f"{'─'*60}\n")


if __name__ == "__main__":
    main()

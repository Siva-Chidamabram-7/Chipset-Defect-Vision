"""
Hard Negative Mining — run AFTER first training.

Usage:
    python training/scripts/mine_hard_negatives.py

What it does:
  1. Runs your trained model on all raw_data/ images at LOW confidence (0.15)
  2. Finds images where the model fires false detections
  3. Copies those images to data/ with EMPTY label files (background)
  4. These hard negatives are then used in the NEXT training round

After running:
  Re-run: python training/train_precision.py
  (Change name= to 'pcb_precision_v2' to avoid overwriting v1)
"""
import shutil
from pathlib import Path
from ultralytics import YOLO

PROJECT_ROOT  = Path(__file__).resolve().parents[2]
RAW_DATA_DIR  = PROJECT_ROOT / "raw_data"
TRAIN_IMG_DIR = PROJECT_ROOT / "data" / "images" / "train"
TRAIN_LBL_DIR = PROJECT_ROOT / "data" / "labels" / "train"

# Use your LATEST best.pt after first training
MODEL_PATH = PROJECT_ROOT / "runs" / "detect" / "pcb_precision_v1" / "weights" / "best.pt"

# Low threshold to catch ALL false positives
CONF_THRESHOLD = 0.15
MAX_HARD_NEGATIVES = 150


def main():
    if not MODEL_PATH.exists():
        print(f"[HNM] Model not found: {MODEL_PATH}")
        print("[HNM] Run training first: python training/train_precision.py")
        return

    model = YOLO(str(MODEL_PATH))

    # Collect all raw images across all class folders
    all_images = []
    for cls_dir in RAW_DATA_DIR.iterdir():
        if not cls_dir.is_dir():
            continue
        for img in cls_dir.iterdir():
            if img.suffix.lower() in ('.jpg', '.jpeg', '.png'):
                all_images.append(img)

    print(f"[HNM] Scanning {len(all_images)} raw images for false positives...")
    print(f"[HNM] Model: {MODEL_PATH}")
    print(f"[HNM] Confidence threshold: {CONF_THRESHOLD}")

    found = 0
    for img_path in all_images:
        if found >= MAX_HARD_NEGATIVES:
            break

        results = model.predict(
            source=str(img_path),
            conf=CONF_THRESHOLD,
            iou=0.45,
            verbose=False,
            save=False,
        )[0]

        if results.boxes is None or len(results.boxes) == 0:
            # Model correctly fires nothing — not a hard negative
            continue

        # Model fired on this image — it's a false positive candidate
        dst_img = TRAIN_IMG_DIR / f"hn_{found:04d}_{img_path.name}"
        dst_lbl = TRAIN_LBL_DIR / f"hn_{found:04d}_{img_path.stem}.txt"

        shutil.copy2(img_path, dst_img)
        dst_lbl.write_text("")  # Empty = no annotations = pure background

        det_classes = [int(b.cls[0]) for b in results.boxes]
        print(f"  [{found+1:3d}] Hard negative: {img_path.parent.name}/{img_path.name} "
              f"(model predicted classes: {det_classes})")
        found += 1

    print(f"\n[HNM] Done. Added {found} hard negatives to training set.")

    if found == 0:
        print("[HNM] No false positives found at conf=0.15 — model is already precise.")
        return

    print("\n[HNM] Next steps:")
    print("  1. Open training/train_precision.py")
    print("  2. Change name='pcb_precision_v1' to name='pcb_precision_v2'")
    print("  3. Re-run: python training/train_precision.py")


if __name__ == "__main__":
    main()

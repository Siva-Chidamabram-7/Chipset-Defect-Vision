"""
train.py — Fine-tune YOLOv8 on the PCB solder defect dataset
─────────────────────────────────────────────────────────────
OFFLINE-FIRST: this script never downloads anything from the internet.
  • The base model (yolov8n.pt) must already exist in weights/.
  • The dataset must already be prepared under data/.

Usage (run from project root):
    python training/train.py
    python training/train.py --epochs 30 --imgsz 640 --device cpu
    python training/train.py --epochs 50 --batch 8  --device 0   # GPU

Outputs:
    weights/best.pt   ← best checkpoint (auto-copied after training)
    weights/last.pt   ← last checkpoint
    runs/detect/<name>/  ← full ultralytics run directory (plots, metrics, etc.)
"""

import argparse
import shutil
import sys
from pathlib import Path

# ── Project root (two levels up from training/train.py) ──────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
WEIGHTS_DIR  = PROJECT_ROOT / "weights"


# ── Pre-flight checks ─────────────────────────────────────────────────────────
def check_model_weights(model_arg: str) -> Path:
    """
    Resolve --model to an absolute Path that exists on disk.

    Rules:
      • If the argument is already an absolute or relative path → use as-is.
      • If it is a bare filename (e.g. "yolov8n.pt") → look in weights/.
      • If the resolved file does not exist → print a helpful error and exit.

    This prevents ultralytics from silently downloading the file when it is
    passed a bare name that doesn't resolve to an existing file.
    """
    candidate = Path(model_arg)

    # Bare name with no directory component → search weights/ first
    if candidate.parent == Path(".") and not candidate.is_absolute():
        local = WEIGHTS_DIR / candidate
        if local.exists():
            return local
        # Don't fall through to auto-download; fail loudly.
        print(
            f"[Train] ERROR: Model weights not found.\n"
            f"        Looked for:  {local}\n"
            f"        To fix:\n"
            f"          1. Copy yolov8n.pt into the weights/ directory, OR\n"
            f"          2. Pass the full path:  --model /path/to/yolov8n.pt\n"
            f"        Download yolov8n.pt on a connected machine and transfer it.",
            file=sys.stderr,
        )
        sys.exit(1)

    # Explicit path — must exist
    resolved = candidate if candidate.is_absolute() else PROJECT_ROOT / candidate
    if not resolved.exists():
        print(
            f"[Train] ERROR: Model file not found: {resolved}",
            file=sys.stderr,
        )
        sys.exit(1)

    return resolved


def check_dataset(data_yaml: str) -> Path:
    """
    Verify the dataset YAML exists and that at least one image is present in
    each required split directory.  Exits with a descriptive error otherwise.
    """
    import yaml  # PyYAML — already in requirements.txt

    yaml_path = Path(data_yaml)
    if not yaml_path.is_absolute():
        yaml_path = PROJECT_ROOT / yaml_path

    if not yaml_path.exists():
        print(
            f"[Train] ERROR: Dataset YAML not found: {yaml_path}\n"
            f"        Run training/prepare_dataset.py first.",
            file=sys.stderr,
        )
        sys.exit(1)

    with open(yaml_path) as f:
        cfg = yaml.safe_load(f)

    # Resolve dataset root — path key is relative to the yaml file's directory
    dataset_root = yaml_path.parent / cfg.get("path", ".")
    dataset_root = dataset_root.resolve()

    img_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    errors = []

    for split_key in ("train", "val"):
        split_rel = cfg.get(split_key, "")
        split_dir = (dataset_root / split_rel).resolve()
        if not split_dir.exists():
            errors.append(f"  • {split_key} image dir missing: {split_dir}")
            continue
        images = [p for p in split_dir.iterdir() if p.suffix.lower() in img_exts]
        if not images:
            errors.append(f"  • {split_key} image dir is empty: {split_dir}")

    if errors:
        print(
            "[Train] ERROR: Dataset is not ready.\n"
            + "\n".join(errors) + "\n"
            "        Steps to fix:\n"
            "          1. Label images with LabelImg (see README § Dataset Preparation)\n"
            "          2. Run: python training/prepare_dataset.py --src raw_data --dst data",
            file=sys.stderr,
        )
        sys.exit(1)

    return yaml_path


# ── CLI ───────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(
        description="YOLOv8 PCB solder defect — offline training",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--model",
        default="yolov8n.pt",
        help=(
            "Base model weights.  Bare filenames (e.g. yolov8n.pt) are looked "
            "up in weights/.  Pass a full path to use weights from elsewhere."
        ),
    )
    p.add_argument(
        "--data",
        default="training/data.yaml",
        help="Path to dataset YAML (relative to project root or absolute).",
    )
    p.add_argument("--epochs",   type=int,   default=50)
    p.add_argument("--imgsz",    type=int,   default=640)
    p.add_argument("--batch",    type=int,   default=16)
    p.add_argument(
        "--device",
        default="cpu",
        help="Training device: 'cpu', '0' (first GPU), '0,1' (multi-GPU), 'mps'.",
    )
    p.add_argument("--name",     default="pcb_solder_v1", help="Run name.")
    p.add_argument("--patience", type=int,   default=20,  help="Early-stop patience.")
    return p.parse_args()


# ── Train ─────────────────────────────────────────────────────────────────────
def train(args, model_path: Path, data_yaml: Path):
    from ultralytics import YOLO

    print(f"[Train] Base model  : {model_path}")
    print(f"[Train] Dataset YAML: {data_yaml}")
    print(f"[Train] epochs={args.epochs}  imgsz={args.imgsz}  "
          f"batch={args.batch}  device={args.device}")

    model = YOLO(str(model_path))

    results = model.train(
        data     = str(data_yaml),
        epochs   = args.epochs,
        imgsz    = args.imgsz,
        batch    = args.batch,
        device   = args.device,
        name     = args.name,
        patience = args.patience,
        # ── Augmentation ──────────────────────────────────────────────────────
        hsv_h=0.015, hsv_s=0.7, hsv_v=0.4,
        flipud=0.1,  fliplr=0.5,
        mosaic=1.0,  mixup=0.1,
        # ── Optimiser ─────────────────────────────────────────────────────────
        optimizer="AdamW",
        lr0=0.001, lrf=0.01,
        warmup_epochs=3,
        # ── Logging ───────────────────────────────────────────────────────────
        plots=True, save=True, save_period=10,
    )
    return results


# ── Copy best weights → weights/ ─────────────────────────────────────────────
def copy_weights(run_name: str):
    runs_dir = PROJECT_ROOT / "runs" / "detect" / run_name / "weights"
    dest_dir = WEIGHTS_DIR
    dest_dir.mkdir(exist_ok=True)

    for fname in ("best.pt", "last.pt"):
        src = runs_dir / fname
        if src.exists():
            dst = dest_dir / fname
            shutil.copy2(src, dst)
            print(f"[Train] Copied {src} → {dst}")
        else:
            print(f"[Train] Warning: {src} not found — skipping.", file=sys.stderr)


# ── Validate ──────────────────────────────────────────────────────────────────
def validate(weights_path: Path, data_yaml: Path, imgsz: int, device: str):
    from ultralytics import YOLO

    model   = YOLO(str(weights_path))
    metrics = model.val(data=str(data_yaml), imgsz=imgsz, device=device)
    print(f"\n[Validate] mAP50:    {metrics.box.map50:.4f}")
    print(f"[Validate] mAP50-95: {metrics.box.map:.4f}")
    return metrics


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    args = parse_args()

    # Pre-flight: fail early with a clear message if anything is missing.
    model_path = check_model_weights(args.model)
    data_yaml  = check_dataset(args.data)

    train(args, model_path, data_yaml)
    copy_weights(args.name)

    best = WEIGHTS_DIR / "best.pt"
    if best.exists():
        validate(best, data_yaml, args.imgsz, args.device)

    print(f"\n[Train] Done.  Best weights: {best}")

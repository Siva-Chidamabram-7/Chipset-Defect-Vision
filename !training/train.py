"""
train.py — Fine-tune YOLOv8 on the PCB solder defect dataset
─────────────────────────────────────────────────────────────
OFFLINE-FIRST: this script never downloads anything from the internet.
  • The base model (yolov8n.pt) must already exist in weights/.
  • The dataset must already be prepared under data/.

Cloud / Vertex AI:
  • Pass --data gs://bucket/data/ to load the dataset from GCS.
  • Pass --model gs://bucket/weights/yolov8n.pt to load base weights from GCS.
  • Pass --output-model gs://bucket/models/ to upload best.pt after training.
  • Vertex AI standard env vars are read automatically:
      AIP_TRAINING_DATA_URI  → overrides --data
      AIP_MODEL_DIR          → overrides --output-model

Usage (run from project root):
    # Local
    python !training/train.py
    python !training/train.py --epochs 30 --imgsz 640 --device cpu
    python !training/train.py --epochs 50 --batch 8  --device 0   # GPU

    # GCS / Vertex AI
    python !training/train.py \
        --model        gs://my-bucket/weights/yolov8n.pt \
        --data         gs://my-bucket/data/ \
        --output-model gs://my-bucket/models/ \
        --epochs 50 --device 0

Outputs:
    weights/best.pt   ← best checkpoint (auto-copied after training)
    weights/last.pt   ← last checkpoint
    runs/detect/<name>/  ← full ultralytics run directory (plots, metrics, etc.)
    <output-model>/best.pt  ← also uploaded to GCS if --output-model is a gs:// path
"""

import argparse
import logging
import os
import shutil
import sys
import tempfile
from pathlib import Path

# ── Project root (two levels up from !training/train.py) ─────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
WEIGHTS_DIR  = PROJECT_ROOT / "weights"

# ── Logging ───────────────────────────────────────────────────────────────────
log = logging.getLogger("train")


# ── Pre-flight checks ─────────────────────────────────────────────────────────
def check_model_weights(model_arg: str) -> Path:
    """
    Resolve --model to an absolute Path that exists on disk.

    Rules:
      • If it is a gs:// URI → download to a temp file; return local path.
      • If the argument is already an absolute or relative path → use as-is.
      • If it is a bare filename (e.g. "yolov8n.pt") → look in weights/.
      • If the resolved file does not exist → print a helpful error and exit.

    This prevents ultralytics from silently downloading the file when it is
    passed a bare name that doesn't resolve to an existing file.
    """
    from scripts.gcs_utils import is_gcs_path, download_gcs_file

    if is_gcs_path(model_arg):
        local = WEIGHTS_DIR / Path(model_arg).name
        WEIGHTS_DIR.mkdir(exist_ok=True)
        log.info("[Train] Downloading model weights from GCS: %s", model_arg)
        download_gcs_file(model_arg, local)
        return local

    candidate = Path(model_arg)

    # Bare name with no directory component → search weights/ first
    if candidate.parent == Path(".") and not candidate.is_absolute():
        local = WEIGHTS_DIR / candidate
        if local.exists():
            return local
        # Don't fall through to auto-download; fail loudly.
        log.error(
            "[Train] Model weights not found.\n"
            "        Looked for:  %s\n"
            "        To fix:\n"
            "          1. Copy yolov8n.pt into the weights/ directory, OR\n"
            "          2. Pass the full path:  --model /path/to/yolov8n.pt\n"
            "        Download yolov8n.pt on a connected machine and transfer it.",
            local,
        )
        sys.exit(1)

    # Explicit path — must exist
    resolved = candidate if candidate.is_absolute() else PROJECT_ROOT / candidate
    if not resolved.exists():
        log.error("[Train] Model file not found: %s", resolved)
        sys.exit(1)

    return resolved


def check_dataset(data_yaml: str) -> Path:
    """
    Verify the dataset YAML exists and that at least one image is present in
    each required split directory.  Exits with a descriptive error otherwise.
    """
    import yaml  # PyYAML — already in requirements-training.txt

    yaml_path = Path(data_yaml)
    if not yaml_path.is_absolute():
        yaml_path = PROJECT_ROOT / yaml_path

    if not yaml_path.exists():
        log.error(
            "[Train] Dataset YAML not found: %s\n"
            "        Export your dataset from Roboflow in YOLOv8 format,\n"
            "        place images/labels under data/, then re-run.",
            yaml_path,
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
        log.error(
            "[Train] Dataset is not ready.\n%s\n"
            "        Steps to fix:\n"
            "          1. Export your dataset from Roboflow in YOLOv8 format.\n"
            "          2. Place the split directories under data/images/ and data/labels/.",
            "\n".join(errors),
        )
        sys.exit(1)

    return yaml_path


def build_data_yaml_for_gcs(local_data_dir: Path) -> Path:
    """
    Write a temporary data.yaml pointing to the locally-downloaded GCS dataset.
    Returns the path to the generated YAML file.
    """
    import yaml

    yaml_path = local_data_dir / "data.yaml"
    cfg = {
        "path":  str(local_data_dir),
        "train": "images/train",
        "val":   "images/val",
        "nc":    6,
        "names": {
            0: "Missing_hole",
            1: "Mouse_bite",
            2: "Open_circuit",
            3: "Short",
            4: "Spur",
            5: "Spurious_copper",
        },
    }
    with open(yaml_path, "w") as f:
        yaml.dump(cfg, f, default_flow_style=False)
    log.info("[Train] Generated data.yaml at %s", yaml_path)
    return yaml_path


# ── CLI ───────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(
        description="YOLOv8 PCB solder defect — offline / cloud training",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--model",
        default=None,
        help=(
            "Base model weights.  Bare filenames (e.g. yolov8n.pt) are looked "
            "up in weights/.  Pass a full path or gs:// URI."
        ),
    )
    p.add_argument(
        "--data",
        default=None,
        help=(
            "Path to dataset YAML (local) OR dataset root directory on GCS "
            "(gs://bucket/data/).  When a GCS path is given the dataset is "
            "downloaded and a data.yaml is auto-generated."
        ),
    )
    p.add_argument(
        "--output-model",
        default=None,
        help=(
            "Destination for the trained best.pt.  "
            "Accepts a local directory or gs://bucket/models/.  "
            "When set, best.pt is copied/uploaded here after training."
        ),
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
    p.add_argument(
        "--log-level", default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity.",
    )
    args = p.parse_args()

    # ── Environment-variable overrides (Vertex AI standard + custom) ──────────
    # Priority: CLI arg > env var > built-in default
    args.data         = args.data         or os.environ.get("AIP_TRAINING_DATA_URI", "!training/data.yaml")
    args.output_model = args.output_model or os.environ.get("AIP_MODEL_DIR",         "")
    args.model        = args.model        or os.environ.get("YOLO_BASE_MODEL",       "yolov8n.pt")

    return args


# ── Train ─────────────────────────────────────────────────────────────────────
def train(args, model_path: Path, data_yaml: Path):
    from ultralytics import YOLO

    log.info("[Train] Base model  : %s", model_path)
    log.info("[Train] Dataset YAML: %s", data_yaml)
    log.info(
        "[Train] epochs=%d  imgsz=%d  batch=%d  device=%s",
        args.epochs, args.imgsz, args.batch, args.device,
    )

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
            log.info("[Train] Copied %s → %s", src, dst)
        else:
            log.warning("[Train] %s not found — skipping.", src)


# ── Upload best.pt to output destination (local dir or GCS) ──────────────────
def export_model(output_dest: str):
    """
    Copy / upload weights/best.pt to the configured output destination.

    Accepts:
      • A local directory path  → copies best.pt into that directory.
      • A GCS URI (gs://…)      → uploads best.pt to that GCS prefix.
    """
    from scripts.gcs_utils import is_gcs_path, upload_gcs_file

    best = WEIGHTS_DIR / "best.pt"
    if not best.exists():
        log.warning("[Train] weights/best.pt not found — cannot export model.")
        return

    if is_gcs_path(output_dest):
        # Ensure the GCS path ends in a "directory-like" prefix
        dest_blob = output_dest.rstrip("/") + "/best.pt"
        log.info("[Train] Uploading best.pt → %s", dest_blob)
        upload_gcs_file(best, dest_blob)
        log.info("[Train] Model upload complete.")
    else:
        dest_dir = Path(output_dest)
        dest_dir.mkdir(parents=True, exist_ok=True)
        dst = dest_dir / "best.pt"
        shutil.copy2(best, dst)
        log.info("[Train] Copied best.pt → %s", dst)


# ── Validate ──────────────────────────────────────────────────────────────────
def validate(weights_path: Path, data_yaml: Path, imgsz: int, device: str):
    from ultralytics import YOLO

    log.info("[Validate] Running validation with %s ...", weights_path.name)
    model   = YOLO(str(weights_path))
    metrics = model.val(data=str(data_yaml), imgsz=imgsz, device=device)
    log.info("[Validate] mAP50:    %.4f", metrics.box.map50)
    log.info("[Validate] mAP50-95: %.4f", metrics.box.map)
    return metrics


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    args = parse_args()

    # ── Logging setup (Vertex AI + Docker compatible) ─────────────────────────
    sys.path.insert(0, str(PROJECT_ROOT))   # ensure scripts/ is importable
    from scripts.gcs_utils import setup_logging, is_gcs_path, download_gcs_dir
    setup_logging(level=getattr(logging, args.log_level.upper(), logging.INFO))

    log.info("=" * 60)
    log.info("[Train] YOLOv8 PCB Defect Training — starting")
    log.info("[Train] Model         : %s", args.model)
    log.info("[Train] Data          : %s", args.data)
    log.info("[Train] Output model  : %s", args.output_model or "(local only)")
    log.info("[Train] Epochs        : %d", args.epochs)
    log.info("[Train] Device        : %s", args.device)
    log.info("=" * 60)

    # ── Handle GCS dataset ────────────────────────────────────────────────────
    _tmpdir      = None
    local_data_yaml = args.data

    if is_gcs_path(args.data):
        log.info("[Train] Downloading dataset from GCS: %s", args.data)
        _tmpdir     = tempfile.mkdtemp(prefix="pcb_train_")
        local_data  = Path(_tmpdir) / "data"
        local_data.mkdir()
        download_gcs_dir(args.data, local_data)
        local_data_yaml = str(build_data_yaml_for_gcs(local_data))
        log.info("[Train] Dataset downloaded to: %s", local_data)

    try:
        # Pre-flight: fail early with a clear message if anything is missing.
        model_path = check_model_weights(args.model)
        data_yaml  = check_dataset(local_data_yaml)

        log.info("[Train] Pre-flight checks passed. Starting training ...")
        train(args, model_path, data_yaml)
        copy_weights(args.name)

        best = WEIGHTS_DIR / "best.pt"
        if best.exists():
            validate(best, data_yaml, args.imgsz, args.device)

        # ── Export model to output destination ────────────────────────────────
        if args.output_model:
            export_model(args.output_model)

        log.info("[Train] Done.  Best weights: %s", best)

    finally:
        if _tmpdir:
            shutil.rmtree(_tmpdir, ignore_errors=True)

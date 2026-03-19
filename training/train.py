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

Usage (run from any directory — paths are always resolved from project root):

    # Local training
    python training/train.py
    python training/train.py --epochs 30 --imgsz 640 --device cpu
    python training/train.py --epochs 50 --batch 8  --device 0   # GPU

    # GCS / Vertex AI
    python training/train.py \\
        --model        gs://my-bucket/weights/yolov8n.pt \\
        --data         gs://my-bucket/data/ \\
        --output-model gs://my-bucket/models/ \\
        --epochs 50 --device 0

Outputs:
    weights/best.pt          ← best checkpoint (auto-copied after training)
    weights/last.pt          ← last checkpoint
    runs/detect/<name>/      ← full ultralytics run directory (plots, metrics)
    <output-model>/best.pt   ← uploaded to GCS if --output-model is gs://…
"""

import argparse
import logging
import os
import shutil
import sys
import tempfile
from pathlib import Path

# ── Project root — one level up from training/train.py ───────────────────────
# Resolved at import time so all downstream path operations are absolute
# regardless of the current working directory when the script is invoked.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
WEIGHTS_DIR  = PROJECT_ROOT / "weights"

# ── sys.path — make training/scripts/ importable as "scripts.*" ──────────────
# PROJECT_ROOT        → enables "training.scripts.*" if ever needed
# PROJECT_ROOT/training → enables "scripts.gcs_utils" etc. (current style)
# Both insertions are idempotent (guarded by the `not in` check).
for _p in (str(PROJECT_ROOT), str(PROJECT_ROOT / "training")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# ── Logging ───────────────────────────────────────────────────────────────────
log = logging.getLogger("train")


# ─────────────────────────────────────────────────────────────────────────────
# Pre-flight checks
# ─────────────────────────────────────────────────────────────────────────────

def check_model_weights(model_arg: str) -> Path:
    """
    Resolve --model to an absolute Path that exists on disk.

    Rules:
      • GCS URI (gs://…)          → download to weights/<filename>; return local path.
      • Bare filename (yolov8n.pt) → look in weights/ only.  No auto-download.
      • Explicit relative/absolute path → resolve relative to PROJECT_ROOT.

    Raises RuntimeError if the file cannot be found — never silently substitutes
    a different model.
    """
    from scripts.gcs_utils import is_gcs_path, download_gcs_file

    if is_gcs_path(model_arg):
        local = WEIGHTS_DIR / Path(model_arg).name
        WEIGHTS_DIR.mkdir(exist_ok=True)
        log.info("[Train] Downloading model weights from GCS: %s", model_arg)
        download_gcs_file(model_arg, local)
        return local

    candidate = Path(model_arg)

    # Bare name with no directory component → weights/ only, no fallback
    if candidate.parent == Path(".") and not candidate.is_absolute():
        local = WEIGHTS_DIR / candidate
        if local.exists():
            return local
        raise RuntimeError(
            f"[Train] FATAL: base model weights not found.\n"
            f"        Expected: {local}\n"
            f"\n"
            f"        To fix:\n"
            f"          • Download yolov8n.pt on a machine with internet access\n"
            f"            and copy it into weights/:\n"
            f"              cp /path/to/yolov8n.pt {WEIGHTS_DIR}/yolov8n.pt\n"
            f"          • Or pass a full path:  --model /absolute/path/to/yolov8n.pt\n"
        )

    # Explicit path — must exist as given (relative paths anchored to PROJECT_ROOT)
    resolved = candidate if candidate.is_absolute() else PROJECT_ROOT / candidate
    if not resolved.exists():
        raise RuntimeError(
            f"[Train] FATAL: model file not found: {resolved}"
        )
    return resolved


def check_dataset(data_yaml: str) -> Path:
    """
    Verify the dataset YAML exists and that both split directories contain
    images and label files.

    Validates:
      • data.yaml file exists
      • data/images/train/  exists and is non-empty
      • data/images/val/    exists and is non-empty
      • data/labels/train/  exists and is non-empty
      • data/labels/val/    exists and is non-empty

    Raises RuntimeError with the full list of problems if anything is missing.
    Never calls sys.exit — raises so the caller can handle cleanup.
    """
    import yaml

    yaml_path = Path(data_yaml)
    if not yaml_path.is_absolute():
        yaml_path = PROJECT_ROOT / yaml_path

    if not yaml_path.exists():
        raise RuntimeError(
            f"[Train] FATAL: dataset YAML not found: {yaml_path}\n"
            f"\n"
            f"        To fix:\n"
            f"          • Ensure training/data.yaml exists at the project root.\n"
            f"          • Prepare the dataset under data/ using the SAM pipeline:\n"
            f"              python training/scripts/sam_to_yolo.py\n"
        )

    with open(yaml_path) as f:
        cfg = yaml.safe_load(f)

    # dataset_root is resolved relative to the YAML file's own directory.
    # For training/data.yaml with path: ../data → PROJECT_ROOT/data
    dataset_root = (yaml_path.parent / cfg.get("path", ".")).resolve()

    img_exts   = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    label_exts = {".txt"}
    errors     = []

    for split_key in ("train", "val"):
        # ── Image directory ────────────────────────────────────────────────────
        img_rel  = cfg.get(split_key, "")           # e.g. "images/train"
        img_dir  = (dataset_root / img_rel).resolve()

        if not img_dir.exists():
            errors.append(f"  • {split_key} image dir missing: {img_dir}")
        else:
            images = [p for p in img_dir.iterdir() if p.suffix.lower() in img_exts]
            if not images:
                errors.append(f"  • {split_key} image dir is empty: {img_dir}")

        # ── Label directory ────────────────────────────────────────────────────
        # Mirrors image dir: data/images/train → data/labels/train
        label_dir = (dataset_root / "labels" / split_key).resolve()

        if not label_dir.exists():
            errors.append(f"  • {split_key} label dir missing: {label_dir}")
        else:
            labels = [p for p in label_dir.iterdir() if p.suffix.lower() in label_exts]
            if not labels:
                errors.append(f"  • {split_key} label dir is empty: {label_dir}")

    if errors:
        raise RuntimeError(
            "[Train] FATAL: dataset is not ready.\n"
            + "\n".join(errors)
            + "\n\n"
            "        To fix:\n"
            "          1. Run the SAM annotation pipeline to generate labels:\n"
            "               python training/scripts/sam_to_yolo.py\n"
            "          2. Split images and labels into train/val under data/.\n"
            "          3. Re-run training.\n"
        )

    return yaml_path


def build_data_yaml_for_gcs(local_data_dir: Path) -> Path:
    """
    Write a temporary data.yaml pointing to the locally-downloaded GCS dataset.
    Returns the path to the generated YAML file.

    Class schema must exactly mirror training/data.yaml (nc=7, 7 named classes).
    """
    import yaml

    yaml_path = local_data_dir / "data.yaml"
    cfg = {
        "path":  str(local_data_dir),
        "train": "images/train",
        "val":   "images/val",
        "nc":    7,
        "names": {
            0: "Missing_hole",
            1: "Mouse_bite",
            2: "Open_circuit",
            3: "Short",
            4: "Spur",
            5: "Spurious_copper",
            6: "Good",
        },
    }
    with open(yaml_path, "w") as f:
        yaml.dump(cfg, f, default_flow_style=False)
    log.info("[Train] Generated data.yaml at %s", yaml_path)
    return yaml_path


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

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
            "up in weights/ only — no auto-download.  Pass a full path or gs:// URI."
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
    # Priority: CLI arg > env var > hard-coded default (absolute path)
    # Default data path is always absolute so training works from any directory.
    _default_data = str(PROJECT_ROOT / "training" / "data.yaml")
    args.data         = args.data         or os.environ.get("AIP_TRAINING_DATA_URI", _default_data)
    args.output_model = args.output_model or os.environ.get("AIP_MODEL_DIR",         "")
    args.model        = args.model        or os.environ.get("YOLO_BASE_MODEL",       "yolov8n.pt")

    return args


# ─────────────────────────────────────────────────────────────────────────────
# Training
# ─────────────────────────────────────────────────────────────────────────────

def train(args, model_path: Path, data_yaml: Path):
    from ultralytics import YOLO

    log.info("[Train] Base model  : %s", model_path)
    log.info("[Train] Dataset YAML: %s", data_yaml)
    log.info(
        "[Train] epochs=%d  imgsz=%d  batch=%d  device=%s",
        args.epochs, args.imgsz, args.batch, args.device,
    )

    model = YOLO(str(model_path))

    model.train(
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
        # ── Artefacts ─────────────────────────────────────────────────────────
        plots=True, save=True, save_period=10,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Post-training steps
# ─────────────────────────────────────────────────────────────────────────────

def copy_weights(run_name: str) -> None:
    """Copy best.pt and last.pt from the ultralytics run directory → weights/."""
    runs_dir = PROJECT_ROOT / "runs" / "detect" / run_name / "weights"
    WEIGHTS_DIR.mkdir(exist_ok=True)

    for fname in ("best.pt", "last.pt"):
        src = runs_dir / fname
        if src.exists():
            dst = WEIGHTS_DIR / fname
            shutil.copy2(src, dst)
            log.info("[Train] Copied %s → %s", src, dst)
        else:
            log.warning("[Train] %s not found — skipping copy.", src)


def export_model(output_dest: str) -> None:
    """
    Copy / upload weights/best.pt to the configured output destination.

    Accepts:
      • A local directory path → copies best.pt into that directory.
      • A GCS URI (gs://…)    → uploads best.pt to that GCS prefix.
    """
    from scripts.gcs_utils import is_gcs_path, upload_gcs_file

    best = WEIGHTS_DIR / "best.pt"
    if not best.exists():
        raise RuntimeError(
            f"[Train] FATAL: weights/best.pt not found after training.\n"
            f"        Expected: {best}\n"
            f"        Training may have failed — check run logs."
        )

    if is_gcs_path(output_dest):
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


def validate(weights_path: Path, data_yaml: Path, imgsz: int, device: str) -> None:
    from ultralytics import YOLO

    log.info("[Validate] Running validation with %s ...", weights_path.name)
    model   = YOLO(str(weights_path))
    metrics = model.val(data=str(data_yaml), imgsz=imgsz, device=device)
    log.info("[Validate] mAP50:    %.4f", metrics.box.map50)
    log.info("[Validate] mAP50-95: %.4f", metrics.box.map)


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    args = parse_args()

    # Setup logging before any other output
    from scripts.gcs_utils import setup_logging, is_gcs_path, download_gcs_dir
    setup_logging(level=getattr(logging, args.log_level.upper(), logging.INFO))

    log.info("=" * 60)
    log.info("[Train] YOLOv8 PCB Defect Training — starting")
    log.info("[Train] Project root  : %s", PROJECT_ROOT)
    log.info("[Train] Model         : %s", args.model)
    log.info("[Train] Data          : %s", args.data)
    log.info("[Train] Output model  : %s", args.output_model or "(local only)")
    log.info("[Train] Epochs        : %d", args.epochs)
    log.info("[Train] Device        : %s", args.device)
    log.info("=" * 60)

    # ── Handle GCS dataset ────────────────────────────────────────────────────
    _tmpdir         = None
    local_data_yaml = args.data

    if is_gcs_path(args.data):
        log.info("[Train] Downloading dataset from GCS: %s", args.data)
        _tmpdir    = tempfile.mkdtemp(prefix="pcb_train_")
        local_data = Path(_tmpdir) / "data"
        local_data.mkdir()
        download_gcs_dir(args.data, local_data)
        local_data_yaml = str(build_data_yaml_for_gcs(local_data))
        log.info("[Train] Dataset downloaded to: %s", local_data)

    try:
        # Pre-flight: fail early and loudly if anything is missing.
        model_path = check_model_weights(args.model)
        data_yaml  = check_dataset(local_data_yaml)

        log.info("[Train] Pre-flight checks passed — starting training ...")
        train(args, model_path, data_yaml)
        copy_weights(args.name)

        best = WEIGHTS_DIR / "best.pt"
        if not best.exists():
            raise RuntimeError(
                f"[Train] FATAL: training completed but best.pt was not produced.\n"
                f"        Expected: {best}\n"
                f"        Check ultralytics run logs for training errors."
            )

        validate(best, data_yaml, args.imgsz, args.device)

        if args.output_model:
            export_model(args.output_model)

        log.info("[Train] Done.  Best weights: %s", best)

    finally:
        if _tmpdir:
            shutil.rmtree(_tmpdir, ignore_errors=True)

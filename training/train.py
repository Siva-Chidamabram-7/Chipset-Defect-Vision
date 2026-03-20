from __future__ import annotations

import argparse
import logging
import os
import shutil
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import yaml

from training.config import (
    CLASS_NAMES,
    DEFAULT_BASE_MODEL,
    DEFAULT_HYPERPARAMETERS,
    DEFAULT_LOCAL_DATA_CONFIG,
    DEFAULT_RUN_NAME,
    PROJECT_ROOT,
    RUNS_DIR,
    WEIGHTS_DIR,
)
from training.scripts.gcs_utils import (
    download_gcs_dir,
    download_gcs_file,
    is_gcs_path,
    setup_logging,
    upload_gcs_file,
)

log = logging.getLogger("training.train")

DATASET_SPLITS = ("train", "val")
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


@dataclass(frozen=True)
class TrainConfig:
    model: str
    data: str
    output_model: str
    epochs: int
    batch_size: int
    learning_rate: float
    image_size: int
    optimizer: str
    device: str
    run_name: str
    patience: int


def _env_or_default(name: str, default: str) -> str:
    return os.environ.get(name, default)


def parse_args() -> TrainConfig:
    parser = argparse.ArgumentParser(
        description="Train YOLOv8 locally or on Vertex AI with local or GCS datasets.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model",
        default=_env_or_default("YOLO_BASE_MODEL", DEFAULT_BASE_MODEL),
        help="Base model checkpoint path. Supports local files or gs:// URIs.",
    )
    parser.add_argument(
        "--data",
        default=_env_or_default(
            "TRAIN_DATA_PATH",
            _env_or_default("AIP_TRAINING_DATA_URI", str(DEFAULT_LOCAL_DATA_CONFIG)),
        ),
        help="Dataset YAML path, dataset directory, or gs:// dataset prefix.",
    )
    parser.add_argument(
        "--output-model",
        default=_env_or_default("TRAIN_OUTPUT_MODEL", _env_or_default("AIP_MODEL_DIR", "")),
        help="Where to copy the trained best.pt. Supports local directories or gs:// prefixes.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=int(_env_or_default("TRAIN_EPOCHS", str(DEFAULT_HYPERPARAMETERS["epochs"]))),
    )
    parser.add_argument(
        "--batch",
        dest="batch_size",
        type=int,
        default=int(_env_or_default("TRAIN_BATCH_SIZE", str(DEFAULT_HYPERPARAMETERS["batch_size"]))),
    )
    parser.add_argument(
        "--lr",
        dest="learning_rate",
        type=float,
        default=float(
            _env_or_default("TRAIN_LEARNING_RATE", str(DEFAULT_HYPERPARAMETERS["learning_rate"]))
        ),
        help="Initial learning rate (YOLO lr0).",
    )
    parser.add_argument(
        "--imgsz",
        "--image-size",
        dest="image_size",
        type=int,
        default=int(_env_or_default("TRAIN_IMAGE_SIZE", str(DEFAULT_HYPERPARAMETERS["image_size"]))),
        help="Training image size.",
    )
    parser.add_argument(
        "--optimizer",
        default=_env_or_default("TRAIN_OPTIMIZER", str(DEFAULT_HYPERPARAMETERS["optimizer"])),
        help="YOLO optimizer name such as AdamW, SGD, or Adam.",
    )
    parser.add_argument(
        "--device",
        default=_env_or_default("TRAIN_DEVICE", str(DEFAULT_HYPERPARAMETERS["device"])),
        help="Training device such as cpu, cuda, cuda:0, or 0.",
    )
    parser.add_argument(
        "--name",
        dest="run_name",
        default=_env_or_default("TRAIN_RUN_NAME", DEFAULT_RUN_NAME),
        help="Training run name under runs/detect/.",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=int(_env_or_default("TRAIN_PATIENCE", str(DEFAULT_HYPERPARAMETERS["patience"]))),
        help="Early stopping patience.",
    )
    parser.add_argument(
        "--log-level",
        default=_env_or_default("TRAIN_LOG_LEVEL", "INFO"),
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity.",
    )

    args = parser.parse_args()
    setup_logging(level=getattr(logging, args.log_level.upper(), logging.INFO))
    return TrainConfig(
        model=args.model,
        data=args.data,
        output_model=args.output_model,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        image_size=args.image_size,
        optimizer=args.optimizer,
        device=args.device,
        run_name=args.run_name,
        patience=args.patience,
    )


def resolve_path(path_value: str) -> Path:
    candidate = Path(path_value)
    return candidate if candidate.is_absolute() else PROJECT_ROOT / candidate


def generate_data_yaml(dataset_root: Path, output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(
            {
                "path": str(dataset_root),
                "train": "images/train",
                "val": "images/val",
                "nc": len(CLASS_NAMES),
                "names": CLASS_NAMES,
            },
            handle,
            sort_keys=False,
        )
    return output_path


def validate_dataset_yaml(data_yaml_path: Path) -> Path:
    if not data_yaml_path.exists():
        raise RuntimeError(f"[Train] Dataset config not found: {data_yaml_path}")

    with data_yaml_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}

    dataset_root = Path(config.get("path", "."))
    if not dataset_root.is_absolute():
        dataset_root = (data_yaml_path.parent / dataset_root).resolve()

    errors: list[str] = []
    for split in DATASET_SPLITS:
        image_dir = dataset_root / "images" / split
        label_dir = dataset_root / "labels" / split

        if not image_dir.exists():
            errors.append(f"Missing image directory: {image_dir}")
        else:
            image_count = sum(
                1 for file_path in image_dir.iterdir() if file_path.suffix.lower() in IMAGE_EXTENSIONS
            )
            if image_count == 0:
                errors.append(f"No images found in: {image_dir}")

        if not label_dir.exists():
            errors.append(f"Missing label directory: {label_dir}")

    if errors:
        raise RuntimeError("[Train] Dataset validation failed:\n" + "\n".join(errors))

    return data_yaml_path


def stage_dataset(data_source: str, workspace: Path) -> Path:
    if is_gcs_path(data_source):
        dataset_root = workspace / "dataset"
        dataset_root.mkdir(parents=True, exist_ok=True)
        file_count = download_gcs_dir(data_source, dataset_root)
        if file_count == 0:
            raise RuntimeError(f"[Train] No dataset files found at {data_source}")

        explicit_yaml = dataset_root / "data.yaml"
        if explicit_yaml.exists():
            return validate_dataset_yaml(explicit_yaml)

        generated_yaml = workspace / "generated-data.yaml"
        return validate_dataset_yaml(generate_data_yaml(dataset_root, generated_yaml))

    local_path = resolve_path(data_source)
    if local_path.is_file():
        return validate_dataset_yaml(local_path)

    if local_path.is_dir():
        explicit_yaml = local_path / "data.yaml"
        if explicit_yaml.exists():
            return validate_dataset_yaml(explicit_yaml)

        generated_yaml = workspace / "generated-data.yaml"
        return validate_dataset_yaml(generate_data_yaml(local_path, generated_yaml))

    raise RuntimeError(f"[Train] Unsupported dataset source: {data_source}")


def stage_model(model_source: str, workspace: Path) -> Path:
    if is_gcs_path(model_source):
        local_model_path = workspace / Path(model_source).name
        download_gcs_file(model_source, local_model_path)
        return local_model_path

    local_model_path = resolve_path(model_source)
    if not local_model_path.exists():
        # Allow bare Ultralytics model names like yolov8n.pt so the trainer
        # can resolve or download them at runtime when desired.
        candidate = Path(model_source)
        if candidate.parent == Path(".") and candidate.suffix == ".pt":
            return candidate
        raise RuntimeError(
            "[Train] Base model checkpoint not found.\n"
            f"Expected: {local_model_path}\n"
            "Provide a valid local file, a gs:// checkpoint URI, or a model name like yolov8n.pt."
        )
    return local_model_path


def train_model(config: TrainConfig, model_path: Path, data_yaml_path: Path) -> Path:
    from ultralytics import YOLO

    log.info("Starting training")
    log.info("Model checkpoint : %s", model_path)
    log.info("Dataset config   : %s", data_yaml_path)
    log.info(
        "Hyperparameters  : epochs=%d batch=%d lr=%s imgsz=%d optimizer=%s device=%s",
        config.epochs,
        config.batch_size,
        config.learning_rate,
        config.image_size,
        config.optimizer,
        config.device,
    )

    model = YOLO(str(model_path))
    model.train(
        data=str(data_yaml_path),
        epochs=config.epochs,
        batch=config.batch_size,
        imgsz=config.image_size,
        lr0=config.learning_rate,
        optimizer=config.optimizer,
        device=config.device,
        name=config.run_name,
        patience=config.patience,
        project=str(RUNS_DIR / "detect"),
        save=True,
        plots=True,
    )

    trainer = getattr(model, "trainer", None)
    save_dir = getattr(trainer, "save_dir", None)
    if save_dir is None:
        raise RuntimeError("[Train] Unable to determine YOLO run directory after training.")

    run_dir = Path(save_dir)
    if not run_dir.exists():
        raise RuntimeError(f"[Train] YOLO run directory not found: {run_dir}")
    return run_dir


def copy_training_outputs(run_dir: Path) -> Path:
    weights_dir = run_dir / "weights"
    best_checkpoint = weights_dir / "best.pt"
    last_checkpoint = weights_dir / "last.pt"

    if not best_checkpoint.exists():
        raise RuntimeError(f"[Train] Training finished without best.pt in {weights_dir}")

    WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
    shutil.copy2(best_checkpoint, WEIGHTS_DIR / "best.pt")

    if last_checkpoint.exists():
        shutil.copy2(last_checkpoint, WEIGHTS_DIR / "last.pt")

    return WEIGHTS_DIR / "best.pt"


def validate_model(best_checkpoint: Path, data_yaml_path: Path, image_size: int, device: str) -> None:
    from ultralytics import YOLO

    model = YOLO(str(best_checkpoint))
    metrics = model.val(data=str(data_yaml_path), imgsz=image_size, device=device)
    log.info("Validation mAP50    : %.4f", metrics.box.map50)
    log.info("Validation mAP50-95 : %.4f", metrics.box.map)


def export_model(best_checkpoint: Path, output_model: str) -> None:
    if not output_model:
        return

    if is_gcs_path(output_model):
        destination = output_model.rstrip("/") + "/best.pt"
        upload_gcs_file(best_checkpoint, destination)
        log.info("Uploaded best checkpoint to %s", destination)
        return

    output_dir = resolve_path(output_model)
    output_dir.mkdir(parents=True, exist_ok=True)
    destination = output_dir / "best.pt"
    shutil.copy2(best_checkpoint, destination)
    log.info("Copied best checkpoint to %s", destination)


def main() -> int:
    config = parse_args()
    log.info("Project root       : %s", PROJECT_ROOT)
    log.info("Training data      : %s", config.data)
    log.info("Base model         : %s", config.model)
    log.info("Output model       : %s", config.output_model or "(not exported)")

    with tempfile.TemporaryDirectory(prefix="chipset-train-") as tmp_dir:
        workspace = Path(tmp_dir)
        data_yaml_path = stage_dataset(config.data, workspace)
        model_path = stage_model(config.model, workspace)
        run_dir = train_model(config, model_path, data_yaml_path)
        best_checkpoint = copy_training_outputs(run_dir)
        validate_model(best_checkpoint, data_yaml_path, config.image_size, config.device)
        export_model(best_checkpoint, config.output_model)
        log.info("Training complete. Best checkpoint: %s", best_checkpoint)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

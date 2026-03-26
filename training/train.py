from __future__ import annotations

import argparse
import logging
import os
import shutil
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import yaml

from training.config import (
    DEFAULT_LOCAL_DATA_CONFIG,
    DEFAULT_RUN_NAME,
    PROJECT_ROOT,
    RUNS_DIR,
    TRAINING_DIR,
    WEIGHTS_DIR,
)
from training.scripts.gcs_utils import (
    download_gcs_dir,
    download_gcs_file,
    is_gcs_path,
    setup_logging,
    upload_gcs_file,
)

# Suppress Ultralytics' per-batch progress spam; keep only ERROR-level output
# from the library itself.  Our own log.* calls are unaffected.
logging.getLogger("ultralytics").setLevel(logging.ERROR)

log = logging.getLogger("training.train")

HYPERPARAMETERS_YAML = TRAINING_DIR / "hyperparameters.yaml"
DATA_YAML = TRAINING_DIR / "data.yaml"

DATASET_SPLITS = ("train", "val")
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# Map shorthand model names to Ultralytics model filenames
_MODEL_MAP: dict[str, str] = {
    "yolo-n": "yolov8n.pt",
    "yolo-s": "yolov8s.pt",
    "yolo-m": "yolov8m.pt",
    "yolo-l": "yolov8l.pt",
    "yolo-x": "yolov8x.pt",
}


def load_hyperparameters() -> dict:
    """Load training hyperparameters from the single source of truth."""
    if not HYPERPARAMETERS_YAML.exists():
        raise RuntimeError(
            f"[Train] hyperparameters.yaml not found: {HYPERPARAMETERS_YAML}\n"
            "This file is required. Do not remove it."
        )
    with HYPERPARAMETERS_YAML.open() as f:
        params = yaml.safe_load(f)
    if not isinstance(params, dict):
        raise RuntimeError(f"[Train] hyperparameters.yaml must be a YAML mapping, got: {type(params)}")
    return params


def resolve_model_name(model: str) -> str:
    """Resolve shorthand names like 'yolo-m' to their Ultralytics filenames."""
    resolved = _MODEL_MAP.get(model, model)
    if resolved == model and not any(
        model.endswith(ext) for ext in (".pt", ".yaml")
    ) and not is_gcs_path(model):
        raise RuntimeError(
            f"[Train] Unknown model name: '{model}'\n"
            f"Use a shorthand ({', '.join(_MODEL_MAP)}) or a direct path/gs:// URI."
        )
    return resolved


def _get_class_names() -> list[str]:
    """Read class names from the authoritative training/data.yaml."""
    with DATA_YAML.open() as f:
        cfg = yaml.safe_load(f)
    names = cfg.get("names", [])
    if isinstance(names, dict):
        return [names[i] for i in sorted(names)]
    return list(names)


def _env_or_default(name: str, default: str) -> str:
    return os.environ.get(name, default)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train YOLO on PCB defect data — hyperparameters loaded from hyperparameters.yaml.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--data",
        default=_env_or_default("TRAIN_DATA_PATH", str(DEFAULT_LOCAL_DATA_CONFIG)),
        help="Dataset YAML path or directory. Defaults to training/data.yaml.",
    )
    parser.add_argument(
        "--output-model",
        default=_env_or_default("TRAIN_OUTPUT_MODEL", ""),
        help="Optional directory to copy best.pt into after training. "
             "Local path only. Leave empty to keep the run inside runs/detect/.",
    )
    parser.add_argument(
        "--device",
        default=None,  # resolved at runtime: cuda if available, else cpu
        help="Training device: cpu, cuda, cuda:0, or 0. "
             "Auto-detected if omitted (GPU when available, CPU otherwise).",
    )
    parser.add_argument(
        "--name",
        default=_env_or_default("TRAIN_RUN_NAME", DEFAULT_RUN_NAME),
        help="Run name under runs/detect/.",
    )
    parser.add_argument(
        "--log-level",
        default=_env_or_default("TRAIN_LOG_LEVEL", "INFO"),
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    return parser.parse_args()


def resolve_path(path_value: str) -> Path:
    candidate = Path(path_value)
    return candidate if candidate.is_absolute() else PROJECT_ROOT / candidate


def generate_data_yaml(dataset_root: Path, output_path: Path) -> Path:
    class_names = _get_class_names()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(
            {
                "path": str(dataset_root),
                "train": "images/train",
                "val": "images/val",
                "nc": len(class_names),
                "names": class_names,
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
        # Bare Ultralytics model names (e.g. yolov8m.pt) are auto-downloaded.
        candidate = Path(model_source)
        if candidate.parent == Path(".") and candidate.suffix == ".pt":
            return candidate
        raise RuntimeError(
            "[Train] Base model checkpoint not found.\n"
            f"Expected: {local_model_path}\n"
            "Provide a valid local file, a gs:// URI, or a model name like yolov8m.pt."
        )
    return local_model_path


def _install_image_tracker(model: object, sink: list[str]) -> None:
    """Monkey-patch YOLO's dataset so we record the last image path it tries to
    load.  This lets us report the culprit when an OpenCV error crashes training.
    The patch is best-effort: if internal YOLO APIs change it degrades silently."""
    try:
        trainer = getattr(model, "trainer", None)
        if trainer is None:
            return  # trainer doesn't exist before .train() is called — OK, skip

        dataset = getattr(trainer, "train_loader", None)
        dataset = getattr(dataset, "dataset", dataset) if dataset else None
        if dataset is None:
            return

        original_getitem = dataset.__class__.__getitem__

        def _tracked_getitem(self, index):  # type: ignore[no-untyped-def]
            im_file = getattr(self, "im_files", None)
            if im_file:
                try:
                    sink.clear()
                    sink.append(str(im_file[index]))
                except Exception:
                    pass
            return original_getitem(self, index)

        dataset.__class__.__getitem__ = _tracked_getitem
    except Exception:
        pass  # never let the tracker itself crash training


def train_model(
    model_path: Path,
    data_yaml_path: Path,
    hyperparams: dict,
    device: str,
    run_name: str,
) -> Path:
    from ultralytics import YOLO

    # Separate model key (used to initialise YOLO) from training kwargs
    train_kwargs = {k: v for k, v in hyperparams.items() if k != "model"}

    # YOLO uses -1 for auto batch; the yaml stores the human-readable "auto"
    if train_kwargs.get("batch") == "auto":
        train_kwargs["batch"] = -1

    # ── Device resolution — CPU-safe ─────────────────────────────────────────
    # Resolve the effective device before touching YOLO so we never assume CUDA.
    # If the caller requested a GPU device but CUDA is unavailable (e.g. Vertex
    # CPU-only node), we fall back to cpu with a clear warning instead of
    # crashing deep inside PyTorch.
    import torch

    requested_device: str = device or "cpu"
    if requested_device != "cpu":
        if not torch.cuda.is_available():
            log.warning(
                "[Train] Device '%s' requested but CUDA is not available on this machine. "
                "Falling back to cpu.  Pass --device cpu to silence this warning.",
                requested_device,
            )
            requested_device = "cpu"
        else:
            log.info("CUDA device count  : %d", torch.cuda.device_count())

    # Explicitly write resolved device into kwargs — single source of truth,
    # no implicit YOLO defaulting to GPU when one happens to be present.
    train_kwargs["device"] = requested_device
    # ─────────────────────────────────────────────────────────────────────────

    # ── Crash-debugging mode ─────────────────────────────────────────────────
    # workers=0 on CPU disables multiprocessing so the exact failing file
    # surfaces in the traceback instead of being swallowed by a DataLoader
    # worker process.  2 workers are used on GPU where forking is safe.
    # Remove this block (or set a fixed value) once the dataset is clean.
    train_kwargs["workers"] = 0 if requested_device == "cpu" else 2
    # ─────────────────────────────────────────────────────────────────────────

    log.info("Loading model      : %s", model_path)
    log.info("Dataset config     : %s", data_yaml_path)
    log.info("Device (resolved)  : %s", requested_device)
    log.info("Hyperparameters    : %s", train_kwargs)
    log.warning(
        "[Debug] workers=0 is active — multiprocessing is disabled to expose "
        "the exact file that triggers the OpenCV crash. Run data_sanity_check.py "
        "first, fix/remove bad files, then remove the workers=0 line."
    )

    model = YOLO(str(model_path))
    _last_image: list[str] = []  # mutable container so the except block can read it

    try:
        # Patch YOLO's dataset __getitem__ to track the last accessed path so we
        # can report it if training crashes mid-epoch.
        _install_image_tracker(model, _last_image)
        model.train(
            data=str(data_yaml_path),
            name=run_name,
            project=str(RUNS_DIR / "detect"),
            save=True,
            plots=False,
            val=False,
            verbose=False,
            **train_kwargs,  # device is already inside train_kwargs
        )
    except Exception as exc:
        last = _last_image[0] if _last_image else "unknown (tracker not reached)"
        log.error("=" * 72)
        log.error("[Train] CRASH during model.train()")
        log.error("  Exception type : %s", type(exc).__name__)
        log.error("  Message        : %s", exc)
        log.error("  Last image seen: %s", last)
        log.error("")
        log.error("Next steps:")
        log.error("  1. Run:  python -m training.data_sanity_check")
        log.error("  2. Remove or fix the reported broken image / invalid label.")
        log.error("  3. Re-run training.")
        log.error("=" * 72)
        raise

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
    import torch
    from ultralytics import YOLO

    # Mirror the same CPU-safe resolution used in train_model.
    val_device = device or "cpu"
    if val_device != "cpu" and not torch.cuda.is_available():
        log.warning("[Val] CUDA unavailable — running validation on cpu.")
        val_device = "cpu"

    model = YOLO(str(best_checkpoint))
    metrics = model.val(data=str(data_yaml_path), imgsz=image_size, device=val_device)
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
    import torch

    args = parse_args()
    setup_logging(level=getattr(logging, args.log_level.upper(), logging.INFO))

    # ── Device auto-detection ────────────────────────────────────────────────
    if args.device:
        device = args.device
        device_source = "CLI / env"
    elif torch.cuda.is_available():
        device = "cuda"
        device_source = "auto-detected"
    else:
        device = "cpu"
        device_source = "auto-detected (no GPU found)"
    # ─────────────────────────────────────────────────────────────────────────

    hyperparams = load_hyperparameters()

    raw_model = hyperparams.get("model")
    if not raw_model:
        raise SystemExit("[Train] 'model' key is missing from hyperparameters.yaml.")

    resolved_model = resolve_model_name(str(raw_model))
    output_dir = RUNS_DIR / "detect" / args.name

    # ── Clear startup banner ─────────────────────────────────────────────────
    log.info("=" * 60)
    log.info("Chipset Defect — YOLO Training")
    log.info("=" * 60)
    log.info("  Device       : %s  (%s)", device, device_source)
    log.info("  Model        : %s → %s", raw_model, resolved_model)
    log.info("  Dataset      : %s", args.data)
    log.info("  Output dir   : %s", output_dir)
    log.info("  Hyperparams  : %s", HYPERPARAMETERS_YAML)
    if device == "cuda":
        log.info("  GPU count    : %d", torch.cuda.device_count())
        log.info("  GPU name     : %s", torch.cuda.get_device_name(0))
    log.info("=" * 60)
    # ─────────────────────────────────────────────────────────────────────────

    with tempfile.TemporaryDirectory(prefix="chipset-train-") as tmp_dir:
        workspace = Path(tmp_dir)
        data_yaml_path = stage_dataset(args.data, workspace)
        model_path = stage_model(resolved_model, workspace)
        run_dir = train_model(model_path, data_yaml_path, hyperparams, device, args.name)
        best_checkpoint = copy_training_outputs(run_dir)
        validate_model(
            best_checkpoint,
            data_yaml_path,
            int(hyperparams.get("imgsz", 640)),
            device,
        )
        export_model(best_checkpoint, args.output_model)

    log.info("=" * 60)
    log.info("Training complete.")
    log.info("  Best checkpoint : %s", best_checkpoint)
    log.info("  Run artifacts   : %s", run_dir)
    log.info("=" * 60)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

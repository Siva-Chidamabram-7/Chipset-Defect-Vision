#!/usr/bin/env python3
"""
gcs_utils.py — Google Cloud Storage helpers for the PCB defect pipeline
────────────────────────────────────────────────────────────────────────
Provides transparent local ↔ GCS path support for:
  • sam_auto_annotate.py    (Step 1)
  • auto_label_with_yolo.py (Step 3)
  • training/train.py        (Step 2 / Step 4)

Usage pattern:
    from training.scripts.gcs_utils import is_gcs_path, download_gcs_dir, upload_gcs_dir

All functions are no-ops / pass-throughs when given local paths.
Requires: google-cloud-storage  (pip install -r training/requirements-training.txt)

Authentication (pick one):
  • GOOGLE_APPLICATION_CREDENTIALS=/path/to/sa-key.json  (service account)
  • gcloud auth application-default login                 (ADC, interactive)
  • Vertex AI / Cloud Run managed identity               (automatic)
"""

from __future__ import annotations

import logging
import os
import shutil
import tempfile
from pathlib import Path
from typing import Optional

log = logging.getLogger(__name__)


# ═════════════════════════════════════════════════════════════════════════════
# Path helpers
# ═════════════════════════════════════════════════════════════════════════════

def is_gcs_path(path: str) -> bool:
    """Return True if path is a GCS URI (gs://…)."""
    return str(path).startswith("gs://")


def parse_gcs_path(gcs_path: str) -> tuple[str, str]:
    """
    Split a GCS URI into (bucket_name, blob_prefix).

    Example:
        "gs://my-bucket/data/train" → ("my-bucket", "data/train")
        "gs://my-bucket/"           → ("my-bucket", "")
    """
    assert gcs_path.startswith("gs://"), f"Not a GCS path: {gcs_path}"
    without_scheme = gcs_path[5:]                    # strip "gs://"
    parts = without_scheme.split("/", 1)
    bucket = parts[0]
    prefix = parts[1] if len(parts) > 1 else ""
    return bucket, prefix


def _gcs_client():
    """Return a google.cloud.storage.Client, raising ImportError if missing."""
    try:
        from google.cloud import storage  # type: ignore
        return storage.Client()
    except ImportError as exc:
        raise ImportError(
            "google-cloud-storage is not installed.\n"
            "Install it:  pip install -r requirements-training.txt"
        ) from exc


# ═════════════════════════════════════════════════════════════════════════════
# Download helpers
# ═════════════════════════════════════════════════════════════════════════════

def download_gcs_file(gcs_path: str, local_path: Path) -> None:
    """Download a single blob from GCS to a local file."""
    bucket_name, blob_name = parse_gcs_path(gcs_path)
    client = _gcs_client()
    bucket = client.bucket(bucket_name)
    blob   = bucket.blob(blob_name)
    local_path.parent.mkdir(parents=True, exist_ok=True)
    log.info("[GCS] Downloading  %s  →  %s", gcs_path, local_path)
    blob.download_to_filename(str(local_path))
    log.info("[GCS] Download complete: %s", local_path.name)


def download_gcs_dir(gcs_path: str, local_dir: Path) -> int:
    """
    Recursively download all blobs under a GCS prefix to a local directory.

    Returns the number of files downloaded.
    """
    bucket_name, prefix = parse_gcs_path(gcs_path)
    if prefix and not prefix.endswith("/"):
        prefix += "/"

    client = _gcs_client()
    bucket = client.bucket(bucket_name)
    blobs  = list(bucket.list_blobs(prefix=prefix))

    if not blobs:
        log.warning("[GCS] No files found at %s", gcs_path)
        return 0

    log.info("[GCS] Downloading %d file(s) from  %s  →  %s", len(blobs), gcs_path, local_dir)
    count = 0
    for blob in blobs:
        rel  = blob.name[len(prefix):]   # strip the common prefix
        if not rel:                       # skip the "directory" blob itself
            continue
        dst = local_dir / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        blob.download_to_filename(str(dst))
        count += 1

    log.info("[GCS] Download complete: %d file(s)", count)
    return count


# ═════════════════════════════════════════════════════════════════════════════
# Upload helpers
# ═════════════════════════════════════════════════════════════════════════════

def upload_gcs_file(local_path: Path, gcs_path: str) -> None:
    """Upload a single local file to GCS."""
    bucket_name, blob_name = parse_gcs_path(gcs_path)
    client = _gcs_client()
    bucket = client.bucket(bucket_name)
    blob   = bucket.blob(blob_name)
    log.info("[GCS] Uploading  %s  →  %s", local_path, gcs_path)
    blob.upload_from_filename(str(local_path))
    log.info("[GCS] Upload complete: %s", Path(blob_name).name)


def upload_gcs_dir(local_dir: Path, gcs_path: str) -> int:
    """
    Recursively upload all files from a local directory to a GCS prefix.

    Returns the number of files uploaded.
    """
    bucket_name, prefix = parse_gcs_path(gcs_path)
    if prefix and not prefix.endswith("/"):
        prefix += "/"

    client = _gcs_client()
    bucket = client.bucket(bucket_name)

    files = [p for p in local_dir.rglob("*") if p.is_file()]
    log.info("[GCS] Uploading %d file(s) from  %s  →  %s", len(files), local_dir, gcs_path)
    count = 0
    for f in files:
        rel       = f.relative_to(local_dir)
        blob_name = prefix + str(rel).replace("\\", "/")
        blob      = bucket.blob(blob_name)
        blob.upload_from_filename(str(f))
        count += 1

    log.info("[GCS] Upload complete: %d file(s)", count)
    return count


# ═════════════════════════════════════════════════════════════════════════════
# Resolve path — transparently handle local OR gs:// paths
# ═════════════════════════════════════════════════════════════════════════════

def resolve_input_path(path: str, tmp_base: Path, subdir: str) -> Path:
    """
    If path is a GCS URI, download its contents to tmp_base/subdir and return
    that local path.  If it is already a local path, return it as-is.
    """
    if is_gcs_path(path):
        local = tmp_base / subdir
        local.mkdir(parents=True, exist_ok=True)
        download_gcs_dir(path, local)
        return local
    return Path(path)


def resolve_input_file(path: str, tmp_base: Path, filename: str) -> Path:
    """
    If path is a GCS URI, download the single file to tmp_base/filename and
    return the local path.  Otherwise return as-is.
    """
    if is_gcs_path(path):
        local = tmp_base / filename
        download_gcs_file(path, local)
        return local
    return Path(path)


# ═════════════════════════════════════════════════════════════════════════════
# Logging setup — Vertex AI compatible (stdout, timestamps)
# ═════════════════════════════════════════════════════════════════════════════

def setup_logging(level: int = logging.INFO) -> None:
    """
    Configure root logger for Vertex AI / Docker stdout log capture.

    Call once at the top of each pipeline script's main().
    Format: "2026-03-19 12:34:56 [INFO] script: message"
    """
    import sys
    logging.basicConfig(
        level   = level,
        format  = "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt = "%Y-%m-%d %H:%M:%S",
        handlers= [logging.StreamHandler(sys.stdout)],
        force   = True,
    )
    # Suppress noisy third-party loggers
    logging.getLogger("ultralytics").setLevel(logging.WARNING)
    logging.getLogger("PIL").setLevel(logging.WARNING)


# ═════════════════════════════════════════════════════════════════════════════
# Vertex AI environment helpers
# ═════════════════════════════════════════════════════════════════════════════

def vertex_env() -> dict[str, str]:
    """
    Read standard Vertex AI Training environment variables.

    Vertex AI injects these automatically into custom training jobs:
      AIP_MODEL_DIR          gs://…/model/      ← where to save the model
      AIP_TRAINING_DATA_URI  gs://…/data/       ← training dataset root
      AIP_CHECKPOINT_DIR     gs://…/checkpoints/

    Returns a dict with those keys (empty strings if not set).
    """
    return {
        "model_dir":     os.environ.get("AIP_MODEL_DIR",         ""),
        "training_data": os.environ.get("AIP_TRAINING_DATA_URI", ""),
        "checkpoint_dir":os.environ.get("AIP_CHECKPOINT_DIR",    ""),
    }

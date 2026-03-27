"""
app/config.py — Central configuration for the Chipset Defect Vision API.

All path constants, inference thresholds, and size limits are defined here.
Consumed by:
  - app/main.py        (API server, scan directory creation, size guard)
  - app/model/predictor.py  (model path, confidence/NMS thresholds)

Environment variables allow runtime overrides without code changes:
  INFERENCE_CONFIDENCE_THRESHOLD  — min score to keep a detection (default 0.50)
  INFERENCE_NMS_IOU_THRESHOLD     — NMS IoU cutoff (default 0.45)
  MAX_IMAGE_BYTES                 — payload size cap (default 10 MB)
"""

from __future__ import annotations

import os
from pathlib import Path

# ── Application metadata ──────────────────────────────────────────────────────
APP_VERSION = "3.0.0"   # Shown in /health response and footer model label
MODEL_NAME  = "best.pt" # Filename expected inside weights/; copied there by train.py

# ── Directory layout ──────────────────────────────────────────────────────────
# All paths are resolved relative to the project root so the app works
# regardless of the working directory from which it is launched.
PROJECT_ROOT = Path(__file__).resolve().parents[1]   # D:/Work/Chipset-Defect-Vision/
FRONTEND_DIR = PROJECT_ROOT / "frontend"              # Static HTML/CSS/JS served by FastAPI
WEIGHTS_DIR  = PROJECT_ROOT / "weights"               # Trained model checkpoints (.pt files)
MODEL_PATH   = WEIGHTS_DIR  / MODEL_NAME              # Full path to weights/best.pt
INCOMING_DIR = PROJECT_ROOT / "incoming_data"         # Raw images dropped by the camera/uploader
SCANS_DIR    = PROJECT_ROOT / "scans"                 # Per-scan subdirs: input.jpg, output.jpg, result.json, logs.txt

# ── Inference thresholds ──────────────────────────────────────────────────────
# CONFIDENCE_THRESHOLD: any YOLO detection below this score is discarded before
#   the response is built.  YOLO itself runs at a lower internal threshold (0.10)
#   so we can log the full raw distribution for debugging.
# NMS_IOU_THRESHOLD: IoU value above which two overlapping boxes are considered
#   duplicates; the lower-confidence one is suppressed.
CONFIDENCE_THRESHOLD = float(os.getenv("INFERENCE_CONFIDENCE_THRESHOLD", "0.50"))
NMS_IOU_THRESHOLD    = float(os.getenv("INFERENCE_NMS_IOU_THRESHOLD",    "0.45"))

# ── Upload size limit ─────────────────────────────────────────────────────────
# Requests carrying more than MAX_IMAGE_BYTES of image data are rejected with
# HTTP 413 before inference runs, preventing OOM on large uploads.
MAX_IMAGE_BYTES = int(os.getenv("MAX_IMAGE_BYTES", str(10 * 1024 * 1024)))  # default 10 MB

from __future__ import annotations

import os
from pathlib import Path

APP_VERSION = "3.0.0"
MODEL_NAME = "best.pt"

PROJECT_ROOT = Path(__file__).resolve().parents[1]
FRONTEND_DIR = PROJECT_ROOT / "frontend"
WEIGHTS_DIR = PROJECT_ROOT / "weights"
MODEL_PATH = WEIGHTS_DIR / MODEL_NAME
INCOMING_DIR = PROJECT_ROOT / "incoming_data"

CONFIDENCE_THRESHOLD = float(os.getenv("INFERENCE_CONFIDENCE_THRESHOLD", "0.05"))

MAX_IMAGE_BYTES = int(os.getenv("MAX_IMAGE_BYTES", str(10 * 1024 * 1024)))

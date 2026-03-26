from __future__ import annotations

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TRAINING_DIR = PROJECT_ROOT / "training"
RUNS_DIR = PROJECT_ROOT / "runs"
WEIGHTS_DIR = PROJECT_ROOT / "weights"

DEFAULT_LOCAL_DATA_CONFIG = TRAINING_DIR / "data.yaml"
DEFAULT_RUN_NAME = os.getenv("TRAIN_RUN_NAME", "pcb_solder_v1")

# Vertex AI — all values read from environment; defaults target CPU-only jobs.
# To run on GPU set VERTEX_ACCELERATOR_TYPE and VERTEX_ACCELERATOR_COUNT in env.
VERTEX_PROJECT_ID = os.getenv("GCP_PROJECT_ID", "chip-defect-identification")
VERTEX_REGION = os.getenv("GCP_REGION", "asia-south1")
VERTEX_ACCELERATOR_TYPE = os.getenv("VERTEX_ACCELERATOR_TYPE", "")   # empty = no GPU
VERTEX_ACCELERATOR_COUNT = int(os.getenv("VERTEX_ACCELERATOR_COUNT", "0"))  # 0 = CPU-only
VERTEX_MACHINE_TYPE = os.getenv("VERTEX_MACHINE_TYPE", "n1-standard-8")

# GCS paths — must be set via env before running training or vertex_job.py
VERTEX_DEFAULT_DATA_URI = os.getenv("GCS_DATA_URI", "")
VERTEX_DEFAULT_OUTPUT_URI = os.getenv("GCS_MODEL_OUTPUT_URI", "")

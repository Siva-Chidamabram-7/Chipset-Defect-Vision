"""
training/config.py — Central configuration for the YOLO training pipeline.

Consumed by:
  training/train.py      — LOCAL training (model, dataset, run paths)
  training/vertex_job.py — CLOUD training (Vertex AI project, region, machine spec)
  training/pipeline.py   — Pipeline entry point (routes to local or Vertex mode)

All Vertex AI values are read from environment variables so that the same
Docker image can be configured at job submission time without rebuilding.

Quick reference — environment variables:
  TRAIN_RUN_NAME              name of the run directory under runs/detect/
  GCP_PROJECT_ID              GCP project that owns the Vertex AI quota
  GCP_REGION                  GCP region where the training job runs
  VERTEX_MACHINE_TYPE         Vertex AI machine type (CPU tier)
  VERTEX_ACCELERATOR_TYPE     GPU type, e.g. NVIDIA_TESLA_T4 (empty = CPU-only)
  VERTEX_ACCELERATOR_COUNT    Number of GPUs (0 = CPU-only)
  GCS_DATA_URI                gs:// URI of the training dataset
  GCS_MODEL_OUTPUT_URI        gs:// URI where best.pt is uploaded after training
"""

from __future__ import annotations

import os
from pathlib import Path

# ── Directory layout ──────────────────────────────────────────────────────────
# All paths are resolved from the file location so training scripts work
# regardless of the current working directory.
PROJECT_ROOT = Path(__file__).resolve().parents[1]  # D:/Work/Chipset-Defect-Vision/
TRAINING_DIR = PROJECT_ROOT / "training"            # training/ — yaml configs, scripts
RUNS_DIR     = PROJECT_ROOT / "runs"                # runs/detect/<run_name>/ — YOLO outputs
WEIGHTS_DIR  = PROJECT_ROOT / "weights"             # weights/ — checkpoints copied here by train.py

# Path to the dataset YAML consumed by YOLO during local training
DEFAULT_LOCAL_DATA_CONFIG = TRAINING_DIR / "data.yaml"

# Default run name — creates runs/detect/pcb_solder_v1/; override via env for CI/CD
DEFAULT_RUN_NAME = os.getenv("TRAIN_RUN_NAME", "pcb_solder_v1")

# ── Vertex AI settings ────────────────────────────────────────────────────────
# All values read from environment; defaults target CPU-only jobs.
# GPU jobs require explicit VERTEX_ACCELERATOR_TYPE and ACCELERATOR_COUNT.
# Leave VERTEX_ACCELERATOR_TYPE empty (the default) for a CPU-only job;
# Vertex AI rejects the accelerator fields entirely when count is 0.
VERTEX_PROJECT_ID       = os.getenv("GCP_PROJECT_ID",            "chip-defect-identification")
VERTEX_REGION           = os.getenv("GCP_REGION",                "asia-south1")
VERTEX_ACCELERATOR_TYPE = os.getenv("VERTEX_ACCELERATOR_TYPE",   "")   # empty → no GPU
VERTEX_ACCELERATOR_COUNT= int(os.getenv("VERTEX_ACCELERATOR_COUNT", "0"))  # 0 → CPU-only
VERTEX_MACHINE_TYPE     = os.getenv("VERTEX_MACHINE_TYPE",       "n1-standard-8")

# ── GCS paths ─────────────────────────────────────────────────────────────────
# These must be set via env before running vertex_job.py or training on Vertex.
# training/train.py also reads GCS_DATA_URI to transparently download data
# from GCS when running inside the Docker training container.
VERTEX_DEFAULT_DATA_URI   = os.getenv("GCS_DATA_URI",            "")  # gs://bucket/data/
VERTEX_DEFAULT_OUTPUT_URI = os.getenv("GCS_MODEL_OUTPUT_URI",    "")  # gs://bucket/models/

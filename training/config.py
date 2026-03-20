from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TRAINING_DIR = PROJECT_ROOT / "training"
RUNS_DIR = PROJECT_ROOT / "runs"
WEIGHTS_DIR = PROJECT_ROOT / "weights"

DEFAULT_LOCAL_DATA_CONFIG = TRAINING_DIR / "data.yaml"
DEFAULT_BASE_MODEL = "yolov8n.pt"
DEFAULT_RUN_NAME = "pcb_solder_v1"

CLASS_NAMES = {
    0: "Missing_hole",
    1: "Mouse_bite",
    2: "Open_circuit",
    3: "Short",
    4: "Spur",
    5: "Spurious_copper",
    6: "Good",
}

DEFAULT_HYPERPARAMETERS = {
    "epochs": 50,
    "batch_size": 16,
    "learning_rate": 0.001,
    "image_size": 640,
    "optimizer": "AdamW",
    "device": "cpu",
    "patience": 20,
}

VERTEX_PROJECT_ID = "detection-490708"
VERTEX_REGION = "asia-south1"
VERTEX_ACCELERATOR_TYPE = "NVIDIA_TESLA_T4"
VERTEX_ACCELERATOR_COUNT = 1
VERTEX_MACHINE_TYPE = "n1-standard-8"
VERTEX_DEFAULT_DATA_URI = "gs://chip-defect-vision-bucket/data/"
VERTEX_DEFAULT_OUTPUT_URI = "gs://chip-defect-vision-bucket/data/"

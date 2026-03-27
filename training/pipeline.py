"""
training/pipeline.py — Unified training entry point for local and Vertex AI runs.

This script is the Docker container ENTRYPOINT.  It auto-detects whether it is
running locally or inside a Vertex AI custom training job by checking:
  1. /gcs mount point exists  → Vertex AI GCS fuse mount is active
  2. GCS_DATA_URI env var set → explicitly configured for cloud mode

LOCAL mode:
  Reads data from the local data/ directory.
  Delegates directly to training/train.py which handles hyperparameters,
  dataset validation, model training, and copying best.pt to weights/.

VERTEX mode:
  Reads GCS_DATA_URI / GCS_MODEL_OUTPUT_URI from environment variables
  injected by the Vertex AI training job (set in vertex_job.py worker_pool_specs).
  Delegates to training/train.py with --data pointing at the GCS dataset path.

Usage:
  python training/pipeline.py               # local
  GCS_DATA_URI=gs://... python training/pipeline.py  # force Vertex mode locally
"""

import os
import subprocess

# ── Environment detection ─────────────────────────────────────────────────────
# Vertex AI mounts the GCS bucket at /gcs when the GCS_FUSE flag is set.
# Alternatively, the presence of GCS_DATA_URI indicates a cloud training job.
GCS_DATA_URI         = os.getenv("GCS_DATA_URI", "")
GCS_MODEL_OUTPUT_URI = os.getenv("GCS_MODEL_OUTPUT_URI", "")

IS_VERTEX = os.path.exists("/gcs") or bool(GCS_DATA_URI)

# ── Path routing ──────────────────────────────────────────────────────────────
if IS_VERTEX:
    # Cloud mode — dataset lives in GCS; train.py downloads it transparently
    if not GCS_DATA_URI:
        raise SystemExit(
            "Running on Vertex AI but GCS_DATA_URI is not set. "
            "Set GCS_DATA_URI to the gs:// path of your dataset."
        )
    input_path  = GCS_DATA_URI
    output_path = GCS_MODEL_OUTPUT_URI or GCS_DATA_URI   # fall back to data bucket
else:
    # Local mode — data/ directory is expected to be populated already
    input_path  = "data"
    output_path = "data"

print(f"Running in {'VERTEX' if IS_VERTEX else 'LOCAL'} mode")
print(f"Input:  {input_path}")
print(f"Output: {output_path}")

# ── Step 1: Train ─────────────────────────────────────────────────────────────
# Delegate to train.py which:
#   • loads hyperparameters from training/hyperparameters.yaml
#   • validates the dataset structure
#   • runs YOLO training
#   • copies best.pt → weights/best.pt
#   • uploads best.pt to GCS if output_path is a gs:// URI
subprocess.run(
    ["python", "training/train.py"],
    check=True,   # propagate non-zero exit codes as exceptions
)

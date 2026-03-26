import os
import subprocess

# Detect environment: Vertex AI mounts GCS at /gcs, or GCS_DATA_URI is set
GCS_DATA_URI = os.getenv("GCS_DATA_URI", "")
GCS_MODEL_OUTPUT_URI = os.getenv("GCS_MODEL_OUTPUT_URI", "")

IS_VERTEX = os.path.exists("/gcs") or bool(GCS_DATA_URI)

if IS_VERTEX:
    if not GCS_DATA_URI:
        raise SystemExit(
            "Running on Vertex AI but GCS_DATA_URI is not set. "
            "Set GCS_DATA_URI to the gs:// path of your dataset."
        )
    input_path = GCS_DATA_URI
    output_path = GCS_MODEL_OUTPUT_URI or GCS_DATA_URI
else:
    input_path = "data"
    output_path = "data"

print(f"Running in {'VERTEX' if IS_VERTEX else 'LOCAL'} mode")
print(f"Input:  {input_path}")
print(f"Output: {output_path}")

# Step 1: Train
subprocess.run(
    ["python", "training/train.py"],
    check=True,
)

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from training.config import (
    VERTEX_DEFAULT_DATA_URI,
    VERTEX_DEFAULT_OUTPUT_URI,
    VERTEX_MACHINE_TYPE,
    VERTEX_PROJECT_ID,
    VERTEX_REGION,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Submit a CPU-based training job to Google Vertex AI.\n"
            "Hyperparameters are read from training/hyperparameters.yaml inside the container.\n"
            "GPU/accelerator support is disabled by default; pass --accelerator-type and\n"
            "--accelerator-count to re-enable when a GPU quota is available."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--project-id", default=os.getenv("GCP_PROJECT_ID", VERTEX_PROJECT_ID))
    parser.add_argument("--region", default=os.getenv("GCP_REGION", VERTEX_REGION))
    parser.add_argument("--display-name", default="chipset-defect-training")
    parser.add_argument(
        "--image-uri",
        default=os.getenv("VERTEX_TRAIN_IMAGE_URI", ""),
        help="Custom training container image in Artifact Registry (required).",
    )
    parser.add_argument(
        "--data",
        default=os.getenv("GCS_DATA_URI", VERTEX_DEFAULT_DATA_URI),
        help="Dataset gs:// prefix passed to train.py --data.",
    )
    parser.add_argument(
        "--output-model",
        default=os.getenv("GCS_MODEL_OUTPUT_URI", VERTEX_DEFAULT_OUTPUT_URI),
        help="GCS prefix where best.pt will be uploaded by train.py.",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="Training device passed to train.py --device.  Default: cpu.",
    )
    # Machine spec — default to CPU-only
    parser.add_argument(
        "--machine-type",
        default=os.getenv("VERTEX_MACHINE_TYPE", VERTEX_MACHINE_TYPE),
        help="Vertex AI machine type.  n1-standard-8 for CPU-only jobs.",
    )
    parser.add_argument(
        "--accelerator-type",
        default=os.getenv("VERTEX_ACCELERATOR_TYPE", ""),
        help="GPU accelerator type, e.g. NVIDIA_TESLA_T4.  Leave empty for CPU-only.",
    )
    parser.add_argument(
        "--accelerator-count",
        type=int,
        default=int(os.getenv("VERTEX_ACCELERATOR_COUNT", "0")),
        help="Number of GPUs to attach.  0 = CPU-only job.",
    )
    parser.add_argument("--service-account", default=os.getenv("VERTEX_SERVICE_ACCOUNT", ""))
    parser.add_argument("--staging-bucket", default=os.getenv("VERTEX_STAGING_BUCKET", ""))
    parser.add_argument("--async", dest="run_async", action="store_true", help="Return immediately.")
    return parser.parse_args()


def _build_machine_spec(args: argparse.Namespace) -> dict:
    """Build the Vertex AI machine_spec dict.

    When no GPU is requested (accelerator_count == 0), the accelerator fields
    must be *absent* from the spec — Vertex AI rejects them even when set to
    empty/zero values.
    """
    spec: dict = {"machine_type": args.machine_type}
    if args.accelerator_count > 0:
        if not args.accelerator_type:
            raise SystemExit(
                "accelerator_count > 0 but --accelerator-type is empty.  "
                "Provide a value such as NVIDIA_TESLA_T4, or set --accelerator-count 0."
            )
        spec["accelerator_type"] = args.accelerator_type
        spec["accelerator_count"] = args.accelerator_count
    return spec


def main() -> int:
    args = parse_args()

    if not args.image_uri:
        raise SystemExit(
            "Missing Vertex training image URI. "
            "Provide --image-uri or set VERTEX_TRAIN_IMAGE_URI."
        )
    if not args.data:
        raise SystemExit(
            "Missing dataset URI. Provide --data or set GCS_DATA_URI."
        )

    # Consistency guard: if a CPU device is requested, ensure no GPU is attached.
    if args.device == "cpu" and args.accelerator_count > 0:
        raise SystemExit(
            "Conflicting options: --device cpu with --accelerator-count > 0.  "
            "Either remove the accelerator flags for a CPU job, "
            "or set --device cuda for a GPU job."
        )

    try:
        from google.cloud import aiplatform
    except ImportError as exc:  # pragma: no cover
        raise SystemExit(
            "google-cloud-aiplatform is not installed.  "
            "Install training/requirements-training.txt before using vertex_job.py."
        ) from exc

    aiplatform.init(
        project=args.project_id,
        location=args.region,
        staging_bucket=args.staging_bucket or None,
    )

    # Hyperparameters are baked into the training image via hyperparameters.yaml.
    # Only operational arguments are passed to train.py here.
    container_args = [
        f"--data={args.data}",
        f"--output-model={args.output_model}",
        f"--device={args.device}",
    ]

    machine_spec = _build_machine_spec(args)

    worker_pool_specs = [
        {
            "machine_spec": machine_spec,
            "replica_count": 1,
            "container_spec": {
                "image_uri": args.image_uri,
                "args": container_args,
                "env": [
                    {"name": "AIP_TRAINING_DATA_URI", "value": args.data},
                    {"name": "AIP_MODEL_DIR", "value": args.output_model},
                    {"name": "GCS_DATA_URI", "value": args.data},
                    {"name": "GCS_MODEL_OUTPUT_URI", "value": args.output_model},
                ],
            },
        }
    ]

    job = aiplatform.CustomJob(
        display_name=args.display_name,
        worker_pool_specs=worker_pool_specs,
        base_output_dir=args.output_model if args.output_model.startswith("gs://") else None,
    )

    job.run(sync=not args.run_async, service_account=args.service_account or None)

    gpu_info = (
        f"{args.accelerator_count}× {args.accelerator_type}"
        if args.accelerator_count > 0
        else "none (CPU-only)"
    )
    print(f"Submitted Vertex AI job : {args.display_name}")
    print(f"Project                 : {args.project_id}")
    print(f"Region                  : {args.region}")
    print(f"Machine type            : {args.machine_type}")
    print(f"Accelerator             : {gpu_info}")
    print(f"Device arg              : {args.device}")
    print(f"Image                   : {args.image_uri}")
    print(f"Data                    : {args.data}")
    print(f"Output                  : {args.output_model}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

# ═══════════════════════════════════════════════════════════════════════════════
# Chipset Defect Vision — Training Dockerfile
# Base: pytorch/pytorch:2.3.0-cuda12.1-cudnn8-runtime
#
# PURPOSE
# ───────
# This image is the TRAINING container — for use locally (CPU/GPU) and on
# Vertex AI Custom Training Jobs.  It is NOT the inference server.
#
# It includes:
#   • PyTorch 2.3.0 + CUDA 12.1 (GPU-capable; falls back to CPU automatically)
#   • SAM (Segment Anything Model) for Step 1 auto-annotation
#   • YOLOv8 (ultralytics) for Steps 2–4 training + re-labelling
#   • Google Cloud Storage client for GCS I/O
#   • All pipeline scripts (scripts/ + !training/)
#
# ENTRYPOINT
# ──────────
# Defaults to train.py so the image can be used as a Vertex AI custom job
# without any wrapper script.  Override entrypoint to run other steps.
#
# BUILD
# ─────
# Local CPU/GPU build:
#   docker build -f Dockerfile.training -t pcb-training .
#
# Vertex AI (push to Artifact Registry):
#   REGION=us-central1
#   PROJECT=$(gcloud config get-value project)
#   IMAGE="${REGION}-docker.pkg.dev/${PROJECT}/chipset-defect-vision/training:latest"
#   docker build -f Dockerfile.training -t ${IMAGE} .
#   docker push ${IMAGE}
#
# RUN
# ───
# Local — GPU:
#   docker run --gpus all \
#       -v $(pwd)/weights:/app/weights \
#       -v $(pwd)/data:/app/data \
#       pcb-training \
#       python !training/train.py --device 0 --epochs 50
#
# Local — CPU:
#   docker run \
#       -v $(pwd)/weights:/app/weights \
#       -v $(pwd)/data:/app/data \
#       pcb-training \
#       python !training/train.py --device cpu --epochs 50
#
# Full bootstrap (SAM → train → re-label → retrain):
#   docker run --gpus all \
#       -v $(pwd)/weights:/app/weights \
#       -v $(pwd)/dataset:/app/dataset \
#       -v $(pwd)/data:/app/data \
#       --entrypoint bash pcb-training -c "
#         python scripts/sam_auto_annotate.py --device cuda && \
#         python !training/train.py --device 0 --epochs 50 && \
#         python scripts/auto_label_with_yolo.py --device cuda && \
#         python !training/train.py --device 0 --epochs 50
#       "
#
# Vertex AI Custom Training Job:
#   gcloud ai custom-jobs create \
#       --region=us-central1 \
#       --display-name=pcb-defect-training \
#       --worker-pool-spec=machine-type=n1-standard-8,\
#           accelerator-type=NVIDIA_TESLA_T4,accelerator-count=1,\
#           container-image-uri=${IMAGE} \
#       --args="--data=gs://my-bucket/data/,--output-model=gs://my-bucket/models/,\
#               --epochs=50,--device=0"
# ═══════════════════════════════════════════════════════════════════════════════

FROM pytorch/pytorch:2.3.0-cuda12.1-cudnn8-runtime

LABEL maintainer="Chipset Defect Vision"
LABEL description="PCB defect detection — SAM + YOLOv8 training pipeline — Vertex AI ready"

# ── Environment ───────────────────────────────────────────────────────────────
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    #
    # ── Ultralytics offline locks ─────────────────────────────────────────────
    YOLO_CONFIG_DIR=/app/.yolo \
    YOLO_TELEMETRY=False \
    YOLO_AUTOINSTALL=False \
    HF_DATASETS_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    #
    # ── GCS / Vertex AI ───────────────────────────────────────────────────────
    # GOOGLE_APPLICATION_CREDENTIALS is NOT set here — inject at runtime via
    # -e GOOGLE_APPLICATION_CREDENTIALS=/secrets/sa.json or use ADC/Workload Identity.
    PYTHONPATH=/app

WORKDIR /app

# ── Layer 1: System libraries ─────────────────────────────────────────────────
# libglib2.0-0   — cv2 runtime dep
# libgomp1       — PyTorch OpenMP backend
# libgl1         — guard against transitive cv2 GUI imports
# libglib2.0-0   — gcsfs / google-cloud-storage dep
# wget           — for downloading SAM checkpoints in CI/CD
RUN apt-get update && apt-get install -y --no-install-recommends \
        libglib2.0-0 \
        libgomp1 \
        libgl1 \
        libsm6 \
        libxext6 \
        libxrender-dev \
        wget \
        git \
    && rm -rf /var/lib/apt/lists/*

# ── Layer 2: Python training dependencies ─────────────────────────────────────
# PyTorch is already in the base image (pytorch/pytorch).
# We install the remaining dependencies from requirements-training.txt.
# opencv-python is swapped for the headless build (no libGL GUI dep).
COPY requirements-training.txt .
RUN pip install --no-cache-dir -r requirements-training.txt \
    && pip uninstall -y opencv-python || true \
    && pip install --no-cache-dir opencv-python-headless==4.9.0.80

# ── Layer 3: Application code ─────────────────────────────────────────────────
# Copy pipeline scripts and training code.
# weights/ is intentionally NOT copied — mount it at runtime or let GCS handle it.
# dataset/ / data/ are intentionally NOT copied — supply via volume or GCS.
COPY scripts/   scripts/
COPY !training/ !training/
COPY app/       app/

# ── Layer 4: Runtime user & permissions ───────────────────────────────────────
RUN addgroup --system appgroup \
    && adduser --system --ingroup appgroup --home /app --no-create-home appuser \
    && mkdir -p /app/.yolo /app/weights /app/runs \
    && chown -R appuser:appgroup /app

USER appuser

# ── Healthcheck (training jobs don't have HTTP but keep for Vertex AI probes) ─
HEALTHCHECK --interval=60s --timeout=10s --retries=2 \
    CMD python -c "import ultralytics; import torch; print('ok')"

# ── Entrypoint ────────────────────────────────────────────────────────────────
# Default: run train.py.  Override for SAM or YOLO re-labelling steps.
# All configuration is done via CLI args or environment variables.
ENTRYPOINT ["python", "!training/train.py"]
CMD []

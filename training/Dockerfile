FROM pytorch/pytorch:2.3.0-cuda12.1-cudnn8-runtime

LABEL maintainer="Chipset Defect Vision"
LABEL description="Minimal Vertex AI training image for Chipset Defect Vision"

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    YOLO_CONFIG_DIR=/app/.yolo \
    YOLO_TELEMETRY=False \
    YOLO_AUTOINSTALL=False \
    HF_DATASETS_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    PYTHONPATH=/app

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
        libglib2.0-0 \
        libgomp1 \
        libgl1 \
        libsm6 \
        libxext6 \
        libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

COPY requirements-training.txt /app/training/requirements-training.txt

RUN pip install --no-cache-dir -r /app/training/requirements-training.txt \
    && pip uninstall -y opencv-python || true \
    && pip install --no-cache-dir opencv-python-headless==4.9.0.80

COPY . /app/training

RUN addgroup --system appgroup \
    && adduser --system --ingroup appgroup --home /app --no-create-home appuser \
    && mkdir -p /app/.yolo /app/weights /app/runs \
    && chown -R appuser:appgroup /app

USER appuser

CMD ["python", "training/train.py"]

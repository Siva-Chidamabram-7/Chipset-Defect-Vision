# ═══════════════════════════════════════════════════════════════════════════════
# Chipset Defect Vision — Production Inference Dockerfile
# Base: python:3.10-slim (Debian Bookworm)
#
# OFFLINE CONTRACT
# ────────────────
# Build  → requires internet (Python deps) + weights/best.pt present on disk
# Run    → fully air-gapped (zero network calls; all assets baked in)
#
# Pre-build requirement:
#   weights/best.pt must exist before running docker build.
#   Copy it from your training output:
#     cp runs/detect/train/weights/best.pt weights/best.pt
#
# Build:  docker build -t chipset-defect-vision .
# Run:    docker run --network none -p 8080:8080 chipset-defect-vision
# ═══════════════════════════════════════════════════════════════════════════════

FROM python:3.10-slim

LABEL maintainer="Chipset Defect Vision"
LABEL description="YOLOv8 PCB solder defect detection — 7 classes — inference only — offline"

# ── Offline environment locks ──────────────────────────────────────────────────
# YOLO_CONFIG_DIR  → redirect ultralytics config writes away from /root
# YOLO_TELEMETRY   → disable outbound analytics
# YOLO_AUTOINSTALL → prevent GitHub version checks
# HF_*_OFFLINE     → block any HuggingFace Hub downloads
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    YOLO_CONFIG_DIR=/app/.yolo \
    YOLO_TELEMETRY=False \
    YOLO_AUTOINSTALL=False \
    HF_DATASETS_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    PORT=8080

WORKDIR /app

# ── Layer 1: System libraries ─────────────────────────────────────────────────
# libglib2.0-0  required by opencv-python-headless
# libgomp1      required by PyTorch (OpenMP runtime)
# libgl1        guards against transitive deps that import cv2 GUI code
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        libglib2.0-0 \
        libgomp1 \
        libgl1 \
    && rm -rf /var/lib/apt/lists/*

# ── Layer 2: PyTorch CPU-only wheel ───────────────────────────────────────────
# Explicit CPU index keeps this layer at ~500 MB instead of ~2 GB.
# Pinned pair — verified stable with ultralytics 8.2.x on Cloud Run.
RUN pip install --no-cache-dir \
        torch==2.3.0 \
        torchvision==0.18.0 \
        --index-url https://download.pytorch.org/whl/cpu

# ── Layer 3: Application Python dependencies ──────────────────────────────────
# ultralytics pulls in opencv-python (full GUI build).  We immediately swap it
# for the headless variant — same cv2 API surface, no libGL requirement.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt \
    && pip uninstall -y opencv-python 2>/dev/null || true \
    && pip install --no-cache-dir opencv-python-headless==4.9.0.80

# ── Layer 4: Application code ─────────────────────────────────────────────────
# Only the three directories the server actually serves from.
# Everything else (.dockerignore default-deny) never enters the build context.
COPY app/      app/
COPY frontend/ frontend/

# ── Layer 5: Trained model ────────────────────────────────────────────────────
# ONLY weights/best.pt — no other weight files are needed or permitted.
# The .dockerignore default-deny means only this exact file can land here.
RUN mkdir -p weights/
COPY weights/best.pt weights/best.pt

# ── Layer 5a: Build-time model verification ────────────────────────────────────
# Hard-fail the build if best.pt is missing or suspiciously small.
# A missing model at build time is infinitely better than a silent failure
# at runtime in production.
RUN python - <<'EOF'
from pathlib import Path
p = Path("/app/weights/best.pt")
if not p.exists():
    raise SystemExit(
        "\n[Docker] FATAL: weights/best.pt not found.\n"
        "         Run training and copy the result first:\n"
        "           cp runs/detect/train/weights/best.pt weights/best.pt\n"
        "         Then rebuild: docker build -t chipset-defect-vision .\n"
    )
mb = round(p.stat().st_size / 1_048_576, 1)
if mb < 1:
    raise SystemExit(
        f"\n[Docker] FATAL: weights/best.pt is only {mb} MB — likely corrupt.\n"
        "         Re-copy the file from your training output and rebuild.\n"
    )
print(f"[Docker] ✓ best.pt verified: {mb} MB — build OK")
EOF

# ── Layer 6: Non-root runtime user ────────────────────────────────────────────
# appuser/appgroup — principle of least privilege for Cloud Run + K8s.
# incoming_data is pre-created here so the non-root user can write to it.
# chown runs after all COPY/RUN instructions so no file is missed.
RUN addgroup --system appgroup \
    && adduser --system --ingroup appgroup --home /app --no-create-home appuser \
    && mkdir -p /app/.yolo /app/incoming_data \
    && chown -R appuser:appgroup /app

USER appuser

# ── Healthcheck ───────────────────────────────────────────────────────────────
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8080/health')"

# ── Entrypoint ────────────────────────────────────────────────────────────────
EXPOSE 8080
CMD ["python", "-m", "uvicorn", "app.main:app", \
     "--host", "0.0.0.0", \
     "--port", "8080", \
     "--workers", "1", \
     "--log-level", "info"]

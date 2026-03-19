# ═══════════════════════════════════════════════════════════════════════════════
# Chipset Defect Vision — Offline-First Production Dockerfile
# Base: python:3.10-slim (Debian Bookworm)
#
# OFFLINE CONTRACT
# ────────────────
# Build  → requires internet (downloads Python deps) + weights/best.pt on disk
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
LABEL description="YOLOv8 PCB solder defect detection — FastAPI + glassmorphic UI — offline"

# ── Architecture note ─────────────────────────────────────────────────────────
# This image is the INFERENCE server only.
#
# SAM (Segment Anything Model) is intentionally NOT included here.
# SAM is a dataset-creation tool — it runs once offline to generate region
# proposals that a human then labels.  It is never called during inference.
#
#   Dataset creation (local workstation, needs display):
#     pip install -r requirements-sam.txt
#     python scripts/generate_regions.py   # SAM → JSON region proposals
#     python scripts/annotate.py           # human labeling → YOLO .txt
#
#   Training (local or GPU machine):
#     python training/train.py             # YOLO fine-tuning → weights/best.pt
#
#   Inference (this image, fully offline):
#     docker run --network none -p 8080:8080 chipset-defect-vision
#
# To include SAM in this image (e.g. to run scripts/ inside Docker), uncomment
# the SAM layer below.  You will also need to mount the SAM checkpoint:
#   docker run -v $(pwd)/weights/sam_vit_b.pth:/app/weights/sam_vit_b.pth ...
# ─────────────────────────────────────────────────────────────────────────────

# ── Environment ───────────────────────────────────────────────────────────────
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    #
    # ── Ultralytics offline locks ─────────────────────────────────────────────
    # Redirect config/telemetry writes away from /root so any user can run.
    YOLO_CONFIG_DIR=/app/.yolo \
    # Disable telemetry sync (no outbound analytics calls at runtime).
    YOLO_TELEMETRY=False \
    # Prevent ultralytics from checking GitHub for newer versions.
    YOLO_AUTOINSTALL=False \
    # HuggingFace offline mode — blocks any HF Hub downloads.
    HF_DATASETS_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    #
    PORT=8080

WORKDIR /app

# ── Layer 1: System libraries ─────────────────────────────────────────────────
# opencv-python-headless needs only libglib2.0-0 + libgomp1.
# libgl1 guards against transitive deps that import cv2 GUI code.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libglib2.0-0 \
        libgomp1 \
        libgl1 \
    && rm -rf /var/lib/apt/lists/*

# ── Layer 2: PyTorch CPU wheel ────────────────────────────────────────────────
# Explicit CPU-only index keeps the layer at ~500 MB instead of ~2 GB.
# Pinned pair — known stable with ultralytics 8.2.x.
RUN pip install --no-cache-dir \
        torch==2.3.0 \
        torchvision==0.18.0 \
        --index-url https://download.pytorch.org/whl/cpu

# ── Layer 3: Application Python dependencies ──────────────────────────────────
# ultralytics pulls in opencv-python (GUI build).  We immediately swap it for
# the headless build — same cv2 API, no libGL runtime requirement.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt \
    && pip uninstall -y opencv-python || true \
    && pip install --no-cache-dir opencv-python-headless==4.9.0.80

# ── Layer 4 (optional): SAM — uncomment to include in this image ─────────────
# segment-anything is pure Python (~small); the weights (~375 MB) must be
# mounted at runtime — they are NOT baked in to keep the image lean.
# Uncomment the block below and rebuild if you want to run scripts/ in Docker.
#
# RUN pip install --no-cache-dir segment-anything==1.0

# ── Layer 5: Application code ─────────────────────────────────────────────────
# Inference container only.  Intentionally omitted:
#   training/  — YOLO fine-tuning scripts; run locally, not in the container.
#   scripts/   — SAM dataset-creation tools; require a GUI + heavy deps.
#   data/      — training images/labels; excluded by .dockerignore.
#   raw_data/  — pre-split source images; excluded by .dockerignore.
COPY app/      app/
COPY frontend/ frontend/
# weights/best.pt MUST exist locally before running docker build.
# It is intentionally gitignored — copy it from your training output first:
#   cp runs/detect/train/weights/best.pt weights/best.pt
COPY weights/  weights/

# ── Layer 5a: Verify best.pt is present — hard-fail the build if missing ──────
# A missing model must be caught at build time, not at runtime.
RUN test -f /app/weights/best.pt \
    && python -c "from pathlib import Path; p=Path('/app/weights/best.pt'); print('[Docker] best.pt verified: '+str(round(p.stat().st_size/1048576,1))+' MB')" \
    || (echo '[Docker] ERROR: weights/best.pt not found — build aborted.' \
        && echo '         Run training and copy best.pt → weights/best.pt, then rebuild.' \
        && exit 1)

# ── Layer 6: Runtime user & permissions ───────────────────────────────────────
# Non-root for security.  Home=/app so ultralytics fallback config writes
# land in /app/.yolo (already set via YOLO_CONFIG_DIR).
# chown runs AFTER all COPY instructions so nothing is missed.
RUN addgroup --system appgroup \
    && adduser --system --ingroup appgroup --home /app --no-create-home appuser \
    && mkdir -p /app/.yolo \
    && chown -R appuser:appgroup /app

USER appuser

# ── Healthcheck ───────────────────────────────────────────────────────────────
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8080/health')"

# ── Entrypoint ────────────────────────────────────────────────────────────────
EXPOSE 8080
CMD ["python", "-m", "uvicorn", "app.main:app", \
     "--host", "0.0.0.0", "--port", "8080", \
     "--workers", "1", "--log-level", "info"]

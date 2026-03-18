# ═══════════════════════════════════════════════════════════════════════════════
# Chipset Defect Vision — Production Dockerfile
# Single-stage build on python:3.10-slim (Debian Bookworm)
#
# Build:  docker build -t chipset-defect-vision .
# Run:    docker run -p 8080:8080 chipset-defect-vision
# ═══════════════════════════════════════════════════════════════════════════════

FROM python:3.10-slim

LABEL maintainer="Chipset Defect Vision"
LABEL description="YOLOv8 PCB solder defect detection — FastAPI + glassmorphic UI"

# ── Environment ────────────────────────────────────────────────────────────────
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    # Redirect all ultralytics config/telemetry writes away from /root
    # so the app works correctly under any user (including non-root).
    YOLO_CONFIG_DIR=/app/.yolo \
    # Silence ultralytics first-run telemetry prompt
    YOLO_TELEMETRY=False \
    PORT=8080

WORKDIR /app

# ── Layer 1: System libraries ─────────────────────────────────────────────────
# opencv-python-headless only needs libglib2.0-0 + libgomp1 (no libGL required).
# libgl1 is listed as a safe fallback in case a transitive dep pulls cv2 GUI code.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libglib2.0-0 \
        libgomp1 \
        libgl1 \
    && rm -rf /var/lib/apt/lists/*

# ── Layer 2: PyTorch CPU wheel ────────────────────────────────────────────────
# Install from the official CPU-only index to avoid the 2 GB GPU bundle from PyPI.
# Pinned to 2.3.0/0.18.0 — a known-stable pair that works with ultralytics 8.2.x.
RUN pip install --no-cache-dir \
        torch==2.3.0 \
        torchvision==0.18.0 \
        --index-url https://download.pytorch.org/whl/cpu

# ── Layer 3: Application dependencies ────────────────────────────────────────
# ultralytics pulls in opencv-python (full, with libGL) as a dependency.
# We immediately swap it out for the headless variant — same cv2 API, no display
# requirement, and a smaller footprint inside a container.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt \
    && pip uninstall -y opencv-python || true \
    && pip install --no-cache-dir opencv-python-headless==4.9.0.80

# ── Layer 4: Pre-download YOLOv8n base weights ────────────────────────────────
# Bake the base weights into the image so the container never needs internet
# access at runtime (critical for Cloud Run and air-gapped environments).
# If the download fails during build (e.g., no network), we print a warning and
# continue — the predictor will attempt a runtime download as a last resort.
RUN mkdir -p /app/weights && \
    python -c "\
import os, pathlib; \
os.chdir('/app/weights'); \
from ultralytics import YOLO; \
YOLO('yolov8n.pt'); \
p = pathlib.Path('yolov8n.pt'); \
print(f'[Docker] yolov8n.pt ready: {p} ({p.stat().st_size // 1024 // 1024} MB)' if p.exists() else '[Docker] WARNING: weight file not found after download'); \
" || echo "[Docker] WARNING: Weight pre-download failed — will retry at runtime"

# ── Layer 5: Application code ─────────────────────────────────────────────────
COPY app/      app/
COPY frontend/ frontend/
# Copy user-provided weights last so best.pt (fine-tuned) overrides nothing,
# and the pre-downloaded yolov8n.pt from layer 4 is preserved as fallback.
COPY weights/  weights/

# ── Layer 6: Runtime user & permissions ──────────────────────────────────────
# Run as a non-root user for security.
# --home /app  → appuser home is /app; ultralytics fallback writes land in /app/.yolo
# All app files are chowned AFTER the COPY instructions so nothing is missed.
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

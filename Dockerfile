# ═══════════════════════════════════════════════════════════════════════════════
# Chipset Defect Vision — Dockerfile
# ═══════════════════════════════════════════════════════════════════════════════
# Multi-stage build:
#   Stage 1 (builder) — install Python deps into a venv
#   Stage 2 (runtime) — slim image with only the venv + app code
# ═══════════════════════════════════════════════════════════════════════════════

# ── Stage 1: builder ──────────────────────────────────────────────────────────
FROM python:3.10-slim AS builder

WORKDIR /build

# System libs needed to compile opencv / torch wheels
RUN apt-get update && apt-get install -y --no-install-recommends \
        gcc g++ libglib2.0-0 libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*

# Create venv
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Install Python deps (CPU-only torch to keep image lean)
COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install --no-cache-dir \
        torch torchvision --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir -r requirements.txt

# Pre-download YOLOv8n base weights (avoids runtime download)
RUN python -c "from ultralytics import YOLO; YOLO('yolov8n.pt')" || true


# ── Stage 2: runtime ──────────────────────────────────────────────────────────
FROM python:3.10-slim AS runtime

LABEL maintainer="Chipset Defect Vision"
LABEL description="YOLOv8 PCB solder defect detection — FastAPI + glassmorphic UI"

# Runtime system libs only
RUN apt-get update && apt-get install -y --no-install-recommends \
        libglib2.0-0 libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*

# Copy venv from builder
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy pre-downloaded YOLO weights cache
COPY --from=builder /root/.config /root/.config
COPY --from=builder /root/ultralytics /root/ultralytics 2>/dev/null || true

WORKDIR /app

# Copy application code
COPY app/       app/
COPY frontend/  frontend/
COPY weights/   weights/

# Cloud Run expects $PORT; default 8080
ENV PORT=8080
EXPOSE 8080

# Non-root user for security
RUN addgroup --system appgroup && adduser --system --ingroup appgroup appuser
RUN chown -R appuser:appgroup /app
USER appuser

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:${PORT}/health')"

CMD ["python", "-m", "uvicorn", "app.main:app", \
     "--host", "0.0.0.0", "--port", "8080", \
     "--workers", "1", "--log-level", "info"]

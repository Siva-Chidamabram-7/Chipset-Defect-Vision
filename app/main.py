"""
app/main.py — FastAPI application for the Chipset Defect Vision inference service.

This is the main server module.  It wires together:
  • SolderDefectPredictor  (app/model/predictor.py) — YOLO inference
  • image_utils            (app/utils/image_utils.py) — byte decoding/encoding
  • schemas                (app/schemas.py)           — Pydantic response models
  • config                 (app/config.py)            — all path and threshold constants

Routes exposed:
  GET  /           → serves frontend/index.html
  GET  /health     → liveness probe; also used by frontend to show model info
  POST /predict    → accepts a JPEG/PNG file OR a base64 string, returns JSON

Persistence:
  Every /predict call creates a timestamped subdirectory under scans/:
    scans/<UTC-timestamp>/
      input.jpg    — raw image as uploaded
      output.jpg   — annotated image with bounding boxes
      result.json  — slim prediction JSON (no base64 fields)
      logs.txt     — machine-readable per-detection log lines
      meta.txt     — single human-readable summary line

Start the server:
  uvicorn app.main:app --reload       # from project root (development)
  python -m main                      # via project-root re-export
"""

from __future__ import annotations

import base64
import json
import logging
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
import uvicorn

from app.config import (
    APP_VERSION,
    FRONTEND_DIR,
    INCOMING_DIR,
    MAX_IMAGE_BYTES,
    MODEL_NAME,
    SCANS_DIR,
)
from app.model.predictor import SolderDefectPredictor
from app.schemas import HealthResponse, PredictionResponse
from app.utils.image_utils import decode_base64_image, decode_image_bytes, validate_image

# Named logger — output is captured by uvicorn and surfaced in the terminal
logger = logging.getLogger("chipset_defect_vision.api")


# ── Application lifespan ─────────────────────────────────────────────────────
# FastAPI runs this async context manager once at startup (before the first
# request) and once at shutdown (after the last request).
# We use it to:
#   1. Create required directories so they exist before any request arrives.
#   2. Load the YOLO model into memory so inference is fast from the first call.
# The predictor is stored on application.state so every request handler can
# access it via request.app.state.predictor without a global variable.
@asynccontextmanager
async def lifespan(application: FastAPI):
    INCOMING_DIR.mkdir(parents=True, exist_ok=True)  # drop-zone for incoming images
    SCANS_DIR.mkdir(parents=True, exist_ok=True)      # parent dir for per-scan subdirs
    application.state.predictor = SolderDefectPredictor()  # loads weights/best.pt into RAM
    logger.info("Inference service ready")
    logger.info("Model loaded: %s", application.state.predictor.is_ready())
    logger.info("Scans directory: %s", SCANS_DIR)
    yield  # server is running — handle requests
    logger.info("Inference service stopping")


# ── FastAPI app instance ──────────────────────────────────────────────────────
# lifespan= wires the startup/shutdown hook defined above.
# The OpenAPI docs are auto-generated from Pydantic schemas in app/schemas.py.
app = FastAPI(
    title="Chipset Defect Vision API",
    description="Offline-only YOLOv8 PCB defect inference service",
    version=APP_VERSION,
    lifespan=lifespan,
)

# ── CORS middleware ───────────────────────────────────────────────────────────
# Allow all origins so the frontend can call the API from any host.
# This is safe because the service is offline-only and never exposed to the
# public internet.  Restrict origins here if needed in a production deployment.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Static file serving ───────────────────────────────────────────────────────
# The frontend/ directory is mounted at /static so browsers can load
# styles.css (/static/styles.css) and script.js (/static/script.js).
app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/", include_in_schema=False)
async def root():
    """Serve the single-page frontend application.

    include_in_schema=False hides this route from the OpenAPI /docs page
    since it returns HTML, not JSON.
    Connected to: frontend/index.html → loads script.js and styles.css
    """
    return FileResponse(str(FRONTEND_DIR / "index.html"))


@app.get("/health", response_model=HealthResponse)
async def health(request: Request):
    """Liveness and readiness probe.

    Called by:
      • The frontend every 30 s (HEALTH_INTERVAL in script.js) to update the
        status indicator dot and display the active model name in the footer.
      • Docker / Kubernetes health checks to gate traffic routing.

    Returns HTTP 200 even when model_loaded is False so the service can
    report its own degraded state rather than appearing unreachable.
    """
    predictor: SolderDefectPredictor = request.app.state.predictor
    return {
        "status": "ok",
        "model_loaded": predictor.is_ready(),
        "model": MODEL_NAME,
        "version": APP_VERSION,
    }


# ── Scan record helpers ───────────────────────────────────────────────────────
# These private functions handle the file system work for persisting each scan.
# They are called from the /predict route after inference completes.

def _scan_timestamp() -> str:
    """Return a UTC timestamp string used as the scan directory name.

    Format: 20260327T160654620622Z  (YYYYMMDDTHHMMSSffffffZ)
    The microsecond precision makes collisions in concurrent requests vanishingly
    unlikely without needing a database or counter.
    """
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")


def _create_scan_record(image_bytes: bytes) -> Path:
    """Create a per-scan directory under SCANS_DIR and save the raw input image.

    Called at the start of each /predict request so the input is persisted
    even if inference later crashes.  Returns the scan directory Path so
    _finalise_scan() can write the remaining artefacts into the same dir.
    """
    ts = _scan_timestamp()
    scan_dir = SCANS_DIR / ts       # e.g. scans/20260327T160654620622Z/
    scan_dir.mkdir(parents=True, exist_ok=True)
    (scan_dir / "input.jpg").write_bytes(image_bytes)  # raw bytes as uploaded
    logger.info("[scan] created record: %s", scan_dir)
    return scan_dir


def _finalise_scan(scan_dir: Path, prediction: dict, inference_ms: float) -> None:
    """Write output.jpg, result.json, logs.txt, and meta.txt into scan_dir.

    Called after a successful inference call.  The four artefacts serve:
      output.jpg   — human review of annotated bounding boxes
      result.json  — structured data for downstream tools / dashboards
      logs.txt     — machine-parseable per-detection lines (CI/CD, alerting)
      meta.txt     — one-liner for quick grep / tail inspection

    Base64 fields are stripped from result.json to keep file size small.
    """
    ts_iso = datetime.now(timezone.utc).isoformat(timespec="seconds")

    # output.jpg — annotated image decoded from base64 returned by the predictor
    try:
        annotated_bytes = base64.b64decode(prediction["annotated_image_base64"])
        (scan_dir / "output.jpg").write_bytes(annotated_bytes)
    except Exception as exc:
        logger.warning("[scan] could not save output.jpg: %s", exc)

    # result.json — structured prediction without the large base64 image fields
    result_slim = {k: v for k, v in prediction.items()
                   if k not in ("annotated_image_base64", "image")}
    (scan_dir / "result.json").write_text(
        json.dumps(result_slim, indent=2), encoding="utf-8"
    )

    # logs.txt — key=value lines; one per detection so grep/awk can filter easily
    defect_count = prediction.get("defect_count", 0)
    detections   = prediction.get("detections", [])
    log_lines = [
        f"timestamp={ts_iso}",
        f"model={MODEL_NAME}",
        f"inference_ms={inference_ms:.2f}",
        f"status={prediction.get('status', 'UNKNOWN')}",
        f"defect_count={defect_count}",
    ]
    for i, det in enumerate(detections, 1):
        log_lines.append(
            f"detection[{i}] class={det.get('class', det.get('label', '?'))} "
            f"confidence={det.get('confidence', 0):.4f} "
            f"bbox={det.get('bbox', [])}"
        )
    (scan_dir / "logs.txt").write_text("\n".join(log_lines) + "\n", encoding="utf-8")

    # meta.txt — single summary line for quick terminal inspection (tail, head)
    status = prediction.get("status", "UNKNOWN")
    noun   = "defect" if defect_count == 1 else "defects"
    meta   = f"{status} | {defect_count} {noun} | {inference_ms:.1f}ms | {ts_iso}\n"
    (scan_dir / "meta.txt").write_text(meta, encoding="utf-8")

    logger.info("[scan] finalised: %s — %s", scan_dir.name, meta.strip())


@app.post("/predict", response_model=PredictionResponse)
async def predict(
    request: Request,
    file: UploadFile = File(None),         # multipart file upload (from file picker/drop zone)
    image_base64: str = Form(None),        # base64-encoded image (from camera capture)
):
    """Run YOLO inference on an uploaded PCB image and return detection results.

    Accepts exactly one of:
      - file         — a multipart/form-data image upload
      - image_base64 — a base64-encoded JPEG string (with or without data-URL prefix)

    Request flow:
      1. Validate that exactly one source is provided (HTTP 400 otherwise).
      2. Read / decode the image bytes.
      3. Enforce MAX_IMAGE_BYTES size limit (HTTP 413 if exceeded).
      4. Validate the file magic bytes (HTTP 400 for unsupported formats).
      5. Decode bytes to a BGR ndarray via OpenCV.
      6. Persist raw input to scans/<ts>/input.jpg.
      7. Run SolderDefectPredictor.predict() — the core YOLO inference.
      8. Write scan artefacts (output.jpg, result.json, logs.txt, meta.txt).
      9. Return the full PredictionResponse JSON.

    Connected to:
      - frontend/script.js → runInference() builds a FormData and POST /predict
      - app/model/predictor.py → SolderDefectPredictor.predict()
      - app/schemas.py → PredictionResponse shape
    """
    predictor: SolderDefectPredictor = request.app.state.predictor

    # ── Input validation: exactly one source required ─────────────────────────
    if (file is None and image_base64 is None) or (file is not None and image_base64 is not None):
        raise HTTPException(
            status_code=400,
            detail="Provide exactly one of 'file' or 'image_base64'.",
        )

    # ── Decode image bytes from the chosen source ─────────────────────────────
    try:
        image_bytes = await file.read() if file is not None else decode_base64_image(image_base64)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    # ── Payload size guard (rejects before OpenCV decode to avoid OOM) ────────
    if len(image_bytes) > MAX_IMAGE_BYTES:
        raise HTTPException(status_code=413, detail="Image payload exceeds the configured size limit.")

    # ── Format validation via magic bytes ─────────────────────────────────────
    if not validate_image(image_bytes):
        raise HTTPException(status_code=400, detail="Unsupported image format.")

    # ── OpenCV decode: bytes → BGR ndarray ────────────────────────────────────
    try:
        image_bgr = decode_image_bytes(image_bytes)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    # ── Persist raw input before inference (survives crashes) ─────────────────
    scan_dir = _create_scan_record(image_bytes)

    # ── YOLO inference ────────────────────────────────────────────────────────
    started_at = perf_counter()
    try:
        prediction = predictor.predict(image_bgr)
    except Exception as exc:
        logger.exception("[predict] inference failed: %s", exc)
        raise HTTPException(status_code=500, detail=f"Inference error: {exc}") from exc

    # ── Timing ────────────────────────────────────────────────────────────────
    inference_ms = round((perf_counter() - started_at) * 1000, 2)
    prediction["timings"] = {"inference_ms": inference_ms}   # injected into response

    # ── Persist scan artefacts ────────────────────────────────────────────────
    _finalise_scan(scan_dir, prediction, inference_ms)

    return prediction


# ── Direct execution entry point ──────────────────────────────────────────────
# Preferred launch method: uvicorn main:app (via project-root re-export)
# This block is kept as a fallback for `python app/main.py` invocations.
if __name__ == "__main__":
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=False)

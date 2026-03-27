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

logger = logging.getLogger("chipset_defect_vision.api")


@asynccontextmanager
async def lifespan(application: FastAPI):
    INCOMING_DIR.mkdir(parents=True, exist_ok=True)
    SCANS_DIR.mkdir(parents=True, exist_ok=True)
    application.state.predictor = SolderDefectPredictor()
    logger.info("Inference service ready")
    logger.info("Model loaded: %s", application.state.predictor.is_ready())
    logger.info("Scans directory: %s", SCANS_DIR)
    yield
    logger.info("Inference service stopping")


app = FastAPI(
    title="Chipset Defect Vision API",
    description="Offline-only YOLOv8 PCB defect inference service",
    version=APP_VERSION,
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")


@app.get("/", include_in_schema=False)
async def root():
    return FileResponse(str(FRONTEND_DIR / "index.html"))


@app.get("/health", response_model=HealthResponse)
async def health(request: Request):
    predictor: SolderDefectPredictor = request.app.state.predictor
    return {
        "status": "ok",
        "model_loaded": predictor.is_ready(),
        "model": MODEL_NAME,
        "version": APP_VERSION,
    }


def _scan_timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")


def _create_scan_record(image_bytes: bytes) -> Path:
    """Create a per-scan directory and save the raw input image. Returns the scan dir Path."""
    ts = _scan_timestamp()
    scan_dir = SCANS_DIR / ts
    scan_dir.mkdir(parents=True, exist_ok=True)
    (scan_dir / "input.jpg").write_bytes(image_bytes)
    logger.info("[scan] created record: %s", scan_dir)
    return scan_dir


def _finalise_scan(scan_dir: Path, prediction: dict, inference_ms: float) -> None:
    """Write output.jpg, result.json, logs.txt, and meta.txt into scan_dir."""
    ts_iso = datetime.now(timezone.utc).isoformat(timespec="seconds")

    # output.jpg — annotated image
    try:
        annotated_bytes = base64.b64decode(prediction["annotated_image_base64"])
        (scan_dir / "output.jpg").write_bytes(annotated_bytes)
    except Exception as exc:
        logger.warning("[scan] could not save output.jpg: %s", exc)

    # result.json — full result without bulky base64 fields
    result_slim = {k: v for k, v in prediction.items()
                   if k not in ("annotated_image_base64", "image")}
    (scan_dir / "result.json").write_text(
        json.dumps(result_slim, indent=2), encoding="utf-8"
    )

    # logs.txt — machine-readable per-scan log
    defect_count = prediction.get("defect_count", 0)
    detections = prediction.get("detections", [])
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

    # meta.txt — single human-readable summary line
    status = prediction.get("status", "UNKNOWN")
    noun = "defect" if defect_count == 1 else "defects"
    meta = f"{status} | {defect_count} {noun} | {inference_ms:.1f}ms | {ts_iso}\n"
    (scan_dir / "meta.txt").write_text(meta, encoding="utf-8")

    logger.info("[scan] finalised: %s — %s", scan_dir.name, meta.strip())


@app.post("/predict", response_model=PredictionResponse)
async def predict(
    request: Request,
    file: UploadFile = File(None),
    image_base64: str = Form(None),
):
    predictor: SolderDefectPredictor = request.app.state.predictor

    if (file is None and image_base64 is None) or (file is not None and image_base64 is not None):
        raise HTTPException(
            status_code=400,
            detail="Provide exactly one of 'file' or 'image_base64'.",
        )

    try:
        image_bytes = await file.read() if file is not None else decode_base64_image(image_base64)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    if len(image_bytes) > MAX_IMAGE_BYTES:
        raise HTTPException(status_code=413, detail="Image payload exceeds the configured size limit.")

    if not validate_image(image_bytes):
        raise HTTPException(status_code=400, detail="Unsupported image format.")

    try:
        image_bgr = decode_image_bytes(image_bytes)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    scan_dir = _create_scan_record(image_bytes)

    started_at = perf_counter()
    try:
        prediction = predictor.predict(image_bgr)
    except Exception as exc:
        logger.exception("[predict] inference failed: %s", exc)
        raise HTTPException(status_code=500, detail=f"Inference error: {exc}") from exc

    inference_ms = round((perf_counter() - started_at) * 1000, 2)
    prediction["timings"] = {"inference_ms": inference_ms}

    _finalise_scan(scan_dir, prediction, inference_ms)

    return prediction


if __name__ == "__main__":
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=False)

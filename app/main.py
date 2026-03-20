from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from datetime import datetime, timezone
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
)
from app.model.predictor import SolderDefectPredictor
from app.schemas import HealthResponse, PredictionResponse
from app.utils.image_utils import decode_base64_image, decode_image_bytes, validate_image

logger = logging.getLogger("chipset_defect_vision.api")


@asynccontextmanager
async def lifespan(application: FastAPI):
    INCOMING_DIR.mkdir(parents=True, exist_ok=True)
    application.state.predictor = SolderDefectPredictor()
    logger.info("Inference service ready")
    logger.info("Model loaded: %s", application.state.predictor.is_ready())
    logger.info("Incoming directory: %s", INCOMING_DIR)
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


def _persist_incoming_image(image_bytes: bytes) -> None:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    (INCOMING_DIR / f"{timestamp}.jpg").write_bytes(image_bytes)


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

    _persist_incoming_image(image_bytes)

    started_at = perf_counter()
    prediction = predictor.predict(image_bgr)
    prediction["timings"] = {
        "inference_ms": round((perf_counter() - started_at) * 1000, 2),
    }
    return prediction


if __name__ == "__main__":
    uvicorn.run("app.main:app", host="0.0.0.0", port=8080, reload=False)

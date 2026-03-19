import os
import time
import tempfile
from pathlib import Path

from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

from app.model.predictor import SolderDefectPredictor
from app.utils.image_utils import decode_base64_image, validate_image

# ── App Init ──────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Chipset Defect Vision API",
    description="YOLOv8-powered PCB defect detection — 7 defect classes",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Model ─────────────────────────────────────────────────────────────────────
predictor = SolderDefectPredictor()

# ── Static Frontend ───────────────────────────────────────────────────────────
frontend_dir = Path(__file__).parent.parent / "frontend"
app.mount("/static", StaticFiles(directory=str(frontend_dir)), name="static")

# ── Incoming image storage ────────────────────────────────────────────────────
# Every image sent to /predict is persisted here for audit and retraining.
# Filenames are Unix timestamps (e.g. 1712345678.jpg).
INCOMING_DIR = Path(__file__).parent.parent / "incoming_data"
INCOMING_DIR.mkdir(parents=True, exist_ok=True)


# ── Routes ────────────────────────────────────────────────────────────────────
@app.get("/", include_in_schema=False)
async def root():
    return FileResponse(str(frontend_dir / "index.html"))


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "model_loaded": predictor.is_ready(),
        "version": "2.0.0",
    }


@app.post("/predict")
async def predict(
    file: UploadFile = File(None),
    image_base64: str = Form(None),
):
    """
    Accept either a multipart image file OR a base64-encoded image string.

    Returns:
      {
        "status":     "GOOD" | "DEFECT",
        "detections": [{"class": str, "confidence": float, "bbox": [x1,y1,x2,y2]}],
        "image":      <base64-encoded annotated JPEG>,
        "model":      "fine-tuned" | "base-yolov8n"
      }

    The incoming image is also saved to incoming_data/<timestamp>.jpg for
    audit and future retraining purposes.
    """
    if file is None and image_base64 is None:
        raise HTTPException(status_code=400, detail="Provide 'file' or 'image_base64'.")

    tmp_path = None
    try:
        # ── Decode & validate ─────────────────────────────────────────────────
        if file is not None:
            img_bytes = await file.read()
            if not validate_image(img_bytes):
                raise HTTPException(status_code=400, detail="Invalid image format.")
        else:
            img_bytes = decode_base64_image(image_base64)
            if not validate_image(img_bytes):
                raise HTTPException(status_code=400, detail="Invalid base64 image.")

        # ── Write to temp file for YOLO inference ─────────────────────────────
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
            tmp_path = tmp.name
            tmp.write(img_bytes)

        # ── Persist incoming image ────────────────────────────────────────────
        save_path = INCOMING_DIR / f"{int(time.time())}.jpg"
        save_path.write_bytes(img_bytes)

        result = predictor.predict(tmp_path)
        return JSONResponse(content=result)

    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.remove(tmp_path)


# ── Entry Point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    uvicorn.run("app.main:app", host="0.0.0.0", port=8080, reload=False)

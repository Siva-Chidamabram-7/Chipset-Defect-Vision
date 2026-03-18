import os
import uuid
import base64
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
    description="YOLOv8-powered solder defect detection for PCB inspection",
    version="1.0.0",
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


# ── Routes ────────────────────────────────────────────────────────────────────
@app.get("/", include_in_schema=False)
async def root():
    return FileResponse(str(frontend_dir / "index.html"))


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "model_loaded": predictor.is_ready(),
        "version": "1.0.0",
    }


@app.post("/predict")
async def predict(
    file: UploadFile = File(None),
    image_base64: str = Form(None),
):
    """
    Accept either a multipart image file OR a base64-encoded image string.
    Returns JSON detections + base64-encoded annotated image.
    """
    if file is None and image_base64 is None:
        raise HTTPException(status_code=400, detail="Provide 'file' or 'image_base64'.")

    tmp_path = None
    try:
        suffix = ".jpg"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp_path = tmp.name
            if file is not None:
                content = await file.read()
                if not validate_image(content):
                    raise HTTPException(status_code=400, detail="Invalid image format.")
                tmp.write(content)
            else:
                img_bytes = decode_base64_image(image_base64)
                if not validate_image(img_bytes):
                    raise HTTPException(status_code=400, detail="Invalid base64 image.")
                tmp.write(img_bytes)

        result = predictor.predict(tmp_path)
        return JSONResponse(content=result)

    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.remove(tmp_path)


# ── Entry Point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    uvicorn.run("app.main:app", host="0.0.0.0", port=8080, reload=False)

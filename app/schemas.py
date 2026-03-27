"""
app/schemas.py — Pydantic response models for the Chipset Defect Vision API.

These schemas define the exact JSON shapes returned by:
  POST /predict  →  PredictionResponse
  GET  /health   →  HealthResponse

They are also used by FastAPI to auto-generate the OpenAPI (Swagger) docs
at /docs, so field names here must match what predictor.py returns.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


# ── Per-detection model ───────────────────────────────────────────────────────

class DetectionResult(BaseModel):
    """A single bounding-box detection from YOLO.

    The field alias "class" mirrors the raw model output key while
    `label` is a duplicate kept for front-end convenience.  Both hold
    the same string (e.g. "solder_bridge").
    """
    class_name: str  = Field(alias="class")  # YOLO class name; aliased from "class" key
    label:      str                           # Copy of class_name — used directly by frontend
    confidence: float                         # Post-NMS confidence score, range [0, 1]
    bbox:       list[int]                     # Pixel coords [x1, y1, x2, y2] in the input image

    model_config = {"populate_by_name": True}  # allow populating by Python name OR alias


# ── Aggregated summary counts ─────────────────────────────────────────────────

class PredictionSummary(BaseModel):
    """High-level counts derived from the detection list.

    good_count is always 0 because the "good" class was removed from the
    model — every detection is a defect.  The field is kept for schema
    compatibility with older frontend versions.
    """
    total:        int   # Total accepted detections (== defect_count)
    good_count:   int   # Always 0 — "good" class not present in current model
    defect_count: int   # Number of defect bounding boxes above confidence threshold
    has_defects:  bool  # Convenience flag: True when defect_count > 0


# ── Timing breakdown ──────────────────────────────────────────────────────────

class PredictionTimings(BaseModel):
    """Wall-clock timing of the inference stage, in milliseconds."""
    inference_ms: float  # Time from predictor.predict() call to return, in ms


# ── Full /predict response ────────────────────────────────────────────────────

class PredictionResponse(BaseModel):
    """Complete JSON body returned by POST /predict.

    Both `annotated_image_base64` and `image` carry the same base64 JPEG
    of the annotated result; the duplication preserves backward compatibility
    with earlier frontend builds that expected the key `image`.
    """
    status:                 str                  # "GOOD" or "DEFECT"
    model:                  str                  # Filename of the weights used (e.g. "best.pt")
    detections:             list[DetectionResult]# One entry per accepted bounding box
    summary:                PredictionSummary    # Aggregated counts
    total:                  int                  # Top-level alias of summary.total
    good_count:             int                  # Top-level alias of summary.good_count (always 0)
    defect_count:           int                  # Top-level alias of summary.defect_count
    annotated_image_base64: str                  # Base64 JPEG with drawn bounding boxes
    image:                  str                  # Same as annotated_image_base64 (legacy key)
    timings:                PredictionTimings    # Inference timing breakdown


# ── /health response ──────────────────────────────────────────────────────────

class HealthResponse(BaseModel):
    """JSON body returned by GET /health.

    The frontend polls this every 30 s to update the connection status dot
    and display the active model filename in the footer.
    """
    status:       str   # Always "ok" when the service is running
    model_loaded: bool  # True if SolderDefectPredictor loaded weights successfully
    model:        str   # Model filename (e.g. "best.pt")
    version:      str   # APP_VERSION string from app/config.py

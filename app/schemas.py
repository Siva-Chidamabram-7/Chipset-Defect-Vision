from __future__ import annotations

from pydantic import BaseModel, Field


class DetectionResult(BaseModel):
    class_name: str = Field(alias="class")
    label: str
    confidence: float
    bbox: list[int]

    model_config = {"populate_by_name": True}


class PredictionSummary(BaseModel):
    total: int
    good_count: int
    defect_count: int
    has_defects: bool


class PredictionTimings(BaseModel):
    inference_ms: float


class PredictionResponse(BaseModel):
    status: str
    model: str
    detections: list[DetectionResult]
    summary: PredictionSummary
    total: int
    good_count: int
    defect_count: int
    annotated_image_base64: str
    image: str
    timings: PredictionTimings


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model: str
    version: str

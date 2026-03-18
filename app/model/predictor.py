"""
SolderDefectPredictor
─────────────────────
Wraps YOLOv8 inference and annotation.

Model priority:
  1. weights/best.pt   ← fine-tuned on PCB solder dataset
  2. weights/yolov8n.pt ← downloaded automatically if neither exists

Class mapping (fine-tuned model):
  0 → good
  1 → defect

When running on the base COCO model (no fine-tuned weights), we map
all detections to a demo label so the UI still works end-to-end.
"""

import base64
import os
from pathlib import Path
from typing import Any

import cv2
import numpy as np

# Lazy import – lets the container start even before torch is fully cached
try:
    from ultralytics import YOLO
    _YOLO_AVAILABLE = True
except ImportError:
    _YOLO_AVAILABLE = False

# ── Constants ─────────────────────────────────────────────────────────────────
WEIGHTS_DIR = Path(__file__).parent.parent.parent / "weights"
FINE_TUNED   = WEIGHTS_DIR / "best.pt"
BASE_MODEL   = WEIGHTS_DIR / "yolov8n.pt"

CLASS_NAMES  = {0: "good", 1: "defect"}
COLORS       = {
    "good":   (34, 197, 94),   # green
    "defect": (239, 68, 68),   # red
    "unknown":(234, 179, 8),   # yellow fallback
}
CONF_THRESHOLD = 0.25


# ── Predictor ─────────────────────────────────────────────────────────────────
class SolderDefectPredictor:
    def __init__(self):
        self._model = None
        self._fine_tuned = False
        self._load_model()

    # ── Model loading ─────────────────────────────────────────────────────────
    def _load_model(self):
        if not _YOLO_AVAILABLE:
            print("[Predictor] ultralytics not available – running in stub mode.")
            return

        WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)

        if FINE_TUNED.exists():
            print(f"[Predictor] Loading fine-tuned weights: {FINE_TUNED}")
            self._model = YOLO(str(FINE_TUNED))
            self._fine_tuned = True
        else:
            print(f"[Predictor] Fine-tuned weights not found. Loading base YOLOv8n.")
            self._model = YOLO("yolov8n.pt")           # auto-downloads
            self._fine_tuned = False

    def is_ready(self) -> bool:
        return self._model is not None or not _YOLO_AVAILABLE

    # ── Inference ─────────────────────────────────────────────────────────────
    def predict(self, image_path: str) -> dict[str, Any]:
        img_bgr = cv2.imread(image_path)
        if img_bgr is None:
            raise ValueError(f"Cannot read image: {image_path}")

        if not _YOLO_AVAILABLE or self._model is None:
            return self._stub_response(img_bgr)

        results = self._model.predict(
            source=image_path,
            conf=CONF_THRESHOLD,
            save=False,
            verbose=False,
        )[0]

        detections = []
        annotated  = img_bgr.copy()

        if results.boxes is not None:
            for box in results.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                conf  = float(box.conf[0])
                cls   = int(box.cls[0])

                if self._fine_tuned:
                    label = CLASS_NAMES.get(cls, "unknown")
                else:
                    # Base COCO model: map all high-conf detections to demo labels
                    label = "defect" if conf < 0.6 else "good"

                color = COLORS.get(label, COLORS["unknown"])
                self._draw_box(annotated, x1, y1, x2, y2, label, conf, color)

                detections.append({
                    "label":      label,
                    "confidence": round(conf, 4),
                    "bbox":       [x1, y1, x2, y2],
                })

        annotated_b64 = self._encode_image(annotated)
        return {
            "detections":   detections,
            "total":        len(detections),
            "defect_count": sum(1 for d in detections if d["label"] == "defect"),
            "good_count":   sum(1 for d in detections if d["label"] == "good"),
            "image":        annotated_b64,
            "model":        "fine-tuned" if self._fine_tuned else "base-yolov8n",
        }

    # ── Helpers ───────────────────────────────────────────────────────────────
    @staticmethod
    def _draw_box(img, x1, y1, x2, y2, label, conf, color):
        thickness = 2
        cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)

        text  = f"{label} {conf:.2f}"
        font  = cv2.FONT_HERSHEY_SIMPLEX
        scale = 0.55
        (tw, th), baseline = cv2.getTextSize(text, font, scale, 1)

        # label background pill
        pad = 4
        cv2.rectangle(
            img,
            (x1, y1 - th - baseline - pad * 2),
            (x1 + tw + pad * 2, y1),
            color, -1,
        )
        cv2.putText(
            img, text,
            (x1 + pad, y1 - baseline - pad),
            font, scale, (255, 255, 255), 1, cv2.LINE_AA,
        )

    @staticmethod
    def _encode_image(img: np.ndarray) -> str:
        _, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 90])
        return base64.b64encode(buf).decode("utf-8")

    def _stub_response(self, img: np.ndarray) -> dict:
        """Fallback when ultralytics is not installed (CI / lightweight env)."""
        h, w = img.shape[:2]
        stub_img = img.copy()
        cv2.putText(
            stub_img, "Model not loaded – stub mode",
            (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2,
        )
        return {
            "detections":   [],
            "total":        0,
            "defect_count": 0,
            "good_count":   0,
            "image":        self._encode_image(stub_img),
            "model":        "stub",
        }

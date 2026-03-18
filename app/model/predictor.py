"""
SolderDefectPredictor
─────────────────────
Wraps YOLOv8 inference and annotation.

⚠  OFFLINE-FIRST: this module NEVER attempts a network call.
   All weights must be present on disk before the server starts.

Model priority (checked in order):
  1. weights/best.pt    ← fine-tuned on PCB solder dataset (user-supplied)
  2. weights/yolov8n.pt ← base weights baked into the Docker image at build time

If neither file exists the predictor enters STUB mode: the server still starts
and every prediction returns an annotated image with a visible warning overlay.
This makes misconfiguration obvious without crashing the service.

Class mapping (fine-tuned model):
  0 → good
  1 → defect

When running on the base model (no fine-tuned weights) COCO class IDs are
remapped to demo labels so the UI still works end-to-end.
"""

import base64
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np

# ── ultralytics import ────────────────────────────────────────────────────────
# Lazy — lets the container start even if torch is still loading.
try:
    from ultralytics import YOLO
    _YOLO_AVAILABLE = True
except ImportError:
    _YOLO_AVAILABLE = False

# ── Weight paths ──────────────────────────────────────────────────────────────
# Both paths are LOCAL.  No network fallback exists — by design.
WEIGHTS_DIR  = Path(__file__).parent.parent.parent / "weights"
FINE_TUNED   = WEIGHTS_DIR / "best.pt"      # priority 1
BASE_WEIGHTS = WEIGHTS_DIR / "yolov8n.pt"   # priority 2 — baked in at build time

# ── Class config ─────────────────────────────────────────────────────────────
CLASS_NAMES = {0: "good", 1: "defect"}
COLORS = {
    "good":    (34, 197, 94),    # green
    "defect":  (239, 68, 68),    # red
    "unknown": (234, 179, 8),    # yellow — used for base-model COCO classes
}
CONF_THRESHOLD = 0.25


# ── Predictor ─────────────────────────────────────────────────────────────────
class SolderDefectPredictor:
    def __init__(self):
        self._model      = None
        self._fine_tuned = False
        self._load_model()

    # ── Model loading ─────────────────────────────────────────────────────────
    def _load_model(self):
        """
        Load weights strictly from local disk.
        Raises no exceptions — failures are logged and the predictor enters
        stub mode so the FastAPI server can still report /health correctly.
        """
        if not _YOLO_AVAILABLE:
            print(
                "[Predictor] WARNING: ultralytics is not installed.\n"
                "            Running in stub mode (no inference).",
                file=sys.stderr,
            )
            return

        WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)

        if FINE_TUNED.exists():
            # ── Priority 1: user-trained fine-tuned model ─────────────────────
            print(f"[Predictor] Loading fine-tuned weights: {FINE_TUNED}")
            self._model      = YOLO(str(FINE_TUNED))
            self._fine_tuned = True

        elif BASE_WEIGHTS.exists():
            # ── Priority 2: base YOLOv8n baked into the Docker image ──────────
            print(f"[Predictor] Loading base weights: {BASE_WEIGHTS}")
            self._model      = YOLO(str(BASE_WEIGHTS))
            self._fine_tuned = False

        else:
            # ── No weights found — OFFLINE hard stop ──────────────────────────
            # Do NOT attempt YOLO("yolov8n.pt") — that triggers a network
            # download which is forbidden in an air-gapped environment.
            print(
                "[Predictor] ERROR: No weight files found.\n"
                f"            Expected one of:\n"
                f"              • {FINE_TUNED}\n"
                f"              • {BASE_WEIGHTS}\n"
                "            Rebuild the Docker image (which bakes in yolov8n.pt)\n"
                "            or mount weights/best.pt via -v flag.\n"
                "            Running in STUB mode — inference disabled.",
                file=sys.stderr,
            )
            # _model stays None → is_ready() returns False → stub responses

    def is_ready(self) -> bool:
        """True when a model is loaded and can run inference."""
        return self._model is not None

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
                conf = float(box.conf[0])
                cls  = int(box.cls[0])

                if self._fine_tuned:
                    label = CLASS_NAMES.get(cls, "unknown")
                else:
                    # Base COCO model: remap all detections to demo labels
                    label = "defect" if conf < 0.6 else "good"

                color = COLORS.get(label, COLORS["unknown"])
                self._draw_box(annotated, x1, y1, x2, y2, label, conf, color)
                detections.append({
                    "label":      label,
                    "confidence": round(conf, 4),
                    "bbox":       [x1, y1, x2, y2],
                })

        return {
            "detections":   detections,
            "total":        len(detections),
            "defect_count": sum(1 for d in detections if d["label"] == "defect"),
            "good_count":   sum(1 for d in detections if d["label"] == "good"),
            "image":        self._encode_image(annotated),
            "model":        "fine-tuned" if self._fine_tuned else "base-yolov8n",
        }

    # ── Helpers ───────────────────────────────────────────────────────────────
    @staticmethod
    def _draw_box(img, x1, y1, x2, y2, label, conf, color):
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

        text  = f"{label} {conf:.2f}"
        font  = cv2.FONT_HERSHEY_SIMPLEX
        scale = 0.55
        (tw, th), baseline = cv2.getTextSize(text, font, scale, 1)

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
        """
        Returned when no model is loaded.
        Overlays a visible warning on the image so the problem is unmissable
        in the UI — much better than a silent empty-detections response.
        """
        stub = img.copy()
        # Dark semi-transparent banner
        overlay = stub.copy()
        cv2.rectangle(overlay, (0, 0), (stub.shape[1], 60), (30, 30, 30), -1)
        cv2.addWeighted(overlay, 0.6, stub, 0.4, 0, stub)
        cv2.putText(
            stub, "NO MODEL LOADED — check weights/ directory",
            (10, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 80, 255), 2, cv2.LINE_AA,
        )
        return {
            "detections":   [],
            "total":        0,
            "defect_count": 0,
            "good_count":   0,
            "image":        self._encode_image(stub),
            "model":        "stub — no weights found",
        }

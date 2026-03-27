"""
app/model/predictor.py — YOLOv8 inference wrapper for PCB solder defect detection.

This module is the single point of contact between the FastAPI layer and the
Ultralytics YOLO library.  It owns:
  • Model loading (from weights/best.pt — trained by training/train.py)
  • Inference (with a low internal threshold + strict post-filter)
  • Bounding-box drawing (OpenCV rectangle + label text)
  • Result dict construction (returned to app/main.py /predict route)

Class names and colours are loaded from training/data.yaml at import time so
that predictor.py stays in sync with the training configuration automatically.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np
import yaml

import logging

from app.config import CONFIDENCE_THRESHOLD, MODEL_NAME, MODEL_PATH, NMS_IOU_THRESHOLD
from app.utils.image_utils import encode_image_to_base64

logger = logging.getLogger("chipset_defect_vision.predictor")

# ── Optional YOLO import ──────────────────────────────────────────────────────
# Wrapped in try/except so that importing this module doesn't crash if
# ultralytics is missing — the error is deferred to _load_model() which
# raises a clear RuntimeError with installation instructions.
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False

# ── Class metadata ────────────────────────────────────────────────────────────
# Point to training/data.yaml as the single source of truth for class names.
# This means adding a new defect class only requires updating data.yaml;
# predictor.py picks it up automatically on the next server start.
_DATA_YAML = Path(__file__).resolve().parents[2] / "training" / "data.yaml"

# BGR colour palette for bounding-box drawing — one entry per class index.
# Colours are chosen for visual contrast on typical PCB images.
_CLASS_COLORS = [
<<<<<<< HEAD
    (68, 68, 239),    # 0 missing_hole        — blue/purple
    (8, 179, 234),    # 1 mouse_bite          — cyan
    (246, 130, 59),   # 2 open_circuit        — orange
    (247, 85, 168),   # 3 short               — pink
    (22, 115, 249),   # 4 spur                — blue
    (166, 184, 20),   # 5 spurious_copper     — yellow-green
    (0, 204, 102),    # 6 solder_bridge       — green
    (255, 191, 0),    # 7 excess_solder       — amber
    (204, 51, 51),    # 8 insufficient_solder — red
    (128, 128, 128),  # 9 good                — grey (currently unused by model)
=======
    (68, 68, 239),    # missing_hole     — blue/purple
    (8, 179, 234),    # mouse_bite       — cyan
    (246, 130, 59),   # open_circuit     — orange
    (247, 85, 168),   # short            — pink
    (22, 115, 249),   # spur             — blue
    (166, 184, 20),   # spurious_copper  — yellow-green
    (50, 205, 50),    # good             — green
    (220, 50, 50),    # solder_defect    — red
>>>>>>> b3b38bdc8568e3830d194c147d70e12a2d46a9e2
]


def _load_class_names() -> list[str]:
    """Read class names from training/data.yaml in index order.

    Handles both list and dict YAML formats:
      - list  → ["missing_hole", "mouse_bite", ...]
      - dict  → {0: "missing_hole", 1: "mouse_bite", ...}
    """
    with open(_DATA_YAML) as f:
        data = yaml.safe_load(f)
    names = data.get("names", [])
    if isinstance(names, dict):
        return [names[i] for i in sorted(names)]
    return list(names)


# Module-level constants built once at import time
CLASS_NAMES: list[str] = _load_class_names()
# Map class name → BGR colour for O(1) lookup during box drawing
COLORS: dict[str, tuple[int, int, int]] = {
    name: _CLASS_COLORS[i % len(_CLASS_COLORS)]
    for i, name in enumerate(CLASS_NAMES)
}


class SolderDefectPredictor:
    """Offline-only YOLO predictor that hard-fails unless weights/best.pt exists.

    Design decisions:
    • Offline-only: the model file must be present locally.  There is no
      automatic download fallback — this is intentional for air-gapped
      factory deployments.
    • Two-threshold strategy: YOLO runs internally at 0.10 confidence so
      we can log the full box distribution for debugging, then we apply
      the stricter CONFIDENCE_THRESHOLD (0.50) in post-processing.
    • The predictor instance is created once at server startup (lifespan
      hook in app/main.py) and shared across all request handlers via
      request.app.state.predictor.
    """

    def __init__(
        self,
        model_path: Path | None = None,
        confidence_threshold: float = CONFIDENCE_THRESHOLD,  # from app/config.py or env var
        iou_threshold: float = NMS_IOU_THRESHOLD,            # from app/config.py or env var
    ) -> None:
        self._model_path           = model_path or MODEL_PATH
        self._confidence_threshold = confidence_threshold
        self._iou_threshold        = iou_threshold
        self._model                = None   # populated by _load_model(); None = not ready
        self._load_model()

    def _load_model(self) -> None:
        """Load YOLO weights into memory.  Raises RuntimeError on failure.

        Called once during __init__.  A RuntimeError here propagates to the
        lifespan hook and prevents the server from starting, which is the
        desired behaviour — we never want to silently serve an empty model.
        """
        if not YOLO_AVAILABLE:
            raise RuntimeError(
                "[Predictor] FATAL: ultralytics is not installed. "
                "Inference cannot start without it."
            )

        if not self._model_path.exists():
            raise RuntimeError(
                "[Predictor] FATAL: trained model not found.\n"
                f"Expected: {self._model_path}\n"
                "The inference service only supports weights/best.pt."
            )

        # YOLO() loads the .pt file into RAM (and GPU VRAM if available)
        self._model = YOLO(str(self._model_path))

    def is_ready(self) -> bool:
        """Return True when the model has been loaded successfully.
        Used by the /health endpoint to report model status.
        """
        return self._model is not None

    def predict(self, image_bgr: np.ndarray) -> dict[str, Any]:
        """Run YOLOv8 inference and return a complete prediction dictionary.

        Two-pass confidence strategy:
          1. YOLO internally filters at _RAW_CONF=0.10 — low enough to keep
             all candidate boxes so we can log the score distribution.
          2. We apply the stricter self._confidence_threshold (default 0.50)
             in a post-filter loop before building the response.

        Arguments:
            image_bgr  BGR ndarray as returned by decode_image_bytes().

        Returns a dict matching PredictionResponse schema in app/schemas.py.
        """
        # ── Step 1: raw YOLO inference at a permissive threshold ─────────────
        # _RAW_CONF is intentionally low — we want to see everything the model
        # detects before we apply the stricter business-logic threshold.
        _RAW_CONF = 0.10
        results = self._model.predict(
            source=image_bgr,
            conf=_RAW_CONF,           # internal NMS pre-filter
            iou=self._iou_threshold,  # NMS IoU threshold (from config / env var)
            save=False,               # don't write files; we handle persistence ourselves
            verbose=False,            # suppress per-image Ultralytics progress lines
        )[0]  # [0] extracts the single-image result from the batch list

        detections: list[dict[str, Any]] = []
        annotated  = image_bgr.copy()   # we draw on this copy, not the original

        raw_count      = len(results.boxes) if results.boxes is not None else 0
        filtered_count = 0              # boxes below our strict threshold

        if results.boxes is not None and raw_count > 0:
            # Log confidence distribution to help diagnose threshold tuning
            all_scores = sorted(
                [round(float(b.conf[0]), 3) for b in results.boxes], reverse=True
            )
            logger.info(
                "[predict] model produced %d raw boxes (internal conf>=%.2f, iou<=%.2f)",
                raw_count, _RAW_CONF, self._iou_threshold,
            )
            logger.info(
                "[predict] confidence distribution: min=%.3f max=%.3f samples=%s",
                all_scores[-1], all_scores[0],
                all_scores[:10] if len(all_scores) > 10 else all_scores,
            )

            # ── Step 2: post-filter at the stricter business threshold ────────
            for box in results.boxes:
                confidence = round(float(box.conf[0]), 4)

                if confidence < self._confidence_threshold:
                    # Below threshold — count it for logging but don't draw or include
                    filtered_count += 1
                    continue

                # Extract pixel-coordinate bounding box from YOLO result
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                class_id = int(box.cls[0])
                label    = CLASS_NAMES[class_id] if class_id < len(CLASS_NAMES) else "unknown"
                color    = COLORS.get(label, (120, 120, 120))  # fallback grey

                # Draw bounding box + label text onto the annotated copy
                self._draw_box(annotated, x1, y1, x2, y2, label, confidence, color)

                # Build detection dict matching DetectionResult schema
                detections.append(
                    {
                        "class": label,        # aliased field name used by schema
                        "label": label,        # duplicate for frontend convenience
                        "confidence": confidence,
                        "bbox": [x1, y1, x2, y2],
                    }
                )

        # ── Step 3: build response ────────────────────────────────────────────
        defect_count = len(detections)
        status       = "DEFECT" if defect_count > 0 else "GOOD"

        logger.info(
            "[predict] result: %d kept / %d filtered / %d raw | threshold=%.2f | status=%s",
            defect_count, filtered_count, raw_count,
            self._confidence_threshold, status,
        )

        # Encode annotated BGR image to base64 JPEG for the JSON response
        encoded_image = encode_image_to_base64(annotated)

        # Summary block mirrors PredictionSummary schema fields
        summary = {
<<<<<<< HEAD
            "total":        defect_count,
            "good_count":   0,            # "good" class removed from model; always 0
=======
            "total": defect_count,
            "good_count": 0,          # Good class removed; always 0
>>>>>>> b3b38bdc8568e3830d194c147d70e12a2d46a9e2
            "defect_count": defect_count,
            "has_defects":  defect_count > 0,
        }

        # Top-level keys are duplicated from summary for backward compatibility
        # with frontend versions that read them directly instead of from summary.
        return {
<<<<<<< HEAD
            "status":                 status,
            "model":                  MODEL_NAME,
            "detections":             detections,
            "summary":                summary,
            "total":                  defect_count,
            "good_count":             0,
            "defect_count":           defect_count,
            "annotated_image_base64": encoded_image,  # primary key used by frontend
            "image":                  encoded_image,  # legacy duplicate key
=======
            "status": status,
            "model": MODEL_NAME,
            "detections": detections,
            "summary": summary,
            "total": defect_count,
            "good_count": 0,
            "defect_count": defect_count,
            "annotated_image_base64": encoded_image,
            "image": encoded_image,
>>>>>>> b3b38bdc8568e3830d194c147d70e12a2d46a9e2
        }

    @staticmethod
    def _draw_box(
        img: np.ndarray,
        x1: int,
        y1: int,
        x2: int,
        y2: int,
        label: str,
        confidence: float,
        color: tuple[int, int, int],  # BGR tuple matching _CLASS_COLORS
    ) -> None:
        """Draw a bounding box and label badge onto img in-place.

        Layout:
          ┌──────────────────────────┐  ← filled label background rectangle
          │ label_name  0.92         │
          ┗━━━━━━━━━━━━━━━━━━━━━━━━━━┓  ← bounding box rectangle (2 px border)
          ┃                          ┃
          ┃    defect region         ┃
          ┗━━━━━━━━━━━━━━━━━━━━━━━━━━┛

        White text on a coloured filled rectangle is used for the label so it
        remains readable regardless of the PCB background colour.
        """
        # ── Bounding box rectangle ────────────────────────────────────────────
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

        # ── Label text + background ───────────────────────────────────────────
        text  = f"{label} {confidence:.2f}"
        font  = cv2.FONT_HERSHEY_SIMPLEX
        scale = 0.55
        (text_width, text_height), baseline = cv2.getTextSize(text, font, scale, 1)

        padding = 4  # pixels of padding around text inside the label background
        # Filled rectangle drawn immediately above the bounding box top edge
        cv2.rectangle(
            img,
            (x1, y1 - text_height - baseline - padding * 2),
            (x1 + text_width + padding * 2, y1),
            color,
            -1,  # -1 = filled (not outline)
        )
        # White text drawn on top of the filled rectangle
        cv2.putText(
            img,
            text,
            (x1 + padding, y1 - baseline - padding),
            font,
            scale,
            (255, 255, 255),  # white text — readable on any box colour
            1,
            cv2.LINE_AA,      # anti-aliased edges
        )

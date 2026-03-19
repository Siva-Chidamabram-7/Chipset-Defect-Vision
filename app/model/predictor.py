"""
SolderDefectPredictor
─────────────────────
Wraps YOLOv8 inference for PCB solder defect detection.

⚠  OFFLINE-FIRST: this module NEVER attempts a network call.
   All weights must be present on disk before the server starts.

Required weight file:
  weights/best.pt  ← fine-tuned on the 7-class PCB defect dataset (MANDATORY)

If weights/best.pt is missing the server will REFUSE TO START with a clear
RuntimeError.  There is no silent fallback — every deployment must ship the
correct model.

Defect classes:
  0 → Missing_hole
  1 → Mouse_bite
  2 → Open_circuit
  3 → Short
  4 → Spur
  5 → Spurious_copper
  6 → Good

Decision logic:
  • No detections above CONF_THRESHOLD → status = "GOOD"
  • Any detection above CONF_THRESHOLD → status = "DEFECT"
"""

import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np

# ── ultralytics import ────────────────────────────────────────────────────────
try:
    from ultralytics import YOLO
    _YOLO_AVAILABLE = True
except ImportError:
    _YOLO_AVAILABLE = False

# ── Weight paths ──────────────────────────────────────────────────────────────
# LOCAL only — no network fallback exists, by design.
WEIGHTS_DIR = Path(__file__).parent.parent.parent / "weights"
BEST_PT     = WEIGHTS_DIR / "best.pt"   # the ONE required model file

# ── Class config ──────────────────────────────────────────────────────────────
# Must match nc + names defined in !training/data.yaml exactly.
CLASS_NAMES = {
    0: "Missing_hole",
    1: "Mouse_bite",
    2: "Open_circuit",
    3: "Short",
    4: "Spur",
    5: "Spurious_copper",
    6: "Good",
}

# BGR colors — one distinct color per class
COLORS = {
    "Missing_hole":    ( 68,  68, 239),   # red
    "Mouse_bite":      (  8, 179, 234),   # yellow
    "Open_circuit":    (246, 130,  59),   # blue
    "Short":           (247,  85, 168),   # purple
    "Spur":            ( 22, 115, 249),   # orange
    "Spurious_copper": (166, 184,  20),   # teal
    "Good":            ( 50, 205,  50),   # green
}

# Detections below this confidence are ignored entirely.
CONF_THRESHOLD = 0.30


# ── Predictor ─────────────────────────────────────────────────────────────────
class SolderDefectPredictor:
    def __init__(self):
        self._model = None
        self._load_model()

    # ── Model loading ─────────────────────────────────────────────────────────
    def _load_model(self):
        """
        Load weights/best.pt from local disk.

        Raises RuntimeError if:
          - ultralytics is not installed
          - weights/best.pt does not exist

        There is NO silent fallback.  A missing model is always a hard error.
        """
        if not _YOLO_AVAILABLE:
            raise RuntimeError(
                "[Predictor] FATAL: ultralytics is not installed.\n"
                "            Cannot run inference without the ultralytics package."
            )

        if not BEST_PT.exists():
            raise RuntimeError(
                f"[Predictor] FATAL: trained model not found.\n"
                f"            Expected: {BEST_PT}\n"
                f"\n"
                f"            To fix:\n"
                f"              • Run training and copy best.pt → weights/best.pt, then\n"
                f"                rebuild the Docker image:  docker build -t chipset-defect-vision .\n"
                f"              • Or mount the file at runtime:\n"
                f"                docker run -v $(pwd)/weights/best.pt:/app/weights/best.pt ...\n"
            )

        print(f"[Predictor] Loading trained model: {BEST_PT}", flush=True)
        self._model = YOLO(str(BEST_PT))
        print(f"[Predictor] Model ready — classes: {list(CLASS_NAMES.values())}", flush=True)

    def is_ready(self) -> bool:
        """True when the model is loaded and can run inference."""
        return self._model is not None

    # ── Inference ─────────────────────────────────────────────────────────────
    def predict(self, image_path: str) -> dict[str, Any]:
        """
        Run YOLOv8 inference and return a structured result.

        Response schema:
          {
            "status":     "GOOD" | "DEFECT",
            "detections": [{"class": str, "confidence": float, "bbox": [x1,y1,x2,y2]}],
            "image":      <base64-encoded annotated JPEG>,
            "model":      "best.pt"
          }

        Decision rule:
          • No detections → "GOOD"
          • Any detection  → "DEFECT"
        """
        img_bgr = cv2.imread(image_path)
        if img_bgr is None:
            raise ValueError(f"Cannot read image: {image_path}")

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
                conf       = float(box.conf[0])
                cls        = int(box.cls[0])
                class_name = CLASS_NAMES.get(cls, "unknown")
                color      = COLORS.get(class_name, (120, 120, 120))
                self._draw_box(annotated, x1, y1, x2, y2, class_name, conf, color)
                detections.append({
                    "class":      class_name,
                    "confidence": round(conf, 4),
                    "bbox":       [x1, y1, x2, y2],
                })

        # ── Binary decision ───────────────────────────────────────────────────
        status = "GOOD" if not detections else "DEFECT"

        return {
            "status":     status,
            "detections": detections,
            "image":      self._encode_image(annotated),
            "model":      "best.pt",
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
        import base64
        _, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 90])
        return base64.b64encode(buf).decode("utf-8")

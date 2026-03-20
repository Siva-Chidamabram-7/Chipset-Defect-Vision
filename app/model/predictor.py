from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np

from app.config import CONFIDENCE_THRESHOLD, MODEL_NAME, MODEL_PATH
from app.utils.image_utils import encode_image_to_base64

try:
    from ultralytics import YOLO

    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False


CLASS_NAMES = {
    0: "Missing_hole",
    1: "Mouse_bite",
    2: "Open_circuit",
    3: "Short",
    4: "Spur",
    5: "Spurious_copper",
    6: "Good",
}

COLORS = {
    "Missing_hole": (68, 68, 239),
    "Mouse_bite": (8, 179, 234),
    "Open_circuit": (246, 130, 59),
    "Short": (247, 85, 168),
    "Spur": (22, 115, 249),
    "Spurious_copper": (166, 184, 20),
    "Good": (50, 205, 50),
}


class SolderDefectPredictor:
    """Offline-only YOLO predictor that hard-fails unless weights/best.pt exists."""

    def __init__(
        self,
        model_path: Path | None = None,
        confidence_threshold: float = CONFIDENCE_THRESHOLD,
    ) -> None:
        self._model_path = model_path or MODEL_PATH
        self._confidence_threshold = confidence_threshold
        self._model = None
        self._load_model()

    def _load_model(self) -> None:
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

        self._model = YOLO(str(self._model_path))

    def is_ready(self) -> bool:
        return self._model is not None

    def predict(self, image_bgr: np.ndarray) -> dict[str, Any]:
        results = self._model.predict(
            source=image_bgr,
            conf=self._confidence_threshold,
            save=False,
            verbose=False,
        )[0]

        detections: list[dict[str, Any]] = []
        annotated = image_bgr.copy()

        if results.boxes is not None:
            for box in results.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                confidence = round(float(box.conf[0]), 4)
                class_id = int(box.cls[0])
                label = CLASS_NAMES.get(class_id, "unknown")
                color = COLORS.get(label, (120, 120, 120))

                self._draw_box(annotated, x1, y1, x2, y2, label, confidence, color)
                detections.append(
                    {
                        "class": label,
                        "label": label,
                        "confidence": confidence,
                        "bbox": [x1, y1, x2, y2],
                    }
                )

        defect_count = sum(1 for item in detections if item["label"] != "Good")
        good_count = sum(1 for item in detections if item["label"] == "Good")
        total = len(detections)
        status = "DEFECT" if defect_count > 0 else "GOOD"
        encoded_image = encode_image_to_base64(annotated)

        summary = {
            "total": total,
            "good_count": good_count,
            "defect_count": defect_count,
            "has_defects": defect_count > 0,
        }

        return {
            "status": status,
            "model": MODEL_NAME,
            "detections": detections,
            "summary": summary,
            "total": total,
            "good_count": good_count,
            "defect_count": defect_count,
            "annotated_image_base64": encoded_image,
            "image": encoded_image,
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
        color: tuple[int, int, int],
    ) -> None:
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

        text = f"{label} {confidence:.2f}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = 0.55
        (text_width, text_height), baseline = cv2.getTextSize(text, font, scale, 1)

        padding = 4
        cv2.rectangle(
            img,
            (x1, y1 - text_height - baseline - padding * 2),
            (x1 + text_width + padding * 2, y1),
            color,
            -1,
        )
        cv2.putText(
            img,
            text,
            (x1 + padding, y1 - baseline - padding),
            font,
            scale,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

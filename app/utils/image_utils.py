from __future__ import annotations

import base64
import binascii
from typing import Final

import cv2
import numpy as np

IMAGE_MAGIC_BYTES: Final[list[bytes]] = [
    b"\xff\xd8\xff",          # JPEG
    b"\x89PNG\r\n\x1a\n",     # PNG
    b"RIFF",                  # WebP container
    b"GIF87a",                # GIF
    b"GIF89a",                # GIF
]


def validate_image(data: bytes) -> bool:
    """Return True when bytes look like a supported image file."""
    if not data:
        return False

    for magic in IMAGE_MAGIC_BYTES:
        if data[: len(magic)] == magic:
            return True

    return data[:4] == b"RIFF" and data[8:12] == b"WEBP"


def decode_base64_image(b64_string: str) -> bytes:
    """Decode a base64 image string, removing any data URL prefix."""
    if not b64_string:
        raise ValueError("Empty base64 string.")

    if "," in b64_string:
        b64_string = b64_string.split(",", 1)[1]

    normalized = b64_string.strip()
    missing = len(normalized) % 4
    if missing:
        normalized += "=" * (4 - missing)

    try:
        return base64.b64decode(normalized, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise ValueError(f"Invalid base64 data: {exc}") from exc


def decode_image_bytes(data: bytes) -> np.ndarray:
    """Decode image bytes into a BGR image for OpenCV/YOLO."""
    if not validate_image(data):
        raise ValueError("Unsupported image format.")

    buffer = np.frombuffer(data, dtype=np.uint8)
    image = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("Unable to decode image bytes.")
    return image


def encode_image_to_base64(image: np.ndarray) -> str:
    """Encode an image as a base64 JPEG string."""
    ok, buffer = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 90])
    if not ok:
        raise ValueError("Unable to encode annotated image.")
    return base64.b64encode(buffer.tobytes()).decode("utf-8")

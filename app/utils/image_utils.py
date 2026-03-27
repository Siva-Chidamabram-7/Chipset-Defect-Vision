"""
app/utils/image_utils.py — Low-level image I/O helpers for the inference pipeline.

All image validation, decoding, and encoding logic lives here so that
app/main.py and app/model/predictor.py stay free of byte-handling details.

Call chain for a typical /predict request:
  1. validate_image()       — reject unsupported formats early (before decoding)
  2. decode_base64_image()  — strip data-URL prefix, fix padding, base64-decode
     OR
     raw bytes from UploadFile.read()
  3. decode_image_bytes()   — numpy buffer → BGR ndarray via OpenCV
  4. … YOLO inference …
  5. encode_image_to_base64() — annotated BGR ndarray → base64 JPEG string
"""

from __future__ import annotations

import base64
import binascii
from typing import Final

import cv2
import numpy as np

# ── Format guard ──────────────────────────────────────────────────────────────
# These byte signatures (magic numbers) are checked at the start of the raw
# payload to reject clearly unsupported files before handing off to OpenCV.
# WebP is listed twice: once as the 4-byte RIFF header (shared with other RIFF
# formats) and confirmed by the "WEBP" marker at offset 8 in validate_image().
IMAGE_MAGIC_BYTES: Final[list[bytes]] = [
    b"\xff\xd8\xff",          # JPEG — most common camera output
    b"\x89PNG\r\n\x1a\n",     # PNG  — lossless, common in synthetic datasets
    b"RIFF",                  # WebP container (needs secondary check at offset 8)
    b"GIF87a",                # GIF  — rare but accepted
    b"GIF89a",                # GIF  — animated GIF; only first frame is decoded
]


def validate_image(data: bytes) -> bool:
    """Return True when bytes look like a supported image file.

    Checks magic bytes only — does NOT fully decode the image.
    Used in app/main.py before the size check and before decode_image_bytes()
    to return a descriptive 400 instead of an OpenCV error.
    """
    if not data:
        return False

    # Check each known magic prefix
    for magic in IMAGE_MAGIC_BYTES:
        if data[: len(magic)] == magic:
            return True

    # WebP: RIFF header at [0:4] AND "WEBP" marker at [8:12]
    return data[:4] == b"RIFF" and data[8:12] == b"WEBP"


def decode_base64_image(b64_string: str) -> bytes:
    """Decode a base64 image string, removing any data URL prefix.

    Handles both bare base64 and browser-originated data URLs such as:
        data:image/jpeg;base64,/9j/4AAQ...

    Pads the string to a multiple of 4 characters if necessary (browsers
    sometimes strip trailing '=' characters).

    Raises ValueError on empty input or malformed base64.
    Connected to: app/main.py /predict endpoint (image_base64 form field)
    """
    if not b64_string:
        raise ValueError("Empty base64 string.")

    # Strip the "data:image/...;base64," prefix that browsers include
    if "," in b64_string:
        b64_string = b64_string.split(",", 1)[1]

    normalized = b64_string.strip()

    # Re-add padding stripped by some browsers/encoders
    missing = len(normalized) % 4
    if missing:
        normalized += "=" * (4 - missing)

    try:
        return base64.b64decode(normalized, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise ValueError(f"Invalid base64 data: {exc}") from exc


def decode_image_bytes(data: bytes) -> np.ndarray:
    """Decode image bytes into a BGR ndarray for OpenCV/YOLO.

    Uses cv2.imdecode so that JPEG, PNG, WebP, and GIF are all handled
    by the same code path regardless of how the bytes were received
    (raw upload or base64-decoded).

    Returns a uint8 BGR image (H × W × 3).
    Raises ValueError if OpenCV cannot decode the data.
    Connected to: app/main.py /predict, then passed to SolderDefectPredictor.predict()
    """
    if not validate_image(data):
        raise ValueError("Unsupported image format.")

    # np.frombuffer creates a 1-D array of raw bytes; cv2.imdecode interprets them
    buffer = np.frombuffer(data, dtype=np.uint8)
    image  = cv2.imdecode(buffer, cv2.IMREAD_COLOR)  # always decode to BGR
    if image is None:
        raise ValueError("Unable to decode image bytes.")
    return image


def encode_image_to_base64(image: np.ndarray) -> str:
    """Encode a BGR ndarray as a base64 JPEG string.

    Quality is set to 90 — high fidelity while keeping the payload small
    enough for the browser to display inline without noticeably long waits.

    Returns a plain base64 string (no data-URL prefix).
    The frontend prepends "data:image/jpeg;base64," before setting img.src.
    Connected to: SolderDefectPredictor.predict() → /predict response body
    """
    ok, buffer = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 90])
    if not ok:
        raise ValueError("Unable to encode annotated image.")
    return base64.b64encode(buffer.tobytes()).decode("utf-8")

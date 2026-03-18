"""
image_utils.py
──────────────
Helpers for image validation and base64 handling.
"""

import base64
import binascii

# JPEG, PNG, WebP, GIF magic bytes
_MAGIC = [
    b"\xff\xd8\xff",          # JPEG
    b"\x89PNG\r\n\x1a\n",    # PNG
    b"RIFF",                  # WebP (first 4 bytes; "WEBP" follows at offset 8)
    b"GIF87a",                # GIF
    b"GIF89a",                # GIF
]


def validate_image(data: bytes) -> bool:
    """Return True if `data` starts with a known image magic number."""
    for magic in _MAGIC:
        if data[:len(magic)] == magic:
            return True
    # WebP: RIFF....WEBP
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return True
    return False


def decode_base64_image(b64_string: str) -> bytes:
    """
    Decode a base64 image string, stripping any data-URL prefix.
    Raises ValueError on malformed input.
    """
    if not b64_string:
        raise ValueError("Empty base64 string.")

    # Strip optional "data:image/...;base64," prefix
    if "," in b64_string:
        b64_string = b64_string.split(",", 1)[1]

    # Pad if necessary
    b64_string = b64_string.strip()
    missing = len(b64_string) % 4
    if missing:
        b64_string += "=" * (4 - missing)

    try:
        return base64.b64decode(b64_string)
    except (binascii.Error, ValueError) as exc:
        raise ValueError(f"Invalid base64 data: {exc}") from exc

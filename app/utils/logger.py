from __future__ import annotations

import logging
import sys
from pathlib import Path

LOG_FILE = Path(__file__).resolve().parents[2] / "logs" / "inference.log"


def setup_logger() -> logging.Logger:
    _logger = logging.getLogger("chipset_defect_vision")
    if _logger.handlers:
        return _logger

    _logger.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    _logger.addHandler(sh)

    try:
        LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(LOG_FILE, encoding="utf-8")
        fh.setFormatter(fmt)
        _logger.addHandler(fh)
    except Exception:
        pass  # never break the app if log dir is unwritable

    return _logger

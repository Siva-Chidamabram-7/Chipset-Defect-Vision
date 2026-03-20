"""
main.py — project-root entry point for the Chipset Defect Vision API.

Exposes the FastAPI `app` object so the service can be started from the
project root without specifying a package path:

    uvicorn main:app --reload                   # development
    uvicorn main:app --host 0.0.0.0 --port 8080 # production (local)

The actual application is defined in app/main.py.  This file is a thin
re-export so that `uvicorn main:app` resolves correctly.
"""

from app.main import app  # noqa: F401  re-exported for uvicorn

__all__ = ["app"]

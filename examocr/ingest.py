"""Load the question paper as a list of ORIGINAL page images.

PDF pages: if a page is a single full-page embedded image (a scan), that image is
extracted at its native resolution. Otherwise the vector page is rendered at
render_dpi (>=300). Nothing is ever upscaled, sharpened or compressed.
"""
from __future__ import annotations

from pathlib import Path

import cv2
import pymupdf as fitz
import numpy as np

from .config import settings

IMAGE_EXT = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"}


def load_pages(path: str | Path) -> list[np.ndarray]:
    path = Path(path)
    ext = path.suffix.lower()
    if ext == ".pdf":
        return _load_pdf(path)
    if ext in IMAGE_EXT:
        return _load_image(path)
    raise ValueError(f"Unsupported file type: {ext}")


def _load_image(path: Path) -> list[np.ndarray]:
    data = np.fromfile(str(path), dtype=np.uint8)  # handles non-ASCII Windows paths
    if path.suffix.lower() in {".tif", ".tiff"}:
        ok, frames = cv2.imdecodemulti(data, cv2.IMREAD_COLOR)
        if ok and frames:
            return [f for f in frames]
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"Could not decode image: {path}")
    return [img]


def _load_pdf(path: Path) -> list[np.ndarray]:
    dpi = int(settings()["render_dpi"])
    pages: list[np.ndarray] = []
    with fitz.open(str(path)) as doc:
        for page in doc:
            img = _native_scan_image(doc, page)
            if img is None:
                pix = page.get_pixmap(dpi=dpi, colorspace=fitz.csRGB, alpha=False)
                arr = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.h, pix.w, 3)
                img = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
            pages.append(img)
    return pages


def _native_scan_image(doc: fitz.Document, page: fitz.Page) -> np.ndarray | None:
    """Return the embedded image if the page is just one scanned image covering the page."""
    images = page.get_images(full=True)
    if len(images) != 1 or page.get_text("text").strip():
        return None
    xref = images[0][0]
    rects = page.get_image_rects(xref)
    if not rects:
        return None
    covered = rects[0].get_area() / max(page.rect.get_area(), 1)
    if covered < 0.9:
        return None
    info = doc.extract_image(xref)
    arr = np.frombuffer(info["image"], dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        return None
    # Respect page rotation without resampling (90-degree steps are lossless).
    rot = page.rotation % 360
    if rot == 90:
        img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    elif rot == 180:
        img = cv2.rotate(img, cv2.ROTATE_180)
    elif rot == 270:
        img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
    return img

"""Send each region to the right OCR engine based on its ink type. Falls back to Tesseract."""
from __future__ import annotations

import logging
from functools import lru_cache

import numpy as np

from ..config import settings
from .base import OCREngine, OCRResult

log = logging.getLogger(__name__)


@lru_cache(maxsize=4)
def get_engine(name: str) -> OCREngine:
    try:
        if name == "paddle":
            from .paddle import PaddleEngine
            return PaddleEngine()
        if name == "trocr":
            from .trocr import TrOCREngine
            return TrOCREngine()
    except Exception as e:  # missing optional dependency
        log.warning("OCR engine '%s' unavailable (%s); falling back to tesseract", name, e)
    from .tesseract import TesseractEngine
    return TesseractEngine()


def read_region(image: np.ndarray, source_type: str = "printed") -> OCRResult:
    cfg = settings()["ocr"]
    name = cfg["handwritten_engine"] if source_type == "handwritten" else cfg["printed_engine"]
    return get_engine(name).read(image)

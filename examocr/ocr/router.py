"""Send each region to the right OCR engine based on its ink type. Falls back to Tesseract."""
from __future__ import annotations

import logging
import threading

import numpy as np

from ..config import settings
from .base import OCREngine, OCRResult

log = logging.getLogger(__name__)
_ENGINES: dict[str, OCREngine] = {}
_LOAD = threading.Lock()          # the OCR models are GBs: never load them twice at once


def get_engine(name: str) -> OCREngine:
    with _LOAD:
        if name not in _ENGINES:
            _ENGINES[name] = _make(name)
        return _ENGINES[name]


def _make(name: str) -> OCREngine:
    try:
        if name == "paddle":
            from .paddle import PaddleEngine
            return PaddleEngine()
        if name == "trocr":
            from .trocr import TrOCREngine
            return TrOCREngine()
        if name == "glm":
            from .glmocr import GlmOcrEngine
            return GlmOcrEngine()
    except Exception as e:  # missing optional dependency
        log.warning("OCR engine '%s' unavailable (%s); falling back to tesseract", name, e)
    from .tesseract import TesseractEngine
    return TesseractEngine()


def preload_models() -> None:
    """Server start: warm up RDKit in the background (the first structure save is then ~50 ms,
    not ~200). No OCR or layout model is loaded - every box is read by Mathpix."""
    def work():
        try:
            from .. import chem
            chem.render_png(chem.mol_from(smiles="CC"), 120)
        except Exception as e:
            log.warning("RDKit not ready: %s", e)
    threading.Thread(target=work, daemon=True, name="preload-rdkit").start()


preload_handwriting = preload_models      # old name


def read_region(image: np.ndarray, source_type: str = "printed") -> OCRResult:
    cfg = settings()["ocr"]
    name = cfg["handwritten_engine"] if source_type == "handwritten" else cfg["printed_engine"]
    return get_engine(name).read(image)

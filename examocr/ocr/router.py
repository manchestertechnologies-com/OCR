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
    """Load the big offline OCR model(s) in the background at server start, so the first
    crop doesn't also wait for them (GLM-OCR: ~10 s, TrOCR: ~30 s)."""
    cfg = settings()["ocr"]
    names = [n for n in dict.fromkeys((cfg.get("handwritten_engine"), cfg.get("printed_engine")))
             if n in ("glm", "trocr")]

    def work():
        try:                       # RDKit warm-up: the first structure save is then ~50 ms, not ~200
            from .. import chem
            chem.render_png(chem.mol_from(smiles="CC"), 120)
        except Exception as e:
            log.warning("RDKit not ready: %s", e)
        try:
            from ..layout.docmodel import get_layout_model
            get_layout_model()
        except Exception as e:
            log.warning("layout model not loaded: %s", e)
        for n in names:
            try:
                e = get_engine(n)
                log.info("OCR model ready: %s", e.name)
            except Exception as e:      # never take the server down for an optional engine
                log.warning("OCR model %s not loaded: %s", n, e)
    if names or settings()["layout"].get("figure_model"):
        threading.Thread(target=work, daemon=True, name="preload-ocr").start()


preload_handwriting = preload_models      # old name


def read_region(image: np.ndarray, source_type: str = "printed") -> OCRResult:
    cfg = settings()["ocr"]
    name = cfg["handwritten_engine"] if source_type == "handwritten" else cfg["printed_engine"]
    return get_engine(name).read(image)

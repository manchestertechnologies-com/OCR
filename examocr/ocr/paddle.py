"""PaddleOCR engine (optional, needs requirements-ml.txt). Strong on printed text and Indic scripts."""
from __future__ import annotations

import numpy as np

from ..schemas import Box, Word
from .base import OCREngine, OCRResult


class PaddleEngine(OCREngine):
    name = "paddle"

    def __init__(self, lang: str = "en"):
        from paddleocr import PaddleOCR  # imported lazily
        self._ocr = PaddleOCR(use_angle_cls=False, lang=lang, show_log=False)

    def read(self, image: np.ndarray) -> OCRResult:
        res = self._ocr.ocr(image, cls=False) or [[]]
        words: list[Word] = []
        for i, item in enumerate(res[0] or []):
            pts, (txt, conf) = item
            xs = [p[0] for p in pts]
            ys = [p[1] for p in pts]
            # Paddle returns line-level boxes; each line is one "word" with its own line id.
            words.append(Word(text=txt, conf=float(conf),
                              box=Box(x0=int(min(xs)), y0=int(min(ys)),
                                      x1=int(max(xs)), y1=int(max(ys))), line_id=i))
        return OCRResult(words=words)

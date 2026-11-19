from __future__ import annotations

import shutil

import numpy as np
import pytesseract

from ..config import settings
from ..schemas import Box, Word
from .base import OCREngine, OCRResult

_WIN_PATH = r"C:\Program Files\Tesseract-OCR\tesseract.exe"


class TesseractEngine(OCREngine):
    name = "tesseract"

    def __init__(self, lang: str | None = None, psm: int = 6):
        if not shutil.which("tesseract"):
            import os
            if os.path.exists(_WIN_PATH):
                pytesseract.pytesseract.tesseract_cmd = _WIN_PATH
        self.lang = lang or settings()["ocr"]["tesseract_lang"]
        # preserve_interword_spaces keeps spacing verbatim; no dictionary correction.
        self.config = (f"--oem 1 --psm {psm} -c preserve_interword_spaces=1 "
                       f"-c load_system_dawg=0 -c load_freq_dawg=0")

    def read_digits(self, image: np.ndarray) -> str:
        """Read one word as digits only - for question numbers like '8.' that the normal
        pass reads as 'g.' or 'B.'."""
        return pytesseract.image_to_string(
            image, config="--oem 1 --psm 8 -c tessedit_char_whitelist=0123456789.)").strip()

    def read(self, image: np.ndarray) -> OCRResult:
        # Tesseract reads the crop as-is; we do NOT clean the image we hand back to users.
        d = pytesseract.image_to_data(image, lang=self.lang, config=self.config,
                                      output_type=pytesseract.Output.DICT)
        words: list[Word] = []
        for i, txt in enumerate(d["text"]):
            if not txt or not txt.strip():
                continue
            conf = float(d["conf"][i])
            if conf < 0:
                continue
            line_key = d["block_num"][i] * 10000 + d["par_num"][i] * 100 + d["line_num"][i]
            x, y, w, h = d["left"][i], d["top"][i], d["width"][i], d["height"][i]
            words.append(Word(text=txt, conf=conf / 100.0,
                              box=Box(x0=x, y0=y, x1=x + w, y1=y + h), line_id=line_key))
        return OCRResult(words=words)

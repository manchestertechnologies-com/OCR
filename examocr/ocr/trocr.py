"""TrOCR handwriting engine (optional, needs requirements-ml.txt).

TrOCR reads ONE LINE at a time, so we segment lines first (horizontal projection on a
binarized working copy of the crop), then read each line from the ORIGINAL crop pixels.
Fine-tune on your own handwritten samples: training/handwriting/finetune_trocr.py
"""
from __future__ import annotations

import cv2
import numpy as np

from ..schemas import Box, Word
from .base import OCREngine, OCRResult


def segment_lines(gray: np.ndarray, min_gap: int = 3) -> list[tuple[int, int]]:
    binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY_INV, 31, 15)
    prof = binary.sum(axis=1) > 255 * 2
    lines, start, gap = [], None, 0
    for y, on in enumerate(prof):
        if on:
            if start is None:
                start = y
            gap = 0
        elif start is not None:
            gap += 1
            if gap >= min_gap:
                if y - gap - start > 6:
                    lines.append((start, y - gap + 1))
                start, gap = None, 0
    if start is not None:
        lines.append((start, len(prof)))
    return lines


class TrOCREngine(OCREngine):
    name = "trocr"

    def __init__(self, model: str = "microsoft/trocr-base-handwritten"):
        import torch
        from transformers import TrOCRProcessor, VisionEncoderDecoderModel
        self.torch = torch
        self.proc = TrOCRProcessor.from_pretrained(model)
        self.model = VisionEncoderDecoderModel.from_pretrained(model).eval()

    def read(self, image: np.ndarray) -> OCRResult:
        from PIL import Image
        gray = image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        words: list[Word] = []
        for i, (y0, y1) in enumerate(segment_lines(gray)):
            line = image[max(0, y0 - 2):y1 + 2]
            pil = Image.fromarray(cv2.cvtColor(line, cv2.COLOR_BGR2RGB) if line.ndim == 3 else line)
            pv = self.proc(images=pil, return_tensors="pt").pixel_values
            with self.torch.no_grad():
                out = self.model.generate(pv, output_scores=True, return_dict_in_generate=True,
                                          max_new_tokens=128)
            txt = self.proc.batch_decode(out.sequences, skip_special_tokens=True)[0]
            # Mean token probability as line confidence.
            probs = [self.torch.softmax(s, -1).max().item() for s in out.scores]
            conf = float(np.mean(probs)) if probs else 0.0
            words.append(Word(text=txt, conf=conf,
                              box=Box(x0=0, y0=y0, x1=image.shape[1], y1=y1), line_id=i))
        return OCRResult(words=words)

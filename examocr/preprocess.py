"""Working copy for DETECTION ONLY. Never cropped from.

Every box found on the working copy is mapped back to original-page coordinates
with WorkingCopy.to_original() before anything is cut.
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .schemas import Box


@dataclass
class WorkingCopy:
    gray: np.ndarray        # deskewed, denoised greyscale
    binary: np.ndarray      # ink = 255
    matrix: np.ndarray      # 2x3 affine original -> working
    inverse: np.ndarray     # 2x3 affine working -> original
    skew_deg: float
    orig_w: int
    orig_h: int

    def to_original(self, box: Box) -> Box:
        pts = np.array([[box.x0, box.y0], [box.x1, box.y0], [box.x0, box.y1], [box.x1, box.y1]],
                       dtype=np.float64)
        mapped = (self.inverse[:, :2] @ pts.T).T + self.inverse[:, 2]
        x0, y0 = np.floor(mapped.min(axis=0)).astype(int)
        x1, y1 = np.ceil(mapped.max(axis=0)).astype(int)
        return Box(x0=int(x0), y0=int(y0), x1=int(x1), y1=int(y1)).clip(self.orig_w, self.orig_h)


def estimate_skew(binary: np.ndarray, max_deg: float = 10.0) -> float:
    """Projection-profile skew estimate: the angle that makes text rows sharpest."""
    h, w = binary.shape
    scale = 1000 / max(h, w)
    small = cv2.resize(binary, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA) \
        if scale < 1 else binary
    best, best_score = 0.0, -1.0
    center = (small.shape[1] / 2, small.shape[0] / 2)
    for coarse in (np.arange(-max_deg, max_deg + 0.01, 1.0), None):
        angles = coarse if coarse is not None else np.arange(best - 1, best + 1.01, 0.1)
        for a in angles:
            m = cv2.getRotationMatrix2D(center, a, 1.0)
            rot = cv2.warpAffine(small, m, (small.shape[1], small.shape[0]), flags=cv2.INTER_NEAREST)
            prof = rot.sum(axis=1).astype(np.float64)
            s = float(np.sum(np.diff(prof) ** 2))
            if s > best_score:
                best, best_score = float(a), s
    return best


def make_working_copy(original: np.ndarray, deskew: bool = True) -> WorkingCopy:
    gray = original if original.ndim == 2 else cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    # Shadow / uneven lighting removal: divide by a large-kernel background estimate.
    bg = cv2.medianBlur(cv2.dilate(gray, np.ones((7, 7), np.uint8)), 31)
    norm = cv2.divide(gray, bg, scale=255)
    den = cv2.fastNlMeansDenoising(norm, None, h=10, templateWindowSize=7, searchWindowSize=15)
    binary = cv2.adaptiveThreshold(den, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY_INV, 31, 15)

    angle = estimate_skew(binary) if deskew else 0.0
    if abs(angle) < 0.15:
        angle = 0.0
    m = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    inv = cv2.invertAffineTransform(m)
    if angle:
        den = cv2.warpAffine(den, m, (w, h), flags=cv2.INTER_LINEAR, borderValue=255)
        binary = cv2.warpAffine(binary, m, (w, h), flags=cv2.INTER_NEAREST, borderValue=0)
    return WorkingCopy(gray=den, binary=binary, matrix=m, inverse=inv, skew_deg=angle,
                       orig_w=w, orig_h=h)


def remove_ruled_lines(binary: np.ndarray) -> np.ndarray:
    """For notebook-paper photos: remove long horizontal rules (working copy only)."""
    w = binary.shape[1]
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (max(40, w // 8), 1))
    lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    return cv2.subtract(binary, lines)

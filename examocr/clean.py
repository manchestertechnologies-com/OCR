"""The drawing alone, without the photo: a clean copy of a diagram / image-option crop.

A crop cut from a phone photo or a scan shows grey or yellowish paper, shadows, grain and
show-through from the back of the sheet. The clean copy keeps only the ink - every line,
letter and arrow of the drawing - dark on pure white, like a drawn figure in a printed paper.

The locked original crop is never changed (round-trip and hash checks use it); the clean
copy is a separate file used for display and export.
"""
from __future__ import annotations

import cv2
import numpy as np


def clean_drawing(img: np.ndarray) -> np.ndarray:
    """BGR / grey crop -> grey image: ink dark on pure white (255) paper."""
    gray = img if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    H, W = gray.shape[:2]
    if H < 4 or W < 4:
        return gray.copy()
    # paper level everywhere (shadows, uneven light): close away the strokes, smooth
    k = max(15, (min(H, W) // 10) | 1)
    bg = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
    bg = cv2.GaussianBlur(bg, (0, 0), k / 3)
    norm = cv2.divide(gray, np.maximum(bg, 1), scale=255)          # paper ~255, ink darker
    smooth = cv2.GaussianBlur(norm, (0, 0), 1.0).astype(np.float32)
    paper = float(np.median(smooth))
    sigma = 1.4826 * float(np.median(np.abs(smooth - paper))) + 0.5    # paper grain
    # strong ink: clearly dark (Otsu). Weak ink: faint pencil, a few grain-levels below paper
    _, bw = cv2.threshold(norm, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    strong = (bw > 0) & (norm < paper - 25)
    weak_t = paper - max(8.0, 5 * sigma)
    weak = smooth < weak_t
    # a faint stroke stays when it touches a dark one, or is a line in its own right (an
    # "=" bond drawn lightly); grain and dust specks go
    n, lab, st, _ = cv2.connectedComponentsWithStats((weak | strong).astype(np.uint8), 8)
    has_strong = np.zeros(n, bool)
    has_strong[np.unique(lab[strong])] = True
    side = np.maximum(st[:, cv2.CC_STAT_WIDTH], st[:, cv2.CC_STAT_HEIGHT])
    area = st[:, cv2.CC_STAT_AREA]
    keep = (has_strong & (area >= 6)) | ((area >= max(25, int(1e-4 * H * W))) & (side >= 10))
    keep[0] = False
    ink = keep[lab]
    if not ink.any():
        return np.full_like(gray, 255)
    # one consistent dark tone: the darkest ink -> black, the faintest kept ink -> grey 150,
    # and a 1-px rim fading to white keeps the strokes smooth
    lo = float(np.percentile(norm[ink], 3))
    v = (norm.astype(np.float32) - lo) / max(1.0, weak_t - lo)
    tone = np.where(v <= 1, np.clip(v, 0, 1) * 150, 150 + np.clip((v - 1) / 0.6, 0, 1) * 105)
    rim = cv2.dilate(ink.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
    out = np.full_like(gray, 255)
    out[rim] = tone[rim].astype(np.uint8)
    out[ink] = np.minimum(out[ink], 150)
    return out

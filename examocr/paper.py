"""Where is the paper? A phone photo often shows the desk, a keyboard or cloth around the sheet.
Those areas are dark or coloured and would otherwise count as "ink": Snap grows a box over
them, they turn into "diagrams", and the round-trip check reports them as missed content.

background_mask() marks large dark/coloured regions that touch the image border. Scans and
clean PDFs have none, so nothing changes for them. Used on throwaway copies only - crops are
always cut from the untouched original.
"""
from __future__ import annotations

import cv2
import numpy as np


def background_mask(image: np.ndarray) -> np.ndarray:
    """True where the pixel is not paper (same size as image)."""
    H, W = image.shape[:2]
    s = min(1.0, 800.0 / max(H, W))
    small = cv2.resize(image, (max(1, int(W * s)), max(1, int(H * s))), interpolation=cv2.INTER_AREA) \
        if s < 1 else image
    gray = small if small.ndim == 2 else cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    k = max(15, min(gray.shape) // 25) | 1
    se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    # closing wipes out writing (thin dark strokes); what stays dark is a dark AREA
    bg = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, se).astype(np.float32)
    paper = float(np.percentile(bg, 90))
    if paper < 60:                            # a dark image overall: don't guess
        return np.zeros((H, W), bool)
    dark = bg < 0.72 * paper
    if small.ndim == 3:                       # cloth / desk: coloured, paper is not
        sat = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)[:, :, 1]
        sat = cv2.morphologyEx(sat, cv2.MORPH_OPEN, se)
        dark |= (sat > 70) & (bg < 0.93 * paper)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(dark.astype(np.uint8), 8)
    h, w = dark.shape
    keep = np.zeros(n, bool)
    for i in range(1, n):
        x, y, cw, ch, area = stats[i]
        touches = x == 0 or y == 0 or x + cw >= w or y + ch >= h
        keep[i] = touches and area >= 0.004 * h * w
    mask = keep[lab]
    if mask.mean() > 0.6:                     # most of the image "not paper": wrong guess
        return np.zeros((H, W), bool)
    mask = cv2.dilate(mask.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0   # the sheet's shadowed edge
    if s < 1:
        mask = cv2.resize(mask.astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST) > 0
    return mask


def paper_only(image: np.ndarray, mask: np.ndarray | None = None) -> np.ndarray:
    """A copy with everything that is not paper painted white (for detection and OCR only)."""
    mask = background_mask(image) if mask is None else mask
    if not mask.any():
        return image
    out = image.copy()
    out[mask] = 255
    return out

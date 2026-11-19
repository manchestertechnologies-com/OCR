"""Blur and image quality scoring. Scores only - never modifies the image."""
from __future__ import annotations

import cv2
import numpy as np

from .config import settings


def _gray(img: np.ndarray) -> np.ndarray:
    return img if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def blur_score(img: np.ndarray) -> float:
    """Variance of the Laplacian, computed on ink areas so blank paper doesn't dilute it."""
    g = _gray(img)
    if g.size == 0:
        return 0.0
    lap = cv2.Laplacian(g, cv2.CV_64F)
    ink = g < 200
    if ink.sum() > 200:
        # Measure around ink only (dilate so edges are included).
        mask = cv2.dilate(ink.astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool)
        return float(lap[mask].var())
    return float(lap.var())


def regional_blur(img: np.ndarray, grid: int = 4) -> list[list[float]]:
    """Blur per cell - one corner of a phone photo can be sharp while another is not."""
    g = _gray(img)
    h, w = g.shape
    out = []
    for r in range(grid):
        row = []
        for c in range(grid):
            cell = g[r * h // grid:(r + 1) * h // grid, c * w // grid:(c + 1) * w // grid]
            row.append(blur_score(cell) if (cell < 200).sum() > 100 else -1.0)  # -1 = blank
        out.append(row)
    return out


def contrast(img: np.ndarray) -> float:
    return float(_gray(img).std())


def page_quality(img: np.ndarray) -> tuple[float, str, list[str]]:
    q = settings()["quality"]
    score = blur_score(img)
    reasons: list[str] = []
    flag = "ok"
    if score < q["rescan_page_min"]:
        flag = "rescan"
        reasons.append(f"page blur {score:.0f} < rescan threshold {q['rescan_page_min']}")
    elif score < q["blur_page_min"]:
        flag = "blurry"
        reasons.append(f"page blur {score:.0f} < {q['blur_page_min']}")
    if img.shape[1] < q["min_width_px"]:
        flag = flag if flag != "ok" else "low_resolution"
        reasons.append(f"width {img.shape[1]}px < {q['min_width_px']}px")
    if contrast(img) < q["contrast_min"]:
        flag = flag if flag != "ok" else "low_contrast"
        reasons.append("low contrast")
    cells = [v for row in regional_blur(img) for v in row if v >= 0]
    if cells and min(cells) < q["blur_region_min"] and flag == "ok":
        flag = "blurry"
        reasons.append("part of the page is blurry")
    return score, flag, reasons


def crop_quality(img: np.ndarray) -> tuple[float, str]:
    q = settings()["quality"]
    score = blur_score(img)
    if score < q["blur_region_min"]:
        return score, "blurry"
    if contrast(img) < q["contrast_min"] * 0.6:
        return score, "low_contrast"
    return score, "ok"


def touches_edge(img: np.ndarray, band: int = 2) -> bool:
    """True if ink reaches the crop border - a sign a line or label may be cut off."""
    g = _gray(img)
    if g.shape[0] <= 2 * band or g.shape[1] <= 2 * band:
        return True
    ink = g < 160
    return bool(ink[:band].any() or ink[-band:].any() or ink[:, :band].any() or ink[:, -band:].any())

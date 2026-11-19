"""Printed vs handwritten classification per region.

Baseline: stroke-geometry features (printed text has uniform stroke width, aligned
baselines and consistent glyph heights; handwriting varies). If a trained CNN exists at
models/ink_cnn.pt it is used instead (see training/ink/train_ink_cnn.py).
"""
from __future__ import annotations

import cv2
import numpy as np


def ink_features(binary_region: np.ndarray) -> dict[str, float]:
    if binary_region.size == 0 or binary_region.max() == 0:
        return {"stroke_cv": 0.0, "height_cv": 0.0, "baseline_cv": 0.0}
    # Pad with background: a region that is all ink has no zero pixel, and the distance
    # transform would return huge values (NaN features).
    ink = cv2.copyMakeBorder((binary_region > 0).astype(np.uint8), 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)
    dist = cv2.distanceTransform(ink, cv2.DIST_L2, 3).astype(np.float64)
    strokes = dist[dist > 0]
    stroke_cv = float(strokes.std() / (strokes.mean() + 1e-6))
    n, _, stats, _ = cv2.connectedComponentsWithStats((binary_region > 0).astype(np.uint8), 8)
    hs = stats[1:, cv2.CC_STAT_HEIGHT]
    bottoms = stats[1:, cv2.CC_STAT_TOP] + hs
    keep = hs > 4
    hs, bottoms = hs[keep], bottoms[keep]
    if len(hs) < 5:
        return {"stroke_cv": stroke_cv, "height_cv": 0.0, "baseline_cv": 0.0}
    height_cv = float(hs.std() / (hs.mean() + 1e-6))
    # baseline jitter relative to glyph height, using residuals from row medians
    order = np.argsort(bottoms)
    diffs = np.abs(np.diff(bottoms[order]))
    small = diffs[diffs < np.median(hs)]
    baseline_cv = float(small.mean() / (np.median(hs) + 1e-6)) if len(small) else 0.0
    return {"stroke_cv": stroke_cv, "height_cv": height_cv, "baseline_cv": baseline_cv}


def classify_ink(binary_region: np.ndarray) -> tuple[str, float]:
    """Returns (printed|handwritten, confidence)."""
    f = ink_features(binary_region)
    # Hand-tuned linear score; replace with the trained CNN once you have labelled samples.
    s = 1.8 * f["stroke_cv"] + 1.2 * f["height_cv"] + 2.0 * f["baseline_cv"] - 1.6
    p_hand = 1 / (1 + np.exp(-3 * s))
    if p_hand >= 0.5:
        return "handwritten", float(p_hand)
    return "printed", float(1 - p_hand)


def page_source_type(labels: list[str]) -> str:
    s = set(labels)
    if len(s) > 1:
        return "mixed"
    return labels[0] if labels else "printed"

"""Fidelity checks.

1. Hash check: every locked original still matches its SHA-256.
2. Round-trip: paste every crop back at its coordinates on a blank canvas. Inside the pasted
   area the canvas must equal the original pixel for pixel; outside it, the original must have
   no ink (otherwise something on the page was not cropped -> "lost ink").
"""
from __future__ import annotations

import cv2
import numpy as np

from . import store
from .crop import load_png, sha256_file
from .schemas import Project


def hash_check(project: Project) -> list[str]:
    d = store.project_dir(project.id)
    errors = []
    for c in project.crops:
        p = d / c.path
        if not p.exists():
            errors.append(f"missing: {c.path}")
        elif sha256_file(p) != c.sha256:
            errors.append(f"hash mismatch: {c.path}")
    for pg in project.pages:
        if sha256_file(d / pg.path) != pg.sha256:
            errors.append(f"hash mismatch: {pg.path}")
    return errors


def roundtrip(project: Project, min_lost_area: int = 30) -> dict:
    d = store.project_dir(project.id)
    report = {"ok": True, "pages": []}
    for pg in project.pages:
        original = load_png(d / pg.path)
        canvas = np.full_like(original, 255)
        covered = np.zeros(original.shape[:2], dtype=bool)
        mismatched_crops = []
        for c in project.crops:
            if c.page != pg.page or c.superseded:
                continue
            crop = load_png(d / c.path)
            b = c.box
            if crop.shape[:2] != (b.h, b.w):
                mismatched_crops.append(c.id)
                continue
            canvas[b.y0:b.y1, b.x0:b.x1] = crop
            covered[b.y0:b.y1, b.x0:b.x1] = True
        pixel_equal = bool(np.array_equal(canvas[covered], original[covered]))
        gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY) if original.ndim == 3 else original
        ink = (gray < 160) & ~covered
        n, _, stats, _ = cv2.connectedComponentsWithStats(ink.astype(np.uint8), 8)
        lost = [dict(x0=int(s[0]), y0=int(s[1]), x1=int(s[0] + s[2]), y1=int(s[1] + s[3]),
                     area=int(s[4]))
                for s in stats[1:] if s[4] >= min_lost_area]
        page_ok = pixel_equal and not mismatched_crops and not lost
        report["ok"] &= page_ok
        report["pages"].append({"page": pg.page, "ok": page_ok, "pixel_equal": pixel_equal,
                                "coverage": float(covered.mean()),
                                "size_mismatch": mismatched_crops,
                                "lost_ink_regions": lost[:50], "lost_ink_count": len(lost)})
    return report

"""Find figures inside the user's box with a trained document-layout model (PP-DocLayoutV3,
33M parameters, Apache-2.0, offline after one download; ~5 s per crop on a laptop CPU).

It finds what the ink rules miss: hand-drawn structures made of letters and short bond
strokes (CH₃-C=C-H drawn in 2-D), small ring drawings, apparatus sketches. Detections:
  image / chart              -> diagram
  table                      -> table
  formula, 3.5+ lines tall   -> diagram (a drawn structure; a one-line formula or a
                                fraction is text and is read by the OCR model)
  figure_title under a figure (the "A" under a structure) -> joins that figure
The crop is padded to a square with white before detection (the model squeezes its input
to 800x800: a wide crop would be stretched 3x and small drawings get lost).
"""
from __future__ import annotations

import logging
import threading

import cv2
import numpy as np

from ..schemas import Box, RegionKind

log = logging.getLogger(__name__)
_MODEL = None
_FAILED = False
_LOCK = threading.Lock()


class LayoutModel:
    def __init__(self, name: str):
        import torch
        import transformers
        from transformers import AutoImageProcessor, AutoModelForObjectDetection
        transformers.logging.set_verbosity_error()
        transformers.logging.disable_progress_bar()
        self.torch = torch
        log.info("loading layout model %s", name)
        self.proc = AutoImageProcessor.from_pretrained(name)
        self.model = AutoModelForObjectDetection.from_pretrained(name).eval()
        self.labels = self.model.config.id2label
        self._lock = threading.Lock()

    def detect(self, img: np.ndarray, threshold: float = 0.2, square: bool = True) -> list[tuple[str, float, Box]]:
        """square: pad with white to a square first (no stretching); False: the crop as is."""
        from PIL import Image
        H, W = img.shape[:2]
        bgr = img if img.ndim == 3 else cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        if square:
            side = max(H, W)
            sq = np.full((side, side, 3), 255, np.uint8)
            sq[:H, :W] = bgr
            size = (side, side)
        else:
            sq, size = bgr, (H, W)
        rgb = Image.fromarray(cv2.cvtColor(sq, cv2.COLOR_BGR2RGB))
        with self.torch.inference_mode(), self._lock:
            inputs = self.proc(images=rgb, return_tensors="pt")
            out = self.model(**inputs)
            res = self.proc.post_process_object_detection(out, threshold=threshold, target_sizes=[size])[0]
        dets = []
        for s, lab, b in zip(res["scores"], res["labels"], res["boxes"]):
            x0, y0, x1, y1 = (int(round(v)) for v in b.tolist())
            bx = Box(x0=max(0, x0), y0=max(0, y0), x1=min(W, x1), y1=min(H, y1))
            if bx.w >= 4 and bx.h >= 4:
                dets.append((self.labels[int(lab)], float(s), bx))
        return dets


def get_layout_model() -> LayoutModel | None:
    """The shared model, or None when it is switched off / cannot be loaded (then the ink
    rules in layout/heuristic.py are used alone)."""
    global _MODEL, _FAILED
    from ..config import settings
    name = settings()["layout"].get("figure_model", "")
    if not name or _FAILED:
        return None
    with _LOCK:
        if _MODEL is None and not _FAILED:
            try:
                _MODEL = LayoutModel(name)
            except Exception as e:          # torch/transformers missing, no download yet
                log.warning("layout model %s unavailable (%s); using ink rules only", name, e)
                _FAILED = True
        return _MODEL


def _inter(a: Box, b: Box) -> int:
    return max(0, min(a.x1, b.x1) - max(a.x0, b.x0)) * max(0, min(a.y1, b.y1) - max(a.y0, b.y0))


def _area(b: Box) -> int:
    return max(1, b.w * b.h)


def select_figures(dets: list[tuple[str, float, Box]], line_h: float, crop_w: int, crop_h: int,
                   image_min: float = 0.25, formula_min: float = 0.3, table_min: float = 0.45,
                   formula_lines: float = 3.0) -> list[tuple[RegionKind, Box, bool]]:
    """Model detections -> figures to cut out (crop coordinates), as (kind, box, formula_only).
    formula_only: found only as a tall "formula" (may still be stacked reaction conditions,
    see find_figures). Pure function (tested)."""
    figs: list[list] = []                                    # [kind, score, box, formula_only]
    titles = []
    for lab, s, b in dets:
        # a weak "image" two lines tall is a line of writing ("a) 1:1:1")
        if lab in ("image", "chart") and s >= image_min and b.w >= 1.2 * line_h \
                and (b.h >= 3.0 * line_h or (s >= 0.5 and b.h >= 1.2 * line_h)):
            figs.append([RegionKind.diagram, s, b, False])
        elif lab == "formula" and s >= formula_min and b.h >= formula_lines * line_h and b.w >= 2 * line_h:
            figs.append([RegionKind.diagram, s, b, True])
        elif lab == "table" and s >= table_min:
            figs.append([RegionKind.table, s, b, False])
        elif lab == "figure_title" and s >= 0.2 and b.h <= 2.5 * line_h:
            titles.append(b)
    # never the whole box: the user drew the box around the question, not around one drawing
    figs = [f for f in figs if _area(f[2]) < 0.85 * crop_w * crop_h]
    # a weaker box around 2+ stronger ones is a group ("A + B + C"): keep the parts
    keep = []
    for i, f in enumerate(figs):
        inside = [g for j, g in enumerate(figs) if j != i and g[1] > f[1] and _inter(f[2], g[2]) >= 0.8 * _area(g[2])]
        if len(inside) < 2:
            keep.append(f)
    # overlapping detections of one drawing -> one box
    merged = True
    while merged:
        merged = False
        for i in range(len(keep)):
            for j in range(i + 1, len(keep)):
                a, b = keep[i], keep[j]
                if _inter(a[2], b[2]) >= 0.4 * min(_area(a[2]), _area(b[2])):
                    kind = RegionKind.table if RegionKind.table in (a[0], b[0]) and a[0] == b[0] else RegionKind.diagram
                    keep[i] = [kind, max(a[1], b[1]), a[2].union(b[2]), a[3] and b[3]]
                    del keep[j]
                    merged = True
                    break
            if merged:
                break
    # the caption right under a figure ("A", "B") belongs in its image
    for t in titles:
        if any(_inter(t, f[2]) > 0.3 * _area(t) for f in keep):
            continue            # it sits on a drawing ("(4)" at a graph's corner): not a caption below
        for f in keep:
            b = f[2]
            ov = min(b.x1, t.x1) - max(b.x0, t.x0)
            if ov > 0.5 * t.w and -0.5 * line_h <= t.y0 - b.y1 <= 1.5 * line_h:
                f[2] = b.union(t)
                break
    keep.sort(key=lambda f: (f[2].y0, f[2].x0))
    return [(f[0], f[2], f[3]) for f in keep]


def find_figures(img: np.ndarray, line_h: float) -> list[tuple[RegionKind, Box]] | None:
    """Figures inside a crop, or None when the model is unavailable."""
    m = get_layout_model()
    if m is None:
        return None
    try:
        # twice: padded to a square (small drawings in wide crops) and as is - each finds
        # drawings the other misses; overlapping finds are merged by select_figures
        dets = m.detect(img, square=True) + m.detect(img, square=False)
    except Exception as e:
        log.warning("layout model failed on a crop: %s", e)
        return None
    from ..ocr.handseg import binarize
    bw = binarize(img) > 0
    out = []
    for kind, b, formula_only in select_figures(dets, max(line_h, 8.0), img.shape[1], img.shape[0]):
        if formula_only and has_arrow(bw[b.y0:b.y1, b.x0:b.x1]):
            continue            # reagents written over / under a reaction arrow: text, read by OCR
        out.append((kind, b))
    return out


def has_arrow(region: np.ndarray) -> bool:
    """A long, thin horizontal stroke across most of the region: a reaction arrow (or a
    fraction bar) with writing above / below it - text, not a drawn structure."""
    if region.size == 0:
        return False
    n, _, st, _ = cv2.connectedComponentsWithStats(region.astype(np.uint8), 8)
    H, W = region.shape[:2]
    if any(st[i, 2] >= 0.45 * W and st[i, 3] <= max(0.25 * H, 3) and st[i, 2] >= 6 * st[i, 3]
           for i in range(1, n)):
        return True
    # a hand-drawn arrow is often slanted and touches the reagents' letters, so it is not a
    # stroke of its own: look for a long, nearly flat straight line. A long vertical line
    # too means axes (a graph), which is a drawing.
    lines = cv2.HoughLinesP(region.astype(np.uint8) * 255, 1, np.pi / 180, threshold=max(20, int(0.25 * W)),
                            minLineLength=int(0.4 * W), maxLineGap=max(3, int(0.04 * W)))
    flat = vert = False
    for x1, y1, x2, y2 in (lines.reshape(-1, 4) if lines is not None else []):
        dx, dy = abs(int(x2) - int(x1)), abs(int(y2) - int(y1))
        flat |= dy <= 0.27 * dx and dx >= 0.4 * W
        vert |= dx <= 0.27 * dy and dy >= 0.5 * H
    return flat and not vert

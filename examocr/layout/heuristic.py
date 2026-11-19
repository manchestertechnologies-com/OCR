"""Heuristic layout analysis (no training needed). Works on the working copy.

1. Zones: split the page into full-width bands and two-column bands (gutter detection).
2. Text: OCR each column segment of the working copy to get line boxes (for LOCATING only;
   the verbatim text is read later from the original crop).
3. Graphics: connected components that are not text-sized are clustered into diagrams;
   clusters with a grid of ruled lines are tables. If a YOLO layout model is configured,
   it replaces step 3.
"""
from __future__ import annotations

import logging

import cv2
import numpy as np

from ..config import settings
from ..ink import classify_ink
from ..ocr.tesseract import TesseractEngine
from ..preprocess import WorkingCopy
from ..schemas import Box, RegionKind, Word
from .types import Graphic, Line, PageLayout, Segment

log = logging.getLogger(__name__)


# ---------------------------------------------------------------- zones / columns
def find_gutter(binary: np.ndarray) -> tuple[int, int] | None:
    """Return (gx0, gx1) of a vertical white gutter near the page centre, or None."""
    h, w = binary.shape
    body = binary[int(h * 0.08):int(h * 0.95)] > 0
    rows_with_ink = body.any(axis=1).sum()
    if rows_with_ink < 50:
        return None
    frac = body.sum(axis=0) / rows_with_ink           # share of text rows inked at each x
    frac = np.convolve(frac, np.ones(5) / 5, mode="same")
    lo, hi = int(w * 0.35), int(w * 0.65)
    quiet = frac[lo:hi] < 0.015
    best, run_start, best_span = None, None, 0
    for i, q in enumerate(np.append(quiet, False)):
        if q and run_start is None:
            run_start = i
        elif not q and run_start is not None:
            if i - run_start > best_span:
                best_span, best = i - run_start, (lo + run_start, lo + i)
            run_start = None
    if best is None or best_span < w * 0.012:
        return None
    left_ink = frac[int(w * 0.05):best[0]].mean()
    right_ink = frac[best[1]:int(w * 0.95)].mean()
    if left_ink < 0.02 or right_ink < 0.02:
        return None
    return best


def find_zones(binary: np.ndarray) -> list[tuple[int, int, list[tuple[int, int]]]]:
    """[(y0, y1, [(x0, x1), ...columns])] in reading order."""
    h, w = binary.shape
    gutter = find_gutter(binary)
    if gutter is None:
        return [(0, h, [(0, w)])]
    gx0, gx1 = gutter
    spanning = (binary[:, gx0:gx1] > 0).any(axis=1).astype(np.uint8)
    # merge spanning rows into bands (close small gaps)
    spanning = cv2.morphologyEx(spanning.reshape(-1, 1), cv2.MORPH_CLOSE,
                                np.ones((25, 1), np.uint8)).ravel()
    zones, y, mid = [], 0, (gx0 + gx1) // 2
    while y < h:
        state = spanning[y]
        y_end = y
        while y_end < h and spanning[y_end] == state:
            y_end += 1
        if state:
            zones.append((y, y_end, [(0, w)]))
        else:
            zones.append((y, y_end, [(0, mid), (mid, w)]))
        y = y_end
    # drop empty zones
    return [z for z in zones if (binary[z[0]:z[1]] > 0).sum() > 50]


# ---------------------------------------------------------------- text lines
def ocr_lines(wc: WorkingCopy, box: Box, engine: TesseractEngine) -> list[Line]:
    region = wc.gray[box.y0:box.y1, box.x0:box.x1]
    if region.size == 0 or (wc.binary[box.y0:box.y1, box.x0:box.x1] > 0).sum() < 30:
        return []
    res = engine.read(region)
    groups: dict[int, list[Word]] = {}
    for wd in res.words:
        wd.box = Box(x0=wd.box.x0 + box.x0, y0=wd.box.y0 + box.y0,
                     x1=wd.box.x1 + box.x0, y1=wd.box.y1 + box.y0)
        groups.setdefault(wd.line_id, []).append(wd)
    lines = []
    for ws in groups.values():
        lb = ws[0].box
        for wd in ws[1:]:
            lb = lb.union(wd.box)
        lines.append(Line(box=lb, words=ws))
    lines.sort(key=lambda l: (l.box.y0, l.box.x0))
    return lines


# ---------------------------------------------------------------- graphics
def detect_graphics(binary: np.ndarray, words: list[Word], med_h: float,
                    text_boxes: list[Box] | None = None, min_area_px: float | None = None) -> list[Graphic]:
    """text_boxes: word boxes that belong to real text lines (protected from being absorbed)."""
    cfg = settings()["layout"]
    h, w = binary.shape
    n, labels, stats, _ = cv2.connectedComponentsWithStats((binary > 0).astype(np.uint8), 8)
    cw_all, ch_all = stats[:, cv2.CC_STAT_WIDTH], stats[:, cv2.CC_STAT_HEIGHT]
    # Components that are letters of real text words must never be pulled into a diagram.
    text_mask = np.zeros((h, w), dtype=bool)
    if text_boxes is None:
        text_boxes = [wd.box for wd in words if len(wd.text) >= 4 and wd.conf > 0.5]
    for b in text_boxes:
        text_mask[b.y0:b.y1, b.x0:b.x1] = True
    cx = np.clip(stats[:, 0] + stats[:, 2] // 2, 0, w - 1)
    cy = np.clip(stats[:, 1] + stats[:, 3] // 2, 0, h - 1)
    is_text = text_mask[cy, cx]
    is_text[0] = True

    tall = (ch_all > 2.2 * med_h) & (cw_all > 1.2 * med_h)
    long_rule = (cw_all > 6 * med_h) & (ch_all < 0.5 * med_h)
    wide_shape = (cw_all > 4 * med_h) & (ch_all > 0.6 * med_h) & ~is_text   # cylinders, pipes, beams
    separator = (cw_all > 0.85 * w) & (ch_all < med_h)
    keep = (tall | long_rule | wide_shape) & ~separator
    keep[0] = False                                   # background
    mask = np.where(keep[labels], 255, 0).astype(np.uint8)
    k = max(3, int(med_h * 1.2))
    merged = cv2.dilate(mask, np.ones((k, k), np.uint8))
    cn, _, cstats, _ = cv2.connectedComponentsWithStats(merged, 8)
    min_area = min_area_px if min_area_px is not None else cfg["diagram_min_area_frac"] * w * h

    def big_enough(b: Box) -> bool:
        # Narrow tall marks (matrix brackets, integral signs, braces) are maths, not drawings;
        # wide short drawings (a piston, a beam) are allowed.
        if b.w < 3 * med_h:
            return False
        if b.h >= 2.5 * med_h:
            return b.area >= min_area
        return b.w >= 6 * med_h and b.h >= 1.0 * med_h and b.area >= 0.4 * min_area

    pieces: list[Box] = []
    for i in range(1, cn):
        x, y, cw, ch, _ = cstats[i]
        # measure the real drawing, not the dilated blob around it
        cluster = _tight(mask, Box(x0=int(x), y0=int(y), x1=int(x + cw), y1=int(y + ch)))
        if not big_enough(cluster):
            continue
        # Stacked / side-by-side figures (graph options (a)-(d)) are merged by the dilation:
        # split the cluster at clear white gaps when every piece is a drawing on its own.
        pieces.extend(_xy_split(mask, cluster, max(12, int(0.35 * med_h)), big_enough))

    small_marks = ~keep          # dashes, dots, arrowheads - not pieces of other drawings
    small_marks[0] = False
    # Each small mark belongs to the NEAREST drawing only, so labels/arrows sitting between
    # two stacked graphs can't chain them together.
    x0s, y0s = stats[:, 0], stats[:, 1]
    x1s, y1s = x0s + stats[:, 2], y0s + stats[:, 3]
    if pieces:
        dist = np.stack([np.maximum.reduce([p.x0 - x1s, np.zeros_like(x0s), x0s - p.x1])
                         + np.maximum.reduce([p.y0 - y1s, np.zeros_like(y0s), y0s - p.y1]) for p in pieces])
        owner = dist.argmin(axis=0)
    boxes = [_absorb_marks(p, stats, is_text | ~small_marks | (owner != j), med_h)
             for j, p in enumerate(pieces)]

    # Labels (A, B, 30°, (a), ?) go to the NEAREST drawing only. Words that belong to a real
    # text line (the question sentence) are never labels.
    protected = {(b.x0, b.y0, b.x1, b.y1) for b in text_boxes}
    owned: dict[int, list[Word]] = {i: [] for i in range(len(boxes))}
    for wd in words:
        if not boxes:
            break
        if (wd.box.x0, wd.box.y0, wd.box.x1, wd.box.y1) in protected:
            continue
        cxw, cyw = (wd.box.x0 + wd.box.x1) / 2, (wd.box.y0 + wd.box.y1) / 2
        dists = [max(b.x0 - cxw, 0, cxw - b.x1) + max(b.y0 - cyw, 0, cyw - b.y1) for b in boxes]
        j = int(np.argmin(dists))
        if dists[j] <= med_h:
            owned[j].append(wd)
    out: list[Graphic] = []
    for j, box in enumerate(boxes):
        for wd in owned[j]:
            if len(wd.text) <= 6:
                box = box.union(wd.box)
        # Text paragraphs that merely touch the cluster must not be swallowed.
        inside_text = sum(wd.box.area for wd in owned[j] if len(wd.text) > 6)
        if inside_text > 0.5 * box.area:
            continue
        kind = RegionKind.table if _is_table(binary[box.y0:box.y1, box.x0:box.x1]) \
            else RegionKind.diagram
        out.append(Graphic(kind=kind, box=box.clip(w, h), labels=owned[j]))
    return _merge_overlapping(out)


def _xy_split(mask: np.ndarray, box: Box, gap: int, ok, depth: int = 0) -> list[Box]:
    """Recursive XY-cut on the graphic mask: split at full-width / full-height white gaps
    of at least `gap` px, but only if every resulting piece passes `ok`."""
    region = mask[box.y0:box.y1, box.x0:box.x1] > 0
    if depth > 4 or region.size == 0:
        return [box]
    for axis in (0, 1):                     # 0: split rows (stacked), 1: split columns (side by side)
        prof = region.any(axis=1 - axis)
        runs, start = [], None
        for i, v in enumerate(np.append(prof, False)):
            if v and start is None:
                start = i
            elif not v and start is not None:
                if runs and start - runs[-1][1] < gap:
                    runs[-1][1] = i
                else:
                    runs.append([start, i])
                start = None
        if len(runs) < 2:
            continue
        if axis == 0:
            parts = [Box(x0=box.x0, y0=box.y0 + s, x1=box.x1, y1=box.y0 + e) for s, e in runs]
        else:
            parts = [Box(x0=box.x0 + s, y0=box.y0, x1=box.x0 + e, y1=box.y1) for s, e in runs]
        parts = [_tight(mask, p) for p in parts]
        if all(ok(p) for p in parts):
            out: list[Box] = []
            for p in parts:
                out.extend(_xy_split(mask, p, gap, ok, depth + 1))
            return out
    return [box]


def _tight(mask: np.ndarray, b: Box) -> Box:
    ys, xs = np.nonzero(mask[b.y0:b.y1, b.x0:b.x1])
    if ys.size == 0:
        return b
    return Box(x0=b.x0 + int(xs.min()), y0=b.y0 + int(ys.min()),
               x1=b.x0 + int(xs.max()) + 1, y1=b.y0 + int(ys.max()) + 1)


def _absorb_marks(box: Box, stats: np.ndarray, is_text: np.ndarray, med_h: float,
                  max_iter: int = 200) -> Box:
    """Grow the diagram box over nearby small marks - dashes, dots, arrowheads, hatching -
    repeatedly, so a dashed line is followed dash by dash and never cut off.
    `is_text` marks components that must NOT be absorbed (text, other drawings' pieces)."""
    x0s, y0s = stats[:, 0], stats[:, 1]
    x1s, y1s = x0s + stats[:, 2], y0s + stats[:, 3]
    reach = max(4, int(med_h * 0.8))
    for _ in range(max_iter):
        p = box.pad(reach)
        near = (x1s > p.x0) & (x0s < p.x1) & (y1s > p.y0) & (y0s < p.y1) & ~is_text
        outside = (x0s < box.x0) | (y0s < box.y0) | (x1s > box.x1) | (y1s > box.y1)
        grow = np.flatnonzero(near & outside)
        if grow.size == 0:
            break
        box = Box(x0=int(min(box.x0, x0s[grow].min())), y0=int(min(box.y0, y0s[grow].min())),
                  x1=int(max(box.x1, x1s[grow].max())), y1=int(max(box.y1, y1s[grow].max())))
    return box


def _is_table(region: np.ndarray) -> bool:
    if region.size == 0:
        return False
    rh, rw = region.shape
    # A table's ruled lines run across most of it. Graph axes and boxes inside a figure
    # (e.g. four small P-V graphs) are much shorter, so they don't count.
    hk = cv2.getStructuringElement(cv2.MORPH_RECT, (max(10, int(rw * 0.6)), 1))
    vk = cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(10, int(rh * 0.6))))
    hl = cv2.morphologyEx(region, cv2.MORPH_OPEN, hk)
    vl = cv2.morphologyEx(region, cv2.MORPH_OPEN, vk)
    n_h = cv2.connectedComponents((hl > 0).astype(np.uint8))[0] - 1
    n_v = cv2.connectedComponents((vl > 0).astype(np.uint8))[0] - 1
    return n_h >= 3 and n_v >= 2


def _merge_overlapping(gs: list[Graphic]) -> list[Graphic]:
    changed = True
    while changed:
        changed = False
        for i in range(len(gs)):
            for j in range(i + 1, len(gs)):
                # merge only real overlaps - neighbouring figures whose label boxes touch stay apart
                small = min(gs[i].box.area, gs[j].box.area)
                if gs[i].box.intersection(gs[j].box) > 0.2 * max(1, small):
                    kind = RegionKind.table if RegionKind.table in (gs[i].kind, gs[j].kind) \
                        else RegionKind.diagram
                    gs[i] = Graphic(kind=kind, box=gs[i].box.union(gs[j].box),
                                    labels=gs[i].labels + gs[j].labels)
                    del gs[j]
                    changed = True
                    break
            if changed:
                break
    return gs


# ---------------------------------------------------------------- main entry
def analyze_page(wc: WorkingCopy, page: int) -> PageLayout:
    cfg = settings()["layout"]
    engine = TesseractEngine(psm=4)
    h, w = wc.binary.shape
    zones = find_zones(wc.binary)

    segments: list[Segment] = []
    all_words: list[Word] = []
    text_boxes: list[Box] = []
    for y0, y1, cols in zones:
        for x0, x1 in cols:
            seg = Segment(page=page, order=len(segments), box=Box(x0=x0, y0=y0, x1=x1, y1=y1))
            seg.lines = ocr_lines(wc, seg.box, engine)
            for ln in seg.lines:
                all_words.extend(ln.words)
                # A line of 3+ confident words is real text; a lone "I" is likely a dash.
                good = [wd for wd in ln.words if wd.conf > 0.5]
                if len(good) >= 3:
                    text_boxes.extend(wd.box for wd in good)
            segments.append(seg)

    heights = [wd.box.h for wd in all_words if wd.conf > 0.5 and wd.box.h > 3]
    med_h = float(np.median(heights)) if heights else max(12.0, h / 120)

    if cfg["engine"] == "yolo":
        from .yolo import detect_graphics_yolo
        graphics = detect_graphics_yolo(wc, all_words)
        if graphics is None:
            graphics = detect_graphics(wc.binary, all_words, med_h, text_boxes)
    else:
        graphics = detect_graphics(wc.binary, all_words, med_h, text_boxes)

    # Assign graphics to segments; drop text lines that are really diagram labels.
    for g in graphics:
        cx, cy = (g.box.x0 + g.box.x1) / 2, (g.box.y0 + g.box.y1) / 2
        target = next((s for s in segments if s.box.x0 <= cx < s.box.x1 and s.box.y0 <= cy < s.box.y1),
                      None)
        if target is None:
            target = max(segments, key=lambda s: s.box.intersection(g.box))
        target.graphics.append(g)
    for seg in segments:
        diag_boxes = [g.box for g in seg.graphics if g.kind == RegionKind.diagram]
        kept = []
        for ln in seg.lines:
            # Only short label-like lines fully inside a drawing are dropped; a real sentence
            # (3+ words) that a diagram box happens to overlap is kept.
            real_text = sum(wd.conf > 0.5 for wd in ln.words) >= 3
            inside = not real_text and any(
                ln.box.intersection(b) > 0.9 * max(1, ln.box.area) for b in diag_boxes)
            if not inside:
                region = wc.binary[ln.box.y0:ln.box.y1, ln.box.x0:ln.box.x1]
                ln.source_type, _ = classify_ink(region)
                kept.append(ln)
        seg.lines = kept

    columns = max(len(c) for _, _, c in zones) if zones else 1
    return PageLayout(page=page, width=w, height=h, columns=columns, segments=segments,
                      median_char_h=med_h)

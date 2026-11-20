"""Cut a handwritten crop into short pieces a line-level OCR model (TrOCR) can read.

Robust to a tilted or curved phone photo: letters are chained to their right-hand neighbour
at the same height instead of cutting the page into global horizontal bands.
  1. chains   - each mark links to its nearest right neighbour that overlaps it vertically
  2. scripts  - small chains (an exponent "2+", a "Δ" over an arrow) join the nearest chain
  3. fractions- a chain above a bar and a chain below it become one piece (M over 3)
  4. lines    - chains that continue each other at the same height, across wide gaps
  5. pieces   - the chains of a line closer than a wide gap; long pieces cut at word gaps
  6. dots, decimal points, i-dots and long strokes go to the piece around them
Works on a throwaway binary copy; the crop itself is never changed.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

from ..schemas import Box


def binarize(image: np.ndarray) -> np.ndarray:
    """Ink = 255. Background-normalised so photo shading and grey paper don't become ink."""
    gray = image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    k = max(15, (min(gray.shape[:2]) // 12) | 1)
    bg = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
    norm = cv2.divide(gray, bg, scale=255)
    _, bw = cv2.threshold(norm, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    bw[norm > 215] = 0          # an almost empty crop: Otsu would pick up paper grain
    return bw


@dataclass
class Piece:
    box: Box                    # crop coordinates, tight around the ink
    line: int                   # line index, reading order
    comps: list[int] = field(default_factory=list)
    scripts: int = 0            # raised / lowered small marks (exponents, subscripts)


@dataclass
class HandLayout:
    pieces: list[Piece]
    line_boxes: list[Box]
    line_h: float               # median line height (ascender to descender)
    char_h: float               # median mark height
    labels: np.ndarray          # connected-component label image
    stats: np.ndarray


class _UF:
    def __init__(self, keys):
        self.p = {k: k for k in keys}

    def find(self, a):
        while self.p[a] != a:
            self.p[a] = self.p[self.p[a]]
            a = self.p[a]
        return a

    def join(self, a, b):
        self.p[self.find(a)] = self.find(b)


def segment(binary: np.ndarray, skip: np.ndarray | None = None) -> HandLayout:
    """skip: boolean per component label - ink to ignore (drawings already cut out)."""
    n, labels, stats, _ = cv2.connectedComponentsWithStats((binary > 0).astype(np.uint8), 8)
    x0s, y0s, ws, hs, areas = (stats[:, i].astype(np.int64) for i in range(5))
    x1s, y1s = x0s + ws, y0s + hs
    cys = y0s + hs / 2.0
    valid = (areas >= 8) & (hs >= 3)
    valid[0] = False
    if skip is not None:
        valid &= ~skip
    if valid.sum() == 0:
        return HandLayout([], [], 24.0, 24.0, labels, stats)
    tall_ok = valid & (hs > 6)
    med = float(np.median(hs[tall_ok])) if tall_ok.any() else float(np.median(hs[valid]))
    speck = valid & (areas < 0.12 * med * med) & (hs < 0.45 * med) & (ws < 0.6 * med)
    tall = valid & (hs > 2.6 * med)
    link = valid & ~speck & ~tall              # letters, digits, minus signs, arrows, bars

    # 1. chains
    idx = np.flatnonzero(link)
    uf = _UF([int(i) for i in idx])
    best_in: dict[int, tuple[float, int]] = {}
    for i in idx:
        gap = x0s[idx] - x1s[i]
        ov = np.minimum(y1s[i], y1s[idx]) - np.maximum(y0s[i], y0s[idx])
        ok = (x0s[idx] > x0s[i] + 0.3 * ws[i]) & (gap > -0.5 * np.minimum(ws[i], ws[idx])) \
            & (gap < 2.5 * med) & (ov >= 0.35 * np.minimum(hs[i], hs[idx]))
        # a flat mark (minus, arrow, bar) must sit within its neighbour's height, not above it
        flat = hs[idx] < 0.45 * med
        ok &= ~flat | ((cys[idx] > y0s[i] + 0.1 * hs[i]) & (cys[idx] < y1s[i] - 0.1 * hs[i]))
        if hs[i] < 0.45 * med:
            ok &= (cys[i] > y0s[idx] + 0.1 * hs[idx]) & (cys[i] < y1s[idx] - 0.1 * hs[idx])
        if not ok.any():
            continue
        score = np.where(ok, np.maximum(gap, 0) + 0.5 * np.abs(cys[idx] - cys[i]), np.inf)
        k = int(np.argmin(score))
        j, sc = int(idx[k]), float(score[k])
        if j not in best_in or sc < best_in[j][0]:
            best_in[j] = (sc, int(i))
    for j, (_, i) in best_in.items():
        uf.join(i, j)
    groups: dict[int, list[int]] = {}
    for i in idx:
        groups.setdefault(uf.find(int(i)), []).append(int(i))
    ch = [sorted(c, key=lambda i: x0s[i]) for c in groups.values()]

    def cbox(c):
        c = np.asarray(c)
        return Box(x0=int(x0s[c].min()), y0=int(y0s[c].min()), x1=int(x1s[c].max()), y1=int(y1s[c].max()))

    if not ch:
        return HandLayout([], [], med, med, labels, stats)

    # 2. small chains join the nearest bigger chain
    hts = [cbox(c).h for c in ch]
    ref_h = float(np.median([h for c, h in zip(ch, hts) if len(c) >= 2] or hts))
    changed = True
    while changed:
        changed = False
        bx = [cbox(c) for c in ch]
        for a in sorted(range(len(ch)), key=lambda k: bx[k].h):
            A = bx[a]
            if len(ch[a]) > 4 or A.h >= 0.8 * ref_h:
                continue
            best, bd = None, None
            for b in range(len(ch)):
                B = bx[b]
                if b == a or B.h <= A.h:
                    continue
                if A.x1 < B.x0 - 0.3 * ref_h or A.x0 > B.x1 + 0.5 * ref_h:
                    continue
                dy = max(B.y0 - A.y1, A.y0 - B.y1, 0)
                if dy > 0.6 * ref_h:
                    continue
                # a short word on the NEXT line ("is") is not a script: joined, it would make a
                # piece two lines tall
                if max(A.y1, B.y1) - min(A.y0, B.y0) > 1.45 * max(ref_h, B.h):
                    continue
                d = dy + 0.2 * max(B.x0 - A.x1, A.x0 - B.x1, 0)
                if bd is None or d < bd:
                    best, bd = b, d
            if best is not None:
                ch[best] = sorted(ch[best] + ch[a], key=lambda i: x0s[i])
                del ch[a]
                changed = True
                break

    # 3. fractions: a short chain (denominator) right under a horizontal bar that runs across
    # it - the bar may be its own mark or touch the numerator, so look at the pixels
    ink = binary > 0

    def bar_over(A: Box, B: Box) -> bool:
        y_lo, y_hi = int(max(A.y0 + 0.4 * A.h, A.y1 - 0.8 * med)), int(min(B.y0 + 0.5 * med, B.y1))
        if y_hi <= y_lo or B.w < 3:
            return False
        rows = ink[y_lo:y_hi, B.x0:B.x1]
        return bool(rows.size) and float(rows.mean(axis=1).max()) >= 0.8

    changed = True
    while changed:
        changed = False
        bx = [cbox(c) for c in ch]
        for b in range(len(ch)):
            B = bx[b]
            if B.h > 1.3 * ref_h or len(ch[b]) > 4:
                continue
            for a in range(len(ch)):
                A = bx[a]
                if b == a or B.y0 < A.y1 - 0.5 * med or B.y0 - A.y1 > 1.2 * med:
                    continue
                if B.x0 < A.x0 - 0.3 * med or B.x1 > A.x1 + 0.3 * med:
                    continue
                if not bar_over(A, B):
                    continue
                ch[a] = sorted(ch[a] + ch[b], key=lambda i: x0s[i])
                del ch[b]
                changed = True
                break
            if changed:
                break

    # 4. lines
    boxes = [cbox(c) for c in ch]
    multi = [b.h for c, b in zip(ch, boxes) if len(c) >= 2]
    lh = float(np.median(multi or [b.h for b in boxes]))
    luf = _UF(list(range(len(ch))))
    order = sorted(range(len(ch)), key=lambda k: boxes[k].x0)

    def body(c):
        """The chain's letters without exponents/subscripts: marks that cross its midline."""
        m = float(np.median(cys[c]))
        b = [i for i in c if y0s[i] < m < y1s[i]]
        return b or c

    taken: dict[int, tuple[float, int]] = {}
    for a in order:
        A = boxes[a]
        end_y = float(np.mean([cys[i] for i in sorted(body(ch[a]), key=lambda i: x1s[i])[-3:]]))
        best, bs = None, None
        for b in order:
            B = boxes[b]
            # B starts after A ends (a little overlap: the two halves of ⇌ end up in both)
            if b == a or B.x0 < A.x1 - max(0.2 * lh, 0.4 * min(A.w, B.w)) or B.x0 <= A.x0:
                continue
            st_y = float(np.mean([cys[i] for i in body(ch[b])[:3]]))
            ov = min(A.y1, B.y1) - max(A.y0, B.y0)
            if abs(end_y - st_y) > 0.5 * lh or ov < 0.3 * min(A.h, B.h):
                continue
            s_ = (B.x0 - A.x1) + 2 * abs(end_y - st_y)
            if bs is None or s_ < bs:
                best, bs = b, s_
        if best is not None and (best not in taken or bs < taken[best][0]):
            taken[best] = (bs, a)
    for b, (_, a) in taken.items():
        luf.join(a, b)
    lines: dict[int, list[int]] = {}
    for k in range(len(ch)):
        lines.setdefault(luf.find(k), []).append(k)
    line_list = sorted(lines.values(),
                       key=lambda ks: float(np.median([cys[i] for k in ks for i in ch[k]])) if len(ks) else 0)
    # reading order: by the line's left end height (a tilted line is read where it starts)
    line_list.sort(key=lambda ks: min((boxes[k].x0, boxes[k].y0 + boxes[k].h / 2) for k in ks)[1])

    # 5. pieces
    pieces: list[Piece] = []
    line_boxes: list[Box] = []
    for li, ks in enumerate(line_list):
        ks.sort(key=lambda k: boxes[k].x0)
        lbx = boxes[ks[0]]
        for k in ks[1:]:
            lbx = lbx.union(boxes[k])
        line_boxes.append(lbx)
        grp: list[list[int]] = []
        for k in ks:
            if grp and boxes[k].x0 - max(boxes[g].x1 for g in grp[-1]) < 1.6 * lh:
                grp[-1].append(k)
            else:
                grp.append([k])
        for g in grp:
            comps = sorted((i for k in g for i in ch[k]), key=lambda i: x0s[i])
            runs: list[list] = []
            for i in comps:
                if runs and x0s[i] <= runs[-1][1] + 0.3 * lh:
                    runs[-1][1] = max(runs[-1][1], int(x1s[i]))
                    runs[-1][2].append(i)
                else:
                    runs.append([int(x0s[i]), int(x1s[i]), [i]])
            for chunk in _chunk(runs, lh):
                cs = [i for r in chunk for i in r[2]]
                pieces.append(Piece(box=cbox(cs), line=li, comps=cs))

    # 6. the rest
    for i in np.flatnonzero(valid & ~link):
        cx, cy = (x0s[i] + x1s[i]) / 2, cys[i]
        best, bd = None, None
        for p in pieces:
            b = p.box
            dx = max(b.x0 - cx, 0, cx - b.x1)
            dy = max(b.y0 - cy, 0, cy - b.y1)
            if dx > 0.4 * lh or dy > 0.6 * lh:
                continue
            if bd is None or dx + dy < bd:
                best, bd = p, dx + dy
        own = Box(x0=int(x0s[i]), y0=int(y0s[i]), x1=int(x1s[i]), y1=int(y1s[i]))
        if best is not None:
            best.comps.append(int(i))
            if not tall[i]:
                best.box = best.box.union(own)
        elif tall[i]:
            line_boxes.append(own)
            pieces.append(Piece(box=own, line=len(line_boxes) - 1, comps=[int(i)]))
    for p in pieces:
        p.scripts = _scripts(p.comps, stats, med)
    return HandLayout(pieces, line_boxes, lh, med, labels, stats)


def _chunk(runs: list[list], lh: float, max_aspect: float = 14.0) -> list[list[list]]:
    """Cut a long run of words at a wide word gap near the middle until each piece is
    at most max_aspect line-heights wide (TrOCR squeezes every piece into 384x384)."""
    width = runs[-1][1] - runs[0][0]
    if len(runs) < 2 or width <= max_aspect * lh:
        return [runs]
    gaps = [runs[k + 1][0] - runs[k][1] for k in range(len(runs) - 1)]
    mid = runs[0][0] + width / 2
    k = max(range(len(gaps)), key=lambda k: gaps[k] - 0.15 * abs(runs[k][1] - mid) / max(1.0, lh) * lh / 4)
    return _chunk(runs[:k + 1], lh, max_aspect) + _chunk(runs[k + 1:], lh, max_aspect)


def _scripts(comps: list[int], stats: np.ndarray, med: float) -> int:
    """Small marks sitting clearly above or below the line's middle (10⁴, Hg²⁺, H₂).
    Dots and commas are too small to count."""
    if len(comps) < 2:
        return 0
    c = np.asarray(comps)
    ys = stats[c, 1] + stats[c, 3] / 2.0
    hs = stats[c, 3].astype(float)
    big = hs >= 0.75 * med
    if not big.any():
        return 0
    mid = float(np.median(ys[big]))
    small = (hs >= 0.3 * med) & (hs < 0.75 * med) & (stats[c, 4] > 0.08 * med * med)
    return int((small & (np.abs(ys - mid) > 0.4 * med)).sum())

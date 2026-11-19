"""Snap a rough user-drawn box to the real content.

Users drag loose boxes. Typical problems on real papers:
  * the box slices through the line above/below (e.g. the lower half of a section heading)
    -> half-letters that OCR turns into garbage
  * the question number "1." sits just left of the box -> no number found
  * a column divider (solid or dotted) is inside the box -> read as "|"
  * letters cut at the box edge

snap_box() moves the edges to whole lines / whole words, drops dividers and pulls in the
question marker. It only DECIDES the box - the crop is still cut untouched from the original.
"""
from __future__ import annotations

import re

import cv2
import numpy as np

from .config import settings
from .schemas import Box


def _runs(on: np.ndarray, merge_gap: int = 0) -> list[tuple[int, int]]:
    """[start, end) runs of True, merging runs separated by <= merge_gap False values."""
    runs: list[list[int]] = []
    start = None
    for i, v in enumerate(np.append(on, False)):
        if v and start is None:
            start = i
        elif not v and start is not None:
            if runs and start - runs[-1][1] <= merge_gap:
                runs[-1][1] = i
            else:
                runs.append([start, i])
            start = None
    return [(a, b) for a, b in runs]


class _Ink:
    """Binarized context region around the box, with page-coordinate helpers."""

    def __init__(self, original: np.ndarray, box: Box, ctx: int):
        gray = original if original.ndim == 2 else cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
        H, W = gray.shape
        self.X0, self.Y0 = max(0, box.x0 - ctx), max(0, box.y0 - ctx)
        self.X1, self.Y1 = min(W, box.x1 + ctx), min(H, box.y1 + ctx)
        reg = gray[self.Y0:self.Y1, self.X0:self.X1]
        t, _ = cv2.threshold(reg, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        t = min(max(t, 90), 200)
        ink = (reg < t).astype(np.uint8)
        # drop single-pixel scanner speckle, keep dots of dotted lines
        n, lab, stats, _ = cv2.connectedComponentsWithStats(ink, 8)
        keep = stats[:, cv2.CC_STAT_AREA] >= 3
        keep[0] = False
        self.ink = keep[lab]
        self.W, self.H = W, H

    def sub(self, x0, y0, x1, y1) -> np.ndarray:
        x0, x1 = max(x0, self.X0), min(x1, self.X1)
        y0, y1 = max(y0, self.Y0), min(y1, self.Y1)
        if x1 <= x0 or y1 <= y0:
            return np.zeros((0, 0), bool)
        return self.ink[y0 - self.Y0:y1 - self.Y0, x0 - self.X0:x1 - self.X0]

    def row_bands(self, x0, x1, y0, y1, feather: bool = False) -> list[tuple[int, int]]:
        """Text-line bands (page y) over columns x0..x1. Rows need ink in >=1% of the width,
        so the dots of a divider or speckle don't bridge the white gaps between lines."""
        a = self.sub(x0, y0, x1, y1)
        if a.size == 0:
            return []
        prof = a.sum(axis=1)
        thr = max(3, int(0.01 * (x1 - x0)))
        ys = max(y0, self.Y0)
        out = []
        for s, e in _runs(prof >= thr, merge_gap=1):
            # grow over adjoining faint rows (bottoms of brackets, descenders, accents)
            for _ in range(30 if feather else 0):
                if s > 0 and prof[s - 1] > 0:
                    s -= 1
                else:
                    break
            for _ in range(30 if feather else 0):
                if e < len(prof) and prof[e] > 0:
                    e += 1
                else:
                    break
            if out and s <= out[-1][1]:
                out[-1] = (out[-1][0], max(out[-1][1], e))
            else:
                out.append((s, e))
        return [(s + ys, e + ys) for s, e in out]

    def col_groups(self, x0, x1, y0, y1, gap: int) -> list[tuple[int, int]]:
        a = self.sub(x0, y0, x1, y1)
        if a.size == 0:
            return []
        xs = max(x0, self.X0)
        return [(s + xs, e + xs) for s, e in _runs(a.sum(axis=0) > 0, merge_gap=gap)]


def _med_h(ink: _Ink, b: Box) -> float:
    bands = [e - s for s, e in ink.row_bands(b.x0, b.x1, b.y0, b.y1, feather=False) if s > b.y0 and e < b.y1]
    bands = [h for h in bands if h >= 6]
    return float(np.clip(np.median(bands), 12, 120)) if bands else 24.0


def _remove_dividers(ink: _Ink, b: Box, mh: float, notes: list[str]) -> Box:
    """A narrow vertical ink strip at the left/right edge, separated from the text by white
    space and continuing above/below the box, is a column rule - move the edge inside."""
    gap = int(0.6 * mh)
    groups = ink.col_groups(b.x0, b.x1, b.y0, b.y1, gap=max(2, int(0.3 * mh)))
    if len(groups) < 2:
        return b
    span = int(6 * mh)

    def is_rule(g):
        s, e = g
        if e - s > 0.5 * mh:
            return False
        above = ink.sub(s, b.y0 - span, e, b.y0)
        below = ink.sub(s, b.y1, e, b.y1 + span)
        outside = np.concatenate([above.any(axis=1), below.any(axis=1)]) if above.size + below.size else np.array([])
        return outside.size > 0 and outside.mean() >= 0.05

    x0, x1 = b.x0, b.x1
    last, prev = groups[-1], groups[-2]
    if last[1] >= b.x1 - mh and last[0] - prev[1] >= gap and is_rule(last):
        x1 = prev[1] + (last[0] - prev[1]) // 2
        notes.append("removed column divider on the right")
    first, nxt = groups[0], groups[1]
    if first[0] <= b.x0 + mh and nxt[0] - first[1] >= gap and is_rule(first):
        x0 = first[1] + (nxt[0] - first[1]) // 2
        notes.append("removed column divider on the left")
    return Box(x0=x0, y0=b.y0, x1=x1, y1=b.y1)


def _snap_vertical(ink: _Ink, b: Box, mh: float, notes: list[str], has_marker=None) -> Box:
    """has_marker(y0, y1) -> bool OCRs one line. A crossing line that starts with a question
    number is this question's first line at the TOP (keep it) but the NEXT question at the
    BOTTOM (cut it). Otherwise: keep the line if at least half of it is inside the box."""
    reach = int(2.5 * mh)
    bands = ink.row_bands(b.x0, b.x1, b.y0 - reach, b.y1 + reach)
    y0, y1 = b.y0, b.y1
    cut_top = cut_bottom = None                            # the trimmed bands, if any
    for s, e in bands:
        h = e - s
        graphic = h > 2.5 * mh
        need = 0.25 if graphic else 0.5
        if s < b.y0 < e:                                   # crosses the top edge
            keep = (e - b.y0) / h >= need
            if not graphic and has_marker is not None and has_marker(s, e):
                keep = True
            if not keep:
                y0 = e
                cut_top = (s, e)
                notes.append("trimmed a partial line above the question")
            elif s < y0:
                y0 = max(s, b.y0 - (int(15 * mh) if graphic else reach))
                notes.append("extended to the whole first line")
        if s < b.y1 < e:                                   # crosses the bottom edge
            keep = (b.y1 - s) / h >= need
            if keep and not graphic and has_marker is not None and has_marker(s, e):
                keep = False
                notes.append("left out the next question's first line")
            if not keep:
                y1 = s
                cut_bottom = (s, e)
                notes.append("trimmed a partial line below the question")
            elif e > y1:
                y1 = min(e, b.y1 + (int(15 * mh) if graphic else reach))
                notes.append("extended to the whole last line")
    trimmed_top = "trimmed a partial line above the question" in notes
    trimmed_bottom = "trimmed a partial line below the question" in notes
    inner = ink.row_bands(b.x0, b.x1, y0, y1)
    # A trimmed line leaves thin fragments behind (the tail of a 'y' under the line above,
    # accents over the line below). They are CONNECTED to the trimmed line's letters - unlike
    # e.g. a fraction's denominator, which is close but separate.
    if cut_top:
        while inner and _attached(ink, b.x0, b.x1, inner[0], cut_top):
            y0 = inner.pop(0)[1]
    if cut_bottom:
        while inner and _attached(ink, b.x0, b.x1, inner[-1], cut_bottom):
            y1 = inner.pop()[0]
    # never grow back into a trimmed line - including the tips of its tall letters / tails
    # that stick out of its core band (connected to it)
    floor = y0 if trimmed_top else -10 ** 9
    ceiling = y1 if trimmed_bottom else 10 ** 9
    # (clamped to the kept lines' cores, in case a letter of the trimmed line touches one of ours)
    if cut_top:
        floor = max(floor, _letters_extent(ink, b.x0, b.x1, cut_top, mh)[1])
        if inner:
            floor = min(floor, inner[0][0])
        y0 = max(y0, floor)
    if cut_bottom:
        ceiling = min(ceiling, _letters_extent(ink, b.x0, b.x1, cut_bottom, mh)[0] - 1)
        if inner:
            ceiling = max(ceiling, inner[-1][1])
        y1 = min(y1, ceiling)
    # shrink to the ink actually inside (drops empty margins the user left)
    if inner:
        y0, y1 = max(y0, inner[0][0]), min(y1, inner[-1][1])
    # the line cores above exclude faint rows (bracket bottoms, descenders, accents):
    # grow the final edges over adjoining inked rows, at most 0.3 of a line
    limit = max(2, int(0.3 * mh))
    for _ in range(limit):
        a = ink.sub(b.x0, y0 - 1, b.x1, y0)
        if y0 - 1 >= floor and a.size and a.any():
            y0 -= 1
        else:
            break
    for _ in range(limit):
        a = ink.sub(b.x0, y1, b.x1, y1 + 1)
        if y1 + 1 <= ceiling and a.size and a.any():
            y1 += 1
        else:
            break
    return Box(x0=b.x0, y0=y0, x1=b.x1, y1=y1) if y1 > y0 else b


def _attached(ink: _Ink, x0: int, x1: int, band: tuple[int, int], cut: tuple[int, int]) -> bool:
    """True if most ink of `band` belongs to letters that also reach into the trimmed line."""
    top, bottom = min(band[0], cut[0]), max(band[1], cut[1])
    a = ink.sub(x0, top, x1, bottom)
    if a.size == 0:
        return False
    n, lab = cv2.connectedComponents(a.astype(np.uint8), connectivity=8)
    in_band = lab[band[0] - max(top, ink.Y0):band[1] - max(top, ink.Y0)]
    in_cut = lab[cut[0] - max(top, ink.Y0):cut[1] - max(top, ink.Y0)]
    ids, counts = np.unique(in_band[in_band > 0], return_counts=True)
    if ids.size == 0:
        return False
    shared = np.isin(ids, np.unique(in_cut[in_cut > 0]))
    return counts[shared].sum() >= 0.6 * counts.sum()


def _letters_extent(ink: _Ink, x0: int, x1: int, band: tuple[int, int], mh: float) -> tuple[int, int]:
    """(top, bottom) page rows of all letters that touch `band` - a line's real extent
    including ascender tips and descender tails outside its core rows."""
    reach = int(0.6 * mh)
    top = max(band[0] - reach, ink.Y0)
    a = ink.sub(x0, top, x1, band[1] + reach)
    if a.size == 0:
        return band
    n, lab = cv2.connectedComponents(a.astype(np.uint8), connectivity=8)
    core = lab[band[0] - top:band[1] - top]
    ids = np.unique(core[core > 0])
    if ids.size == 0:
        return band
    rows = np.nonzero(np.isin(lab, ids).any(axis=1))[0]
    return int(rows.min()) + top, int(rows.max()) + top + 1


def _snap_horizontal(ink: _Ink, b: Box, mh: float, include_marker: bool, notes: list[str]) -> Box:
    lock_right = "removed column divider on the right" in notes
    lock_left = "removed column divider on the left" in notes
    word_gap = max(2, int(0.5 * mh))
    reach = int(3 * mh)
    x0, x1 = b.x0, b.x1
    # characters cut at the edges -> extend to the end of the word
    for s, e in ink.col_groups(b.x0 - reach, b.x1 + reach, b.y0, b.y1, gap=word_gap):
        if not lock_right and s < b.x1 < e and e - b.x1 <= reach:
            x1 = max(x1, e)
            notes.append("extended to letters cut on the right")
        if not lock_left and s < b.x0 < e and b.x0 - s <= reach:
            x0 = min(x0, s)
            notes.append("extended to letters cut on the left")
    # question number just left of the box, on the first line
    if include_marker and not lock_left:
        lines = ink.row_bands(x0, x1, b.y0, b.y1, feather=False)    # single lines, not merged
        if lines:
            ls, le = lines[0]
            look = int(4 * mh)
            win0 = x0 - look - int(2 * mh)
            groups = [g for g in ink.col_groups(win0, x0, ls, le, gap=word_gap) if g[1] <= x0]
            # drop column rules; they mark the column boundary
            rules = [g for g in groups if _is_vertical_rule(ink, g, ls, le, mh)]
            groups = [g for g in groups if g not in rules]
            if groups:
                g = groups[-1]                                 # the ink closest to the box
                farther = [f for f in groups[:-1]
                           if not any(r[0] >= f[1] and r[1] <= g[0] for r in rules)]  # same column only
                close = g[1] >= x0 - 3 * mh
                small = g[1] - g[0] <= 2.5 * mh                # "1.", "12.", "Q12."
                not_cut = g[0] > win0 + 2                      # not text running out of the window
                isolated = not farther or g[0] - farther[-1][1] >= 0.6 * mh
                if close and small and not_cut and isolated:
                    x0 = g[0]
                    notes.append("included the question number on the left")
    return Box(x0=x0, y0=b.y0, x1=x1, y1=b.y1)


def _is_vertical_rule(ink: _Ink, g: tuple[int, int], ls: int, le: int, mh: float) -> bool:
    """A column rule (solid or dotted) is narrow and continues far above and below the line:
    split +-6 lines into half-line chunks; a rule has ink in almost all of them."""
    if g[1] - g[0] > 0.4 * mh:
        return False
    span, step = int(6 * mh), max(4, int(0.5 * mh))
    chunks = []
    for y in list(range(ls - span, ls, step)) + list(range(le, le + span, step)):
        a = ink.sub(g[0], y, g[1], y + step)
        if a.size:
            chunks.append(bool(a.any()))
    return len(chunks) >= 6 and sum(chunks) / len(chunks) >= 0.7


def _pad_into_white(ink: _Ink, b: Box, m: int) -> Box:
    def clear(x0, y0, x1, y1):
        a = ink.sub(x0, y0, x1, y1)
        return a.size == 0 or not a.any()

    def grow(k_max, test):
        for k in range(k_max, 0, -1):
            if test(k):
                return k
        return 0

    t = grow(m, lambda k: clear(b.x0, b.y0 - k, b.x1, b.y0))
    bt = grow(m, lambda k: clear(b.x0, b.y1, b.x1, b.y1 + k))
    lf = grow(m, lambda k: clear(b.x0 - k, b.y0, b.x0, b.y1))
    rt = grow(m, lambda k: clear(b.x1, b.y0, b.x1 + k, b.y1))
    return Box(x0=b.x0 - lf, y0=b.y0 - t, x1=b.x1 + rt, y1=b.y1 + bt)


def _starts_with_marker(strip: np.ndarray) -> bool:
    """OCR one text line (~0.1 s) and check whether it starts with a question number."""
    if strip.size == 0:
        return False
    from .ocr.tesseract import TesseractEngine
    from .segment import match_marker
    try:
        text = TesseractEngine(psm=7).read(strip).text(0.0).strip()
    except Exception:
        return False
    # The strip may start with a divider ("|"), a speck or the tail of the neighbouring
    # column's words ("TiC : 10. In an ...") - accept a number among the first tokens.
    tokens = text.split()
    for i in range(min(4, len(tokens))):
        rest = " ".join(tokens[i:])
        if re.match(r"^[0-9ilI|!oOQ]", rest) and match_marker(rest, None, False) is not None:
            return True
    return False


def snap_box(original: np.ndarray, box: Box, include_marker: bool = True) -> tuple[Box, list[str]]:
    """Return (snapped box, notes). Falls back to the user's box if anything looks wrong."""
    H, W = original.shape[:2]
    user = box.clip(W, H)
    if user.w < 8 or user.h < 8:
        return user, []
    try:
        ink = _Ink(original, user, ctx=max(200, int(0.3 * max(user.w, user.h))))
        mh = _med_h(ink, user)
        notes: list[str] = []
        cache: dict[tuple[int, int], bool] = {}

        def has_marker(y0: int, y1: int) -> bool:
            if (y0, y1) not in cache:
                strip = original[max(0, y0 - 2):min(H, y1 + 2), max(0, b.x0 - int(4 * mh)):b.x1]
                cache[(y0, y1)] = _starts_with_marker(strip)
            return cache[(y0, y1)]

        b = _remove_dividers(ink, user, mh, notes)
        b = _snap_vertical(ink, b, mh, notes, has_marker)
        b = _snap_horizontal(ink, b, mh, include_marker, notes)
        b = _snap_vertical(ink, b, mh, notes, has_marker)
        b = _pad_into_white(ink, b, int(settings()["layout"]["question_margin_px"]) + 4)
        b = b.clip(W, H)
    except Exception:
        return user, []
    if b.w < 8 or b.h < 8 or b.area < 0.25 * user.area or b.area > 3 * user.area:
        return user, []
    return b, list(dict.fromkeys(notes))

"""Analyze a crop the USER drew (also used by auto mode).

For one box on the original page:
  1. OCR the original pixels once (pass 1) -> words, lines, text height
  2. Find diagrams / tables INSIDE the box (on a binarized working copy of the crop)
  3. If diagrams were found, OCR again with the diagram areas blanked (OCR input only)
     so diagram labels don't leak into the question text; labels are kept for the diagram
  4. Printed vs handwritten per line -> re-read with the handwriting engine if needed
  5. Image options: diagrams sitting at the option labels when options have no text
  6. Question number from the marker on the first line
The saved crop image is never modified - all cleaning happens on throwaway copies.
"""
from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field

import cv2
import numpy as np

from .classify import analyze_type
from .config import settings
from .crop import cut
from .ink import classify_ink
from .layout.heuristic import detect_graphics
from .ocr import read_region
from .ocr.base import OCRResult
from .parse import parse_question
from .preprocess import make_working_copy
from .quality import crop_quality
from .schemas import Box, Project, QuestionRecord, QuestionType, RegionKind
from .segment import match_marker

OPTION_LABEL = re.compile(r"^\(?([A-Da-d1-4])[.)]\)?$|^\(([A-Da-d1-4])\)")
OPTION_START = re.compile(r"^\(?[A-Da-d1-4][.)]\s")


@dataclass
class FoundGraphic:
    kind: RegionKind
    box: Box                     # original-page coordinates (with margin)
    labels: str = ""             # text read inside the drawing (A, B, 30°, ?)
    edge_cut: bool = False       # drawing reaches the edge of the user's box -> may be cut off


@dataclass
class PartAnalysis:
    page: int
    box: Box
    image: np.ndarray
    text: str
    confidence: float
    source_type: str
    blur: float
    quality_flag: str
    q_no: int | None
    graphics: list[FoundGraphic] = field(default_factory=list)
    edge_warnings: list[str] = field(default_factory=list)
    layout: str = ""             # text laid out as on the page, {{fig:n}} = graphics[n-1]
    latex: str = ""              # the same text with formulas as LaTeX $...$ ("" = same as text)
    layout_latex: str = ""       # the same layout with formulas as LaTeX


def analyze_part(original: np.ndarray, page: int, box: Box, find_graphics: bool = True,
                 script: str = "auto") -> PartAnalysis:
    """script: auto | printed | handwritten - what the paper is written in (set at upload).
    The box is used exactly as the user drew it: nothing is moved or extended, the app only
    reads what is inside it."""
    from .ocr import get_engine
    from .paper import paper_only
    cfg = settings()
    unreadable = cfg["ocr"]["unreadable_conf"]
    # a phone photo: desk / keyboard around the sheet is not ink (analysis copy only - the
    # saved crop is cut from the untouched page by the caller)
    original = paper_only(original)
    box = box.clip(original.shape[1], original.shape[0])
    img = cut(original, box)
    blur, flag = crop_quality(img)

    # pass 1, fast (Tesseract): word boxes to find printed diagrams, and print vs handwriting
    fast = cfg["ocr"]["printed_engine"]
    res = get_engine("tesseract" if fast == "glm" else fast).read(img)
    # Tesseract reads print at 0.8+; under 0.35 it is (almost always) handwriting
    hand = script == "handwritten" or (script == "auto" and (res.confidence < 0.35 or looks_handwritten(img, res.confidence)))
    if cfg["ocr"]["handwritten_engine" if hand else "printed_engine"] == "glm":
        engine = get_engine("glm")
        if engine.name == "glm":                             # else: model not installed -> below
            return _analyze_vlm(engine, original, page, box, img, blur, flag, find_graphics, hand, res)
    if hand:
        out = _analyze_handwritten(original, page, box, img, blur, flag, find_graphics)
        if out is not None:                                  # None: handwriting engine not installed
            return out
    lines = _lines(res)
    wc = make_working_copy(img, deskew=False)                # detection copy only
    graphics, local_boxes = (_graphics_printed(original, box, img, res, wc) if find_graphics else ([], []))

    # Ink type per line -> route to the right engine.
    kinds = []
    for ws in lines.values():
        lb = ws[0].box
        for w in ws[1:]:
            lb = lb.union(w.box)
        kinds.append(classify_ink(wc.binary[lb.y0:lb.y1, lb.x0:lb.x1])[0])
    source = Counter(kinds).most_common(1)[0][0] if kinds else "printed"
    if len(set(kinds)) > 1 and min(Counter(kinds).values()) >= 2:
        source = "mixed"

    diagrams = [g for g in graphics if g.kind == RegionKind.diagram]
    engine_changes = source == "handwritten" and cfg["ocr"]["handwritten_engine"] != cfg["ocr"]["printed_engine"]
    if diagrams or engine_changes:
        ocr_in = img.copy()
        for g in diagrams:                                    # blank drawings in the OCR input only
            x0, y0 = max(0, g.box.x0 - box.x0), max(0, g.box.y0 - box.y0)
            x1, y1 = min(img.shape[1], g.box.x1 - box.x0), min(img.shape[0], g.box.y1 - box.y0)
            ocr_in[y0:y1, x0:x1] = 255
        # ...but never blank the question's own sentences: put back every word of a real
        # text line (3+ confident words) that a drawing's box happened to overlap, and every
        # option label - "(a)" right beside an image option is part of the question
        for ws in lines.values():
            sentence = sum(x.conf > 0.5 for x in ws) >= 3
            for w in ws:
                if w.conf > 0.5 and (sentence or OPTION_LABEL.match(w.text)):
                    b = w.box.pad(2).clip(img.shape[1], img.shape[0])
                    ocr_in[b.y0:b.y1, b.x0:b.x1] = img[b.y0:b.y1, b.x0:b.x1]
        res = read_region(ocr_in, "handwritten" if source == "handwritten" else "printed")

    text = res.text(unreadable)
    _mark_option_images(res, box, graphics, text)
    if _labels_inside_options(res, img, box, graphics):
        text = res.text(unreadable)
    q_no = None
    for ln in text.strip().splitlines()[:2]:               # marker is on the first line (or second)
        q_no = match_marker(ln, None, False)
        if q_no is not None:
            break
    if q_no is None:
        q_no = _reread_marker(img, res)
    warnings = _edge_junk(res, img.shape[0])
    if q_no is not None:
        later = text.strip().splitlines()[2:]
        if any(match_marker(ln, q_no + 1, False) == q_no + 1 for ln in later):
            warnings.append(f"box also contains question {q_no + 1} - make it smaller")
    layout = compose_layout(res, local_boxes, unreadable, _marker_fix(img, res, q_no))
    return PartAnalysis(page=page, box=box, image=img, text=text, confidence=res.confidence,
                        source_type=source, blur=blur, quality_flag=flag, q_no=q_no,
                        graphics=graphics, edge_warnings=warnings, layout=layout)


def _lines(res: OCRResult) -> dict[int, list]:
    lines: dict[int, list] = {}
    for w in res.words:
        lines.setdefault(w.line_id, []).append(w)
    return lines


def _graphics_printed(original: np.ndarray, box: Box, img: np.ndarray, res: OCRResult,
                      wc) -> tuple[list[FoundGraphic], list[Box]]:
    """Diagrams / tables in a printed crop: ink that is not a confident text line."""
    cfg = settings()
    lines = _lines(res)
    heights = [w.box.h for w in res.words if w.conf > 0.5 and w.box.h > 3]
    med_h = float(np.median(heights)) if heights else 24.0
    graphics: list[FoundGraphic] = []
    local_boxes: list[Box] = []                              # same graphics, crop coordinates
    text_boxes = [w.box for ws in lines.values() if sum(x.conf > 0.5 for x in ws) >= 3
                  for w in ws if w.conf > 0.5]
    page_area = original.shape[0] * original.shape[1]
    min_area = max(cfg["layout"]["diagram_min_area_frac"] * page_area, (5 * med_h) ** 2)
    margin = cfg["layout"]["diagram_margin_px"]
    found = detect_graphics(wc.binary, res.words, med_h, text_boxes, min_area_px=min_area)
    trimmed = _trim_figure_boxes([g.box for g in found], text_boxes, wc.binary > 0)
    H_, W_ = img.shape[:2]
    for g, local in zip(found, trimmed):
        # cut off = the drawing's INK reaches the crop border (not just a label's box)
        bw = wc.binary[local.y0:local.y1, local.x0:local.x1] > 0
        edge = bool(bw.size) and (
            (local.y0 <= 1 and bw[:2].any()) or (local.y1 >= H_ - 1 and bw[-2:].any())
            or (local.x0 <= 1 and bw[:, :2].any()) or (local.x1 >= W_ - 1 and bw[:, -2:].any()))
        # margin only into white: a fixed pad catches the descenders of the line above or
        # the next figure's labels, which then show up as slivers in the figure image
        padded = _pad_into_white(wc.binary, local, margin)
        pb = Box(x0=padded.x0 + box.x0, y0=padded.y0 + box.y0, x1=padded.x1 + box.x0, y1=padded.y1 + box.y0)
        labels = " ".join(w.text for w in sorted(g.labels, key=lambda w: (w.box.y0 // 10, w.box.x0)))
        graphics.append(FoundGraphic(kind=g.kind, box=pb.clip(original.shape[1], original.shape[0]),
                                     labels=labels, edge_cut=edge))
        local_boxes.append(local)
    return graphics, local_boxes


def _graphics_hand(original: np.ndarray, box: Box, img: np.ndarray) -> tuple[list[FoundGraphic], list[Box], list[Box]]:
    """Hand-drawn figures: found on the ink that is NOT part of a text line, so cursive words
    are never mistaken for a diagram. -> (graphics, crop-coordinate boxes, text line boxes)"""
    from .ocr.handseg import binarize, segment
    cfg = settings()
    bw = binarize(img)
    lay = segment(bw)
    lh, ch = lay.line_h, lay.char_h
    # a text piece: one line tall and a few letters wide
    texty = [p for p in lay.pieces if p.box.h <= 1.6 * lh and p.box.w >= 3.5 * lh]
    text_boxes = [p.box for p in texty]
    graphics: list[FoundGraphic] = []
    local_boxes: list[Box] = []
    if not lay.pieces:
        return graphics, local_boxes, text_boxes
    det = bw.copy()                                          # drawings only: text lines removed
    text_comps = [i for p in texty for i in p.comps]
    if text_comps:
        det[np.isin(lay.labels, text_comps)] = 0
    found = detect_graphics(det, [], ch, text_boxes, min_area_px=(2.2 * lh) ** 2)
    trimmed = _trim_figure_boxes([g.box for g in found], text_boxes, det > 0)
    H_, W_ = img.shape[:2]
    margin = cfg["layout"]["diagram_margin_px"]
    for g, local in zip(found, trimmed):
        sub = det[local.y0:local.y1, local.x0:local.x1] > 0
        edge = bool(sub.size) and (
            (local.y0 <= 1 and sub[:2].any()) or (local.y1 >= H_ - 1 and sub[-2:].any())
            or (local.x0 <= 1 and sub[:, :2].any()) or (local.x1 >= W_ - 1 and sub[:, -2:].any()))
        padded = _pad_into_white(bw, local, margin)
        pb = Box(x0=padded.x0 + box.x0, y0=padded.y0 + box.y0, x1=padded.x1 + box.x0, y1=padded.y1 + box.y0)
        graphics.append(FoundGraphic(kind=g.kind, box=pb.clip(original.shape[1], original.shape[0]),
                                     edge_cut=edge))
        local_boxes.append(local)
    return graphics, local_boxes, text_boxes


def _analyze_vlm(engine, original: np.ndarray, page: int, box: Box, img: np.ndarray, blur: float,
                 flag: str, find_graphics: bool, hand: bool, res: OCRResult) -> PartAnalysis:
    """GLM-OCR reads the crop (printed or handwritten, formulas and tables included).
    Drawings are found first and saved as their own locked images; the model reads the
    text around them with the drawings blanked (OCR input only), in one pass, and each
    drawing's marker is placed where it sits on the page."""
    from .ocr.glmocr import char_height
    graphics: list[FoundGraphic] = []
    local_boxes: list[Box] = []
    ch = char_height(img)
    if hand:
        rules = lambda: _graphics_hand(original, box, img)[:2]  # noqa: E731
    else:
        rules = lambda: _graphics_printed(original, box, img, res, make_working_copy(img, deskew=False))  # noqa: E731
    # the question's own sentences are never blanked or cut into a figure: on print,
    # Tesseract's confident lines (3+ words) and option labels mark them
    sentences, labels = ([], []) if hand else _sentence_boxes(res, img)
    if find_graphics:
        graphics, local_boxes = _graphics_model(original, box, img, ch, rules, add_rules=not hand,
                                                text_boxes=sentences)
    text, layout, conf, notes, latex, layout_latex = read_crop(engine, img, box, graphics, local_boxes, ch,
                                                             sentences + labels)
    q_no = find_q_no(text)
    warnings = notes + cut_edges(img, ch)
    if q_no is not None and contains_next(text, q_no):
        warnings.append(f"box also contains question {q_no + 1} - make it smaller")
    return PartAnalysis(page=page, box=box, image=img, text=text, confidence=conf,
                        source_type="handwritten" if hand else "printed", blur=blur, quality_flag=flag,
                        q_no=q_no, graphics=graphics, edge_warnings=warnings, layout=layout,
                        latex=latex, layout_latex=layout_latex)


def _sentence_boxes(res: OCRResult, img: np.ndarray) -> tuple[list[Box], list[Box]]:
    """Printed question text (crop coordinates) -> (sentence word boxes, option label boxes).
    A sentence: a line with 3+ confident real words (2+ letters) - not a row of graph labels
    like "(1) v  (2) v"."""
    sent, labels = [], []
    for ws in _lines(res).values():
        sentence = sum(x.conf > 0.5 and len(re.sub(r"[^A-Za-z]", "", x.text)) >= 2 for x in ws) >= 3
        for w in ws:
            b = w.box.pad(2).clip(img.shape[1], img.shape[0])
            if w.conf > 0.5 and sentence:
                sent.append(b)
            elif w.conf > 0.5 and OPTION_LABEL.match(w.text):
                labels.append(b)
    return sent, labels


def _graphics_model(original: np.ndarray, box: Box, img: np.ndarray, ch: float,
                    rules, add_rules: bool = True, text_boxes: list[Box] = ()) -> tuple[list[FoundGraphic], list[Box]]:
    """Figures from the trained layout model (layout/docmodel.py), plus any figure the ink
    rules find where the model found nothing. Without the model: the ink rules alone."""
    from .layout.docmodel import find_figures
    from .ocr.handseg import binarize
    line_h = 2.0 * ch if ch > 0 else 24.0
    found = find_figures(img, line_h)
    if found is None:
        return rules()
    cfg = settings()
    H, W = img.shape[:2]
    bw = binarize(img)
    n, labels, st, _ = cv2.connectedComponentsWithStats((bw > 0).astype(np.uint8), 8)
    graphics: list[FoundGraphic] = []
    local_boxes: list[Box] = []
    kinds, tights = [], []
    # every stroke belongs to the detected drawing nearest to it (an enlarged box must not
    # take the label of the drawing below)
    cx = st[:, 0] + st[:, 2] / 2.0
    cy = st[:, 1] + st[:, 3] / 2.0
    dist = np.stack([np.maximum(np.maximum(fb.x0 - cx, cx - fb.x1), 0) + np.maximum(np.maximum(fb.y0 - cy, cy - fb.y1), 0)
                     for _, fb in found]) if found else np.zeros((0, n))
    nearest = dist.argmin(axis=0) if found else np.zeros(n, int)
    for k, (kind, b) in enumerate(found):
        # whole strokes: a bond or label the model's box clips (at least half inside) is taken
        # in, and so is a label hugging the drawing (an axis "v" just past the arrow tip)
        if kind != RegionKind.table:
            m = int(0.6 * line_h)
            b = Box(x0=b.x0 - m, y0=b.y0 - m, x1=b.x1 + m, y1=b.y1 + m).clip(W, H)
        ids = np.unique(labels[b.y0:b.y1, b.x0:b.x1])
        tight = None
        for i in ids[ids > 0]:
            cb = Box(x0=int(st[i, 0]), y0=int(st[i, 1]), x1=int(st[i, 0] + st[i, 2]), y1=int(st[i, 1] + st[i, 3]))
            inside = max(0, min(cb.x1, b.x1) - max(cb.x0, b.x0)) * max(0, min(cb.y1, b.y1) - max(cb.y0, b.y0))
            if inside >= 0.5 * max(1, cb.w * cb.h) and nearest[i] == k:
                tight = cb if tight is None else tight.union(cb)
        if tight is not None and tight.w >= 6 and tight.h >= 6:
            kinds.append(kind)
            tights.append(tight)
    # a sentence above / below poking into a figure's box: the figure ends before it
    tights = _trim_figure_boxes(tights, list(text_boxes), bw > 0) if text_boxes else tights
    for kind, tight in zip(kinds, tights):
        sub = bw[tight.y0:tight.y1, tight.x0:tight.x1] > 0
        edge = (tight.y0 <= 1 and sub[:2].any()) or (tight.y1 >= H - 1 and sub[-2:].any()) \
            or (tight.x0 <= 1 and sub[:, :2].any()) or (tight.x1 >= W - 1 and sub[:, -2:].any())
        padded = _pad_into_white(bw, tight, cfg["layout"]["diagram_margin_px"])
        pb = Box(x0=padded.x0 + box.x0, y0=padded.y0 + box.y0, x1=padded.x1 + box.x0, y1=padded.y1 + box.y0)
        graphics.append(FoundGraphic(kind=kind, box=pb.clip(original.shape[1], original.shape[0]), edge_cut=bool(edge)))
        local_boxes.append(tight)
    try:
        extra_g, extra_b = rules() if add_rules else ([], [])
    except Exception:
        extra_g, extra_b = [], []
    for g, lb in zip(extra_g, extra_b):
        big = lb.w * lb.h > 0.5 * W * H
        hit = any(max(0, min(lb.x1, m.x1) - max(lb.x0, m.x0)) * max(0, min(lb.y1, m.y1) - max(lb.y0, m.y0))
                  > 0.2 * max(1, min(lb.w * lb.h, m.w * m.h)) for m in local_boxes)
        if not big and not hit:
            graphics.append(g)
            local_boxes.append(lb)
    return graphics, local_boxes


BARE_NO = re.compile(r"^\s*[\[(]?(?:Q\.?\s*)?(\d{1,3})\s*[.):\]]?\s+(?=[A-Za-z(])")


def find_q_no(text: str) -> int | None:
    """The question number: a marker at the start of one of the first lines ("2)", "Q.12",
    "7]"). A bare "1 In the following..." counts too: the user's box starts at the question.
    A few lines are tried because a box may start with the end of the previous question."""
    lines = [x for x in text.strip().splitlines() if x.strip()][:4]
    for ln in lines:
        n = match_marker(ln, None, False)
        if n is not None:
            return n
    m = BARE_NO.match(lines[0]) if lines else None
    return int(m.group(1)) if m and 0 < int(m.group(1)) <= 200 else None


def contains_next(text: str, q_no: int) -> bool:
    """Does the box run into question q_no + 1? Not when that number is an item of a numbered
    list inside the question ("1. Reaction ... 2. ... 3. ..." in a match-the-column)."""
    later = text.strip().splitlines()[2:]
    nums = [match_marker(ln, None, False) for ln in later]
    if q_no + 1 not in nums:
        return False
    items = set(n for n in nums if n is not None)
    return not (1 in items or len(items) >= 3)


OPT_TOKEN = re.compile(r"(?:(?<=\s)|^)(\(?[A-Da-d1-4][.)])(?=\s|$)")


def read_crop(engine, img: np.ndarray, box: Box, graphics: list[FoundGraphic], local_boxes: list[Box],
              ch: float, protect: list[Box] = ()) -> tuple[str, str, float, list[str], str, str]:
    """Read the whole crop in ONE pass with the drawings blanked (OCR input only), then put a
    {{fig:n}} marker where each drawing sits.
    -> (text, layout, confidence, notes, latex text, latex layout) - latex: formulas as $...$.
    One pass on purpose: reading the crop in bands, the model invented the half of a line a
    band edge cut through ("B) Q = 1" on a tilted photo). Tables are not blanked - their
    text is read - and the table image is placed after it."""
    from .ocr.glmocr import has_ink
    H, W = img.shape[:2]
    ocr_in = img.copy()
    for g in graphics:
        if g.kind != RegionKind.table:
            x0, y0 = max(0, g.box.x0 - box.x0), max(0, g.box.y0 - box.y0)
            x1, y1 = min(W, g.box.x1 - box.x0), min(H, g.box.y1 - box.y0)
            ocr_in[y0:y1, x0:x1] = 255
    for b in protect:                        # ...but never the question's own sentences
        ocr_in[b.y0:b.y1, b.x0:b.x1] = img[b.y0:b.y1, b.x0:b.x1]
    if not has_ink(ocr_in):
        t, tl, conf, looped = "", "", 1.0, False
    elif hasattr(engine, "read_text_both"):
        t, tl, conf, looped = engine.read_text_both(ocr_in, ch)
    else:
        t, conf, looped = engine.read_text(ocr_in, ch)
        tl = t
    notes = ["the reader repeated itself on this crop - check the text"] if looped else []
    lines = [ln.rstrip() for ln in t.strip("\n").splitlines()] if t.strip() else []
    lat = [ln.rstrip() for ln in tl.strip("\n").splitlines()] if tl.strip() else []
    if len(lat) != len(lines):
        lat = list(lines)                         # never mis-aligned: plain is always right
    if graphics:
        layout, layout_l = place_figures(lines, ocr_in, graphics, local_boxes, twin=lat)
    else:
        layout, layout_l = list(lines), list(lat)
    text = "\n".join(lines).strip("\n")
    return (text, "\n".join(layout).strip("\n"), round(conf if text else 1.0, 4), notes,
            "\n".join(lat).strip("\n"), "\n".join(layout_l).strip("\n"))


def _visual_rows(boxes: list[Box]) -> list[list]:
    """Boxes side by side (overlapping in height) form one row: [[y0, y1, [indexes]], ...]."""
    rows: list[list] = []
    for i in sorted(range(len(boxes)), key=lambda k: boxes[k].y0):
        b = boxes[i]
        r = next((r for r in rows if min(b.y1, r[1]) - max(b.y0, r[0]) > 0.4 * min(b.h, r[1] - r[0])), None)
        if r is None:
            rows.append([b.y0, b.y1, [i]])
        else:
            r[0], r[1] = min(r[0], b.y0), max(r[1], b.y1)
            r[2].append(i)
    return sorted(rows, key=lambda r: r[0])


def place_figures(lines: list[str], ocr_in: np.ndarray, graphics: list[FoundGraphic],
                  local_boxes: list[Box], twin: list[str] | None = None):
    """Layout lines: the text lines with each figure's marker at its height. The model gives
    no positions, so line k is placed at the k-th visual text row of the (blanked) crop,
    scaled when the counts differ. A row of drawings, each beside its own option label with
    nothing else written ("A)  [drawing]   B)  [drawing]"), is a row of image options: each
    marker goes right after its label.
    twin: the same lines with formulas as LaTeX - given, it gets exactly the same markers
    (decided on the plain lines) and (plain layout, latex layout) is returned."""
    if twin is not None and len(twin) != len(lines):
        twin = None                              # never mis-align: the twin falls back to plain
    if twin is None:
        res = _place_figures(lines, ocr_in, graphics, local_boxes, [list(lines)])
        return res[0]
    return tuple(_place_figures(lines, ocr_in, graphics, local_boxes, [list(lines), list(twin)]))


def _place_figures(lines: list[str], ocr_in: np.ndarray, graphics: list[FoundGraphic],
                   local_boxes: list[Box], copies: list[list[str]]) -> list[list[str]]:
    from .ocr.handseg import binarize, segment
    lay = segment(binarize(ocr_in))
    trows = _visual_rows(lay.line_boxes)
    filled = [k for k, ln in enumerate(lines) if ln.strip()]
    est: dict[int, float] = {}
    for r, k in enumerate(filled):
        if trows:
            row = trows[min(len(trows) - 1, int(r * len(trows) / len(filled)))]
            est[k] = (row[0] + row[1]) / 2
        else:
            est[k] = 0.0
    out = copies[0]                              # decisions are made on the plain lines
    after: dict[int, list[str]] = {}
    # Option labels standing alone on their line ("A)") had a drawing as their content.
    # They pair up, in reading order, with the LAST drawings of the crop (options come at the
    # end of a question; a reaction scheme above them is a diagram of the question).
    alone = [k for k in filled if re.fullmatch(r"\s*\(?[A-Da-d1-4][.)]\s*", out[k])]
    order = [i for r in _visual_rows(local_boxes) for i in sorted(r[2], key=lambda j: local_boxes[j].x0)]
    order = order[len(order) - len(alone):] if len(order) >= len(alone) else []
    if len(alone) >= 2 and order and all(graphics[i].kind != RegionKind.table for i in order):
        for k, i in zip(alone, order):
            graphics[i].kind = RegionKind.option_image
            for c in copies:
                c[k] = f"{c[k].strip()} {FIG % (i + 1)}"
        return copies
    for y0, y1, ids in _visual_rows(local_boxes):
        ids = sorted(ids, key=lambda i: local_boxes[i].x0)
        span = [k for k in filled if y0 <= est[k] <= y1]
        text = "\n".join(out[k] for k in span)
        labels = list(OPT_TOKEN.finditer(text))
        textless = len(re.sub(r"\s", "", OPT_TOKEN.sub("", text))) <= 3 * len(labels)
        if len(ids) >= 2 and labels and len(labels) == len(ids) and textless:
            for i in ids:
                graphics[i].kind = RegionKind.option_image
            for c in copies:
                it = iter(ids)

                def put(m, it=it):
                    i = next(it, None)
                    return m.group(1) if i is None else f"{m.group(1)} {FIG % (i + 1)}"
                for k in span:
                    c[k] = OPT_TOKEN.sub(put, c[k])
            continue
        cy = (y0 + y1) / 2
        pos = span[-1] if span else max((k for k in filled if est[k] < cy), default=-1)
        after.setdefault(pos, []).append("   ".join(FIG % (i + 1) for i in ids))
    result = []
    for c in copies:
        res = list(after.get(-1, []))
        for k, ln in enumerate(c):
            res.append(ln)
            res += after.get(k, [])
        result.append(res)
    return result


def cut_edges(img: np.ndarray, ch: float) -> list[str]:
    """Writing cut by the user's box edge: letters touching the top / bottom / side. The box is
    never changed by the app - this only tells the user, so they can redraw it larger."""
    from .ocr.handseg import binarize
    bw = (binarize(img) > 0).astype(np.uint8)
    H, W = bw.shape[:2]
    n, _, st, _ = cv2.connectedComponentsWithStats(bw, 8)
    ch = max(ch, 6.0)
    marks = [s for s in st[1:] if s[3] >= 0.4 * ch and s[4] >= 10 and s[2] < 0.6 * W and s[3] < 0.6 * H]
    top = sum(int(s[2]) for s in marks if s[1] <= 0)
    bottom = sum(int(s[2]) for s in marks if s[1] + s[3] >= H)
    left = sum(int(s[3]) for s in marks if s[0] <= 0)
    right = sum(int(s[3]) for s in marks if s[0] + s[2] >= W)
    out = []
    for name, v in (("top", top), ("bottom", bottom), ("left", left), ("right", right)):
        if v >= 1.5 * ch:
            out.append(f"your box cuts through writing at the {name} edge - text there may be missing; "
                       f"redraw the box (M) a little larger")
    return out


def looks_handwritten(img: np.ndarray, tess_conf: float) -> bool:
    """Handwriting vs print for one crop. Tesseract reads print at 0.85+ and handwriting at
    0.2-0.45; handwritten letters also vary in height far more (height_cv ~0.45 vs ~0.2).
    The stroke classifier alone misses thick, even pens, so the signals are combined."""
    from .ink import ink_features
    from .ocr.handseg import binarize, segment
    if tess_conf >= 0.8:
        return False
    bw = binarize(img)
    lay = segment(bw)
    kinds, hcv = [], []
    for lb in lay.line_boxes:
        if lb.w < 3 * lay.line_h:
            continue
        region = bw[lb.y0:lb.y1, lb.x0:lb.x1]
        kinds.append(classify_ink(region)[0])
        hcv.append(ink_features(region)["height_cv"])
    if not kinds:
        return False
    frac = kinds.count("handwritten") / len(kinds)
    h = float(np.median(hcv))
    return frac >= 0.6 or (tess_conf < 0.6 and h >= 0.38) or (tess_conf < 0.45 and frac >= 0.3)


HAND_LABEL = re.compile(r"^(\(?\s*(?:\d{1,3}|[A-Da-d]))\s+([.)\]])(?=\s|$)")


def _analyze_handwritten(original: np.ndarray, page: int, box: Box, img: np.ndarray, blur: float,
                         flag: str, find_graphics: bool) -> PartAnalysis | None:
    """Handwritten crop: offline TrOCR (text) + TrOCR math (formulas), see ocr/trocr.py.
    Drawings are found on the ink that is NOT part of a text line, so cursive words are
    never mistaken for a diagram and a hand-drawn figure is cut out as its own locked image."""
    from .ocr import get_engine
    cfg = settings()
    engine = get_engine(cfg["ocr"]["handwritten_engine"])
    if engine.name != "trocr":
        return None
    unreadable = cfg["ocr"].get("unreadable_conf_handwritten", 0.1)
    graphics: list[FoundGraphic] = []
    local_boxes: list[Box] = []
    text_boxes: list[Box] = []
    if find_graphics:                                        # (no drawings -> nothing to blank)
        graphics, local_boxes, text_boxes = _graphics_hand(original, box, img)

    ocr_in = img.copy()                                      # OCR input only - the crop is untouched
    for g in graphics:
        if g.kind == RegionKind.diagram:
            x0, y0 = max(0, g.box.x0 - box.x0), max(0, g.box.y0 - box.y0)
            x1, y1 = min(img.shape[1], g.box.x1 - box.x0), min(img.shape[0], g.box.y1 - box.y0)
            ocr_in[y0:y1, x0:x1] = 255
    for b in text_boxes:                                     # a sentence a drawing's box overlaps stays
        b = b.pad(2).clip(img.shape[1], img.shape[0])
        ocr_in[b.y0:b.y1, b.x0:b.x1] = img[b.y0:b.y1, b.x0:b.x1]
    res = engine.read(ocr_in)
    for w in res.words:                                      # "7 )" -> "7)": spacing only
        w.text = HAND_LABEL.sub(r"\1\2", w.text)

    text = res.text(unreadable)
    _mark_option_images(res, box, graphics, text, handwritten=True)
    q_no = None
    for ln in text.strip().splitlines()[:2]:
        q_no = match_marker(ln, None, False)
        if q_no is not None:
            break
    warnings = _edge_junk(res, img.shape[0], low=0.15)
    if q_no is not None:
        later = text.strip().splitlines()[2:]
        if any(match_marker(ln, q_no + 1, False) == q_no + 1 for ln in later):
            warnings.append(f"box also contains question {q_no + 1} - make it smaller")
    layout = compose_layout(res, local_boxes, unreadable)
    return PartAnalysis(page=page, box=box, image=img, text=text, confidence=res.confidence,
                        source_type="handwritten", blur=blur, quality_flag=flag, q_no=q_no,
                        graphics=graphics, edge_warnings=warnings, layout=layout)


def _trim_figure_boxes(boxes: list[Box], text_boxes: list[Box], binary: np.ndarray | None = None) -> list[Box]:
    """A figure's box can grow over a neighbour: the axis labels of the graph above, or the
    descenders of the sentence above it. Those slivers then show up in the figure's image.
      * two figures overlapping by a strip (less than a third of the lower one's height):
        the lower one starts below the upper one
      * a word of a question sentence reaching into the top / bottom quarter of a figure:
        the figure starts below / ends above it
    Real overlaps (a label inside a drawing) are left alone."""
    out = list(boxes)

    def hover(a: Box, b: Box) -> bool:
        return min(a.x1, b.x1) > max(a.x0, b.x0)

    def inked(b: Box, y: int) -> bool:
        return binary is not None and 0 <= y < binary.shape[0] and bool(binary[y, b.x0:b.x1].any())

    def past_ink_down(b: Box, y: int, limit: int) -> int:
        """The neighbour's ink can reach a few rows past its box (tails, arrow tips): move on
        to the first white row - the gap between the two."""
        while y < limit and inked(b, y):
            y += 1
        return y

    def past_ink_up(b: Box, y: int, limit: int) -> int:
        while y > limit and inked(b, y - 1):
            y -= 1
        return y

    for i in range(len(out)):
        for j in range(len(out)):
            a, b = out[i], out[j]
            if i == j or not hover(a, b) or not (a.y0 < b.y0 < a.y1 < b.y1):
                continue                                   # need: a above b, overlapping a strip
            if a.y1 - b.y0 < b.h / 3:
                y = past_ink_down(b, a.y1, a.y1 + (a.y1 - b.y0))
                out[j] = Box(x0=b.x0, y0=y, x1=b.x1, y1=b.y1)
    for i, b in enumerate(out):
        y0, y1 = b.y0, b.y1
        for t in text_boxes:
            if not hover(t, b):
                continue
            if t.y0 < b.y0 < t.y1 <= b.y0 + b.h / 4:       # sentence above, poking in at the top
                y0 = max(y0, past_ink_down(b, t.y1, t.y1 + t.h // 2))
            if t.y1 > b.y1 > t.y0 >= b.y1 - b.h / 4:       # sentence below, poking in at the bottom
                y1 = min(y1, past_ink_up(b, t.y0, t.y0 - t.h // 2))
        if y1 - y0 >= 8:
            out[i] = Box(x0=b.x0, y0=y0, x1=b.x1, y1=y1)
    return out


def _pad_into_white(binary: np.ndarray, b: Box, m: int) -> Box:
    """Grow b by up to m px on each side, but only as far as the strip stays free of ink."""
    H, W = binary.shape[:2]

    def clear(x0, y0, x1, y1):
        x0, y0, x1, y1 = max(0, x0), max(0, y0), min(W, x1), min(H, y1)
        return x1 <= x0 or y1 <= y0 or not binary[y0:y1, x0:x1].any()

    def grow(test):
        return next((k for k in range(m, 0, -1) if test(k)), 0)

    t = grow(lambda k: clear(b.x0, b.y0 - k, b.x1, b.y0))
    bt = grow(lambda k: clear(b.x0, b.y1, b.x1, b.y1 + k))
    lf = grow(lambda k: clear(b.x0 - k, b.y0 - t, b.x0, b.y1 + bt))
    rt = grow(lambda k: clear(b.x1, b.y0 - t, b.x1 + k, b.y1 + bt))
    return Box(x0=max(0, b.x0 - lf), y0=max(0, b.y0 - t), x1=min(W, b.x1 + rt), y1=min(H, b.y1 + bt))


def _labels_inside_options(res: OCRResult, img: np.ndarray, box: Box, graphics: list[FoundGraphic]) -> bool:
    """An option label printed right against its graph ("(a)" + plot) ends up inside the
    figure box, so the blanked OCR pass never sees it. Read each image option sparsely and
    put a confident label found at its left edge back into the text, on the line it sits on."""
    from .ocr.tesseract import TesseractEngine
    from .schemas import Word
    have = {w.text for w in res.words if OPTION_LABEL.match(w.text)}
    added = False
    for g in graphics:
        if g.kind != RegionKind.option_image:
            continue
        x0, y0 = max(0, g.box.x0 - box.x0), max(0, g.box.y0 - box.y0)
        x1, y1 = min(img.shape[1], g.box.x1 - box.x0), min(img.shape[0], g.box.y1 - box.y0)
        if x1 - x0 < 8 or y1 - y0 < 8:
            continue
        try:
            found = TesseractEngine(psm=11).read(img[y0:y1, x0:x1]).words
        except Exception:
            continue
        for w in found:
            if w.conf < 0.8 or not OPTION_LABEL.match(w.text) or w.text in have or w.box.x1 > 0.35 * (x1 - x0):
                continue
            wb = Box(x0=w.box.x0 + x0, y0=w.box.y0 + y0, x1=w.box.x1 + x0, y1=w.box.y1 + y0)
            cy = (wb.y0 + wb.y1) / 2
            line = next((v.line_id for v in res.words if v.box.y0 <= cy <= v.box.y1), None)
            res.words.append(Word(text=w.text, conf=w.conf, box=wb,
                                  line_id=line if line is not None else -1 - len(res.words)))
            have.add(w.text)
            added = True
            # the label is now text in the question: start the figure's crop after it, so the
            # final output doesn't show "(a) (a)". (Only the box moves; the crop is still cut
            # untouched from the original page.)
            g.box = Box(x0=min(g.box.x1 - 8, box.x0 + wb.x1 + 6), y0=g.box.y0, x1=g.box.x1, y1=g.box.y1)
    return added


def _first_word(res: OCRResult):
    """Leftmost word of the first two lines (OCR's own line grouping follows a tilted page)."""
    groups: dict[int, list] = {}
    for x in res.words:
        groups.setdefault(x.line_id, []).append(x)
    ordered = sorted(groups.values(), key=lambda ws: sum(x.box.y0 for x in ws) / len(ws))
    first = [x for ws in ordered[:2] for x in ws]
    return (min(first, key=lambda w: w.box.x0), groups) if first else (None, groups)


def _marker_fix(img: np.ndarray, res: OCRResult, q_no: int | None) -> tuple | None:
    """The layout copy should show the number as printed. OCR often misreads a coloured '8.'
    as 'g.' or misses it. Only when a digits-only re-read of that very spot returns q_no:
      ("replace", word, "8.")  - the first word is the misread number
      ("prefix", word, "8.")   - the number was not read at all (ink left of the first word)
    The verbatim OCR text is not changed."""
    if q_no is None or not res.words:
        return None
    w, groups = _first_word(res)
    from .ocr.tesseract import TesseractEngine

    def digits(piece) -> int | None:
        try:
            m = re.match(r"^(\d{1,3})[.)]?$", TesseractEngine().read_digits(piece))
        except Exception:
            return None
        return int(m.group(1)) if m else None

    b = w.box
    if re.match(r"^\S{1,3}[.)]$", w.text):
        if w.text[:-1] == str(q_no):
            return None
        piece = img[max(0, b.y0 - 4):b.y1 + 4, max(0, b.x0 - 4):b.x1 + 4]
        return ("replace", w, f"{q_no}{w.text[-1]}") if digits(piece) == q_no else None
    line = groups[w.line_id]
    y0, y1 = max(0, min(x.box.y0 for x in line) - 4), max(x.box.y1 for x in line) + 4
    left = img[y0:y1, 0:max(0, b.x0 - 3)]
    if left.size:
        gl = left if left.ndim == 2 else cv2.cvtColor(left, cv2.COLOR_BGR2GRAY)
        if (gl < 170).sum() > 15 and digits(left) == q_no:
            return ("prefix", w, f"{q_no}.")
    return None


FIG = "{{fig:%d}}"
CROP_REF = re.compile(r"\{\{crop:(\w+)\}\}")
FIG_NAME = {RegionKind.diagram: "diagram", RegionKind.table: "table", RegionKind.option_image: "option image"}


def _build_display(q: QuestionRecord, children: list, order) -> None:
    """q.display_text = the parts' layouts with each {{crop:id}} as "[diagram n]" etc., and
    q.figures[n-1] = that figure's image. Figures added by hand (D / T / I) go at the end on
    their own lines; removed or superseded ones disappear from the text."""
    by_id = {c.id: c for c in children}
    texts = [p.layout_text or p.text for p in q.parts]
    ids: list[str] = []
    for t in texts:
        ids += [m for m in CROP_REF.findall(t) if m in by_id and m not in ids]
    extra = [c.id for c in sorted(children, key=order) if c.id not in ids]
    ids += extra
    label: dict[str, str] = {}
    count: Counter = Counter()
    for cid in ids:                                   # numbered per kind: diagram 1, option image 1
        kind = FIG_NAME.get(by_id[cid].kind, "figure")
        count[kind] += 1
        label[cid] = f"{kind} {count[kind]}"
    name = lambda cid: f"[{label[cid]}]"  # noqa: E731

    def show(sources: list[str]) -> str:
        out = []
        for t in sources:
            for ln in t.splitlines():
                had = CROP_REF.search(ln) is not None
                ln = CROP_REF.sub(lambda m: name(m.group(1)) if m.group(1) in by_id else "", ln).rstrip()
                if ln.strip() or not had:              # drop a row that only held a removed figure
                    out.append(ln)
        out += [name(cid) for cid in extra]
        return "\n".join(out).strip("\n")
    q.display_text = show(texts)
    # the same layout with formulas as LaTeX: what the editable copy starts from
    q.display_latex = show([p.layout_latex or p.latex_text or p.layout_text or p.text for p in q.parts])
    # shown and exported: a structure redrawn in Ketcher, else the clean drawing (no paper /
    # shadows); the locked photo crop stays available as "original"
    q.figures = [{"label": label[cid],
                  "path": by_id[cid].structure_path or by_id[cid].clean_path or by_id[cid].path,
                  "original": by_id[cid].path, "crop_id": cid, "smiles": by_id[cid].smiles}
                 for cid in ids]


def compose_layout(res: OCRResult, figures: list[Box], unreadable: float, fix=None) -> str:
    """The words as they sit on the page: same lines, the same indents and wide gaps (in
    character widths), and a {{fig:n}} marker where figure n sits - in the line it stands
    in (an image option next to its "(a)"), or on its own row between the lines (a diagram
    under a sentence; figures side by side share a row). Nothing is reworded or corrected:
    unreadable words stay [?]; the only change is a number fix confirmed by _marker_fix."""
    lines: dict[int, list] = {}
    for w in res.words:
        lines.setdefault(w.line_id, []).append(w)
    rows: list[dict] = []
    for ws in lines.values():
        items = []
        for w in ws:
            t = "[?]" if w.conf < unreadable else w.text
            if fix and w is fix[1]:
                if fix[0] == "replace":
                    t = fix[2]
                else:
                    items.append((0, 0, fix[2]))
            items.append((w.box.x0, w.box.x1, t))
        rows.append({"key": min(w.box.y0 for w in ws), "y0": min(w.box.y0 for w in ws),
                     "y1": max(w.box.y1 for w in ws), "items": items})
    text_rows = list(rows)
    fig_rows: list[dict] = []
    for n, b in enumerate(figures, start=1):
        tok = (b.x0, b.x1, FIG % n)
        host = [r for r in text_rows if b.y0 <= (r["y0"] + r["y1"]) / 2 <= b.y1]
        if len(host) == 1:                          # stands in one text line, e.g. "(a) [graph]"
            host[0]["items"].append(tok)
            continue
        # beside several lines -> after them; otherwise at its own height
        key = max(r["y0"] for r in host) + 0.5 if host else b.y0
        row = next((r for r in fig_rows if r["y0"] < b.y1 and b.y0 < r["y1"]), None)
        if row is None:
            row = {"key": key, "y0": b.y0, "y1": b.y1, "items": []}
            fig_rows.append(row)
        row["items"].append(tok)
        row["y0"], row["y1"] = min(row["y0"], b.y0), max(row["y1"], b.y1)
    rows += fig_rows
    if not rows:
        return ""
    widths = [(w.box.x1 - w.box.x0) / len(w.text) for w in res.words if w.conf > 0.5 and len(w.text) >= 3] \
        or [(w.box.x1 - w.box.x0) / len(w.text) for w in res.words if len(w.text) >= 3]
    cw = max(4.0, float(np.median(widths))) if widths else 12.0
    left = min(it[0] for r in rows for it in r["items"])
    out = []
    for r in sorted(rows, key=lambda r: r["key"]):
        items = sorted(r["items"], key=lambda it: (it[0] + it[1]) / 2)
        s, end = " " * int(round((items[0][0] - left) / cw)), None
        for x0, x1, t in items:
            if end is not None:
                gap = x0 - end
                s += " " if gap < 2 * cw else " " * int(round(gap / cw))
            s += t
            end = max(x1, end or 0)
        out.append(s.rstrip())
    return "\n".join(out)


def _reread_marker(img: np.ndarray, res: OCRResult) -> int | None:
    """The first word looks like a number ('g.', 'B.', 'S)') but isn't digits: read that word
    again with a digits-only whitelist. Used only to find q_no - the text stays verbatim."""
    if not res.words:
        return None
    # Use OCR's own line grouping (follows a tilted page), first two lines.
    groups: dict[int, list] = {}
    for x in res.words:
        groups.setdefault(x.line_id, []).append(x)
    ordered = sorted(groups.values(), key=lambda ws: sum(x.box.y0 for x in ws) / len(ws))
    first_line = [x for ws in ordered[:2] for x in ws]
    w = min(first_line, key=lambda w: w.box.x0)
    first_line = groups[w.line_id]
    b = w.box
    y0, y1 = max(0, min(x.box.y0 for x in first_line) - 4), max(x.box.y1 for x in first_line) + 4
    pieces = []
    if re.match(r"^\S{1,3}[.)]$", w.text) and not w.text[:-1].isdigit():
        pieces.append(img[max(0, b.y0 - 4):b.y1 + 4, max(0, b.x0 - 4):b.x1 + 4])   # misread: 'g.'
    left = img[y0:y1, 0:max(0, b.x0 - 3)]                                         # missed: coloured '8.'
    if left.size:
        gl = left if left.ndim == 2 else cv2.cvtColor(left, cv2.COLOR_BGR2GRAY)
        if (gl < 170).sum() > 15:
            pieces.append(left)
    from .ocr.tesseract import TesseractEngine
    for piece in pieces:
        try:
            digits = re.match(r"^(\d{1,3})[.)]?$", TesseractEngine().read_digits(piece))
        except Exception:
            continue
        if digits and 0 < int(digits.group(1)) <= 300:
            return int(digits.group(1))
    return None


def _edge_junk(res: OCRResult, height: int, low: float = 0.5) -> list[str]:
    """A low-confidence line touching the top/bottom edge usually means the box cuts through
    a neighbouring line. We never delete it (fidelity) - we flag it."""
    lines: dict[int, list] = {}
    for w in res.words:
        lines.setdefault(w.line_id, []).append(w)
    if not lines:
        return []
    ordered = sorted(lines.values(), key=lambda ws: min(w.box.y0 for w in ws))
    out = []
    first, last = ordered[0], ordered[-1]
    if min(w.box.y0 for w in first) <= 3 and sum(w.conf for w in first) / len(first) < low:
        out.append("box cuts through other content at the top - redraw it (M) a little larger or smaller")
    if max(w.box.y1 for w in last) >= height - 3 and sum(w.conf for w in last) / len(last) < low:
        out.append("box cuts through other content at the bottom - redraw it (M) a little larger or smaller")
    return out


def _mark_option_images(res: OCRResult, box: Box, graphics: list[FoundGraphic], text: str,
                        handwritten: bool = False) -> None:
    """Drawings at/after the option labels, when the options themselves have (almost) no text,
    are image options - e.g. 'Which graph is correct? (1) [graph] (2) [graph] ...'.
    handwritten: OCR of handwriting misreads text as a label now and then ("(* indicates" ->
    "c) in"), so there each image option must sit right of its own label, in the same row."""
    diagrams = [g for g in graphics if g.kind == RegionKind.diagram]
    if len(diagrams) < 2:
        return
    # a handwriting "word" is a whole piece ("a) 1:1:1"), so a label may start a longer text
    labels = [w for w in res.words if OPTION_LABEL.match(w.text) or (handwritten and OPTION_START.match(w.text))]
    if len(labels) < 2:
        return
    label_ys = sorted(w.box.y0 + box.y0 for w in labels)
    options = parse_question(text)["options"]
    textless = not options or sum(len(re.sub(r"^\(?\w[.)]\)?\s*", "", o)) < 4 for o in options) >= 2
    if handwritten:
        def labelled(g):
            return any(min(g.box.y1, w.box.y1 + box.y0) - max(g.box.y0, w.box.y0 + box.y0) > 0
                       and w.box.x0 + box.x0 < g.box.x0 for w in labels)
        cand = [g for g in diagrams if labelled(g)]
    else:
        cand = [g for g in diagrams if g.box.y1 >= label_ys[0]]
    if textless and len(cand) >= 2:
        for g in cand:
            g.kind = RegionKind.option_image


def refresh_figures(pr: Project, q: QuestionRecord) -> None:
    """Rebuild only q.figures / q.display_text (e.g. after clean copies were added)."""
    children = [c for c in pr.crops if c.parent_id == q.id and not c.superseded
                and c.kind in (RegionKind.diagram, RegionKind.table, RegionKind.option_image)]
    _build_display(q, children, lambda c: (c.page, c.box.y0 // 40, c.box.x0))


def assemble_question(pr: Project, q: QuestionRecord) -> QuestionRecord:
    """(Re)build text, options, statements, images, type and review flags from the question's
    parts and its child crops. Reviewer decisions (edited text, confirmed labels) are kept."""
    cfg = settings()
    parts = q.parts
    q.text = "\n".join(p.text for p in parts if p.text)
    q.latex_text = "\n".join((p.latex_text or p.text) for p in parts if p.text)
    n = [max(1, len(p.text)) for p in parts]
    q.ocr_confidence = round(sum(p.ocr_confidence * k for p, k in zip(parts, n)) / max(1, sum(n)), 4)
    srcs = {p.source_type for p in parts}
    q.source_type = srcs.pop() if len(srcs) == 1 else "mixed"
    q.blur_score = min((p.blur_score for p in parts), default=0.0)
    q.quality_flag = next((p.quality_flag for p in parts if p.quality_flag != "ok"), "ok")
    if parts:
        q.page, q.box, q.crop_id, q.crop_path = parts[0].page, parts[0].box, parts[0].crop_id, parts[0].crop_path

    children = [c for c in pr.crops if c.parent_id == q.id and not c.superseded
                and c.kind in (RegionKind.diagram, RegionKind.table, RegionKind.option_image)]
    order = lambda c: (c.page, c.box.y0 // 40, c.box.x0)  # noqa: E731
    diagrams = sorted((c for c in children if c.kind == RegionKind.diagram), key=order)
    tables = sorted((c for c in children if c.kind == RegionKind.table), key=order)
    opt_imgs = sorted((c for c in children if c.kind == RegionKind.option_image), key=order)
    q.diagram_paths = [c.path for c in diagrams] + [c.path for c in tables]
    q.diagram_path = diagrams[0].path if diagrams else (tables[0].path if tables else None)
    q.option_images = [c.path for c in opt_imgs]
    _build_display(q, children, order)

    parsed = parse_question(q.text)
    q.options, q.statements = parsed["options"], parsed["statements"]
    if not q.options and opt_imgs:
        q.options = [f"[image option {i + 1}]" for i in range(len(opt_imgs))]
    elif opt_imgs and len(q.options) == len(opt_imgs) and all(OPTION_LABEL.match(o.strip()) for o in q.options):
        q.options = [f"{o.strip()} [option image {i + 1}]" for i, o in enumerate(q.options)]   # bare "(a)"

    tp, tconf, signals = analyze_type(q.text, parsed["options"] or q.options, q.statements,
                                      has_diagram=bool(diagrams), has_table=bool(tables),
                                      has_passage=q.passage_id is not None, exam=q.exam)
    if not q.labels_reviewed:
        q.question_type, q.type_confidence, q.type_signals = tp, tconf, signals
        q.multi_answer = tp == QuestionType.multi_correct

    reasons = []
    if q.ocr_confidence < cfg["ocr"]["confidence_review"]:
        reasons.append(f"OCR confidence {q.ocr_confidence:.2f}")
    if q.quality_flag != "ok":
        reasons.append(f"image quality: {q.quality_flag}")
    page = next((p for p in pr.pages if p.page == q.page), None)
    if page and page.quality_flag == "rescan":
        reasons.append("page too blurry - rescan recommended")
    if any(c.quality_flag != "ok" for c in children):
        reasons.append("blurry diagram")
    if any(c.needs_review for c in children):
        reasons.append("diagram touches the crop edge - may be cut off, widen the box")
    if not q.labels_reviewed and q.type_confidence < 0.7:
        reasons.append(f"question type uncertain ({q.question_type.value})")
    if "[?]" in q.text:
        reasons.append("unreadable words marked [?]")
    still_reading = any(p.status == "reading" for p in parts)
    if (q.q_no is None or q.q_no_inferred) and not still_reading:   # never guess before it is read
        prev = [x for x in pr.questions if x.id != q.id and x.q_no is not None and not x.q_no_inferred
                and (x.page, x.box.y0) < (q.page, q.box.y0)]
        if q.q_no is None and prev:
            q.q_no, q.q_no_inferred = prev[-1].q_no + 1, True
        if q.q_no_inferred:
            reasons.append(f"question number {q.q_no} inferred (no number found in the crop) - please confirm")
        else:
            reasons.append("no question number found")
    for p in parts:
        reasons.extend(p.edge_warnings)
    if not q.text.strip() and not children:
        reasons.append("no content found in the crop")
    if any(p.status == "reading" for p in parts):
        q.status = "reading"
        reasons = ["still being read (in the background) - the text appears when it is done"]
    elif any(p.status == "error" for p in parts):
        q.status = "error"
        reasons.insert(0, f"reading failed: {q.read_error or 'unknown error'} - redraw the box (M)")
    else:
        q.status = "ready"
    q.review_reasons = reasons
    q.needs_review = bool(reasons)
    return q

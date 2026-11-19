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


def analyze_part(original: np.ndarray, page: int, box: Box, find_graphics: bool = True) -> PartAnalysis:
    cfg = settings()
    unreadable = cfg["ocr"]["unreadable_conf"]
    box = box.clip(original.shape[1], original.shape[0])
    img = cut(original, box)
    blur, flag = crop_quality(img)

    res = read_region(img, "printed")                        # pass 1, on original pixels
    lines: dict[int, list] = {}
    for w in res.words:
        lines.setdefault(w.line_id, []).append(w)
    heights = [w.box.h for w in res.words if w.conf > 0.5 and w.box.h > 3]
    med_h = float(np.median(heights)) if heights else 24.0

    wc = make_working_copy(img, deskew=False)                # detection copy only
    graphics: list[FoundGraphic] = []
    if find_graphics:
        text_boxes = [w.box for ws in lines.values() if sum(x.conf > 0.5 for x in ws) >= 3
                      for w in ws if w.conf > 0.5]
        page_area = original.shape[0] * original.shape[1]
        min_area = max(cfg["layout"]["diagram_min_area_frac"] * page_area, (5 * med_h) ** 2)
        margin = cfg["layout"]["diagram_margin_px"]
        for g in detect_graphics(wc.binary, res.words, med_h, text_boxes, min_area_px=min_area):
            local = g.box
            # cut off = the drawing's INK reaches the crop border (not just a label's box)
            bw = wc.binary[local.y0:local.y1, local.x0:local.x1] > 0
            H_, W_ = img.shape[:2]
            edge = bool(bw.size) and (
                (local.y0 <= 1 and bw[:2].any()) or (local.y1 >= H_ - 1 and bw[-2:].any())
                or (local.x0 <= 1 and bw[:, :2].any()) or (local.x1 >= W_ - 1 and bw[:, -2:].any()))
            pb = Box(x0=local.x0 + box.x0, y0=local.y0 + box.y0, x1=local.x1 + box.x0, y1=local.y1 + box.y0)
            labels = " ".join(w.text for w in sorted(g.labels, key=lambda w: (w.box.y0 // 10, w.box.x0)))
            graphics.append(FoundGraphic(kind=g.kind, box=pb.pad(margin).clip(original.shape[1], original.shape[0]),
                                         labels=labels, edge_cut=edge))

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
        # text line (3+ confident words) that a drawing's box happened to overlap
        for ws in lines.values():
            if sum(x.conf > 0.5 for x in ws) >= 3:
                for w in ws:
                    if w.conf > 0.5:
                        b = w.box.pad(2).clip(img.shape[1], img.shape[0])
                        ocr_in[b.y0:b.y1, b.x0:b.x1] = img[b.y0:b.y1, b.x0:b.x1]
        res = read_region(ocr_in, "handwritten" if source == "handwritten" else "printed")

    text = res.text(unreadable)
    _mark_option_images(res, box, graphics, text)
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
    return PartAnalysis(page=page, box=box, image=img, text=text, confidence=res.confidence,
                        source_type=source, blur=blur, quality_flag=flag, q_no=q_no,
                        graphics=graphics, edge_warnings=warnings)


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


def _edge_junk(res: OCRResult, height: int) -> list[str]:
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
    if min(w.box.y0 for w in first) <= 3 and sum(w.conf for w in first) / len(first) < 0.5:
        out.append("box cuts through other content at the top - redraw it or turn on Snap")
    if max(w.box.y1 for w in last) >= height - 3 and sum(w.conf for w in last) / len(last) < 0.5:
        out.append("box cuts through other content at the bottom - redraw it or turn on Snap")
    return out


def _mark_option_images(res: OCRResult, box: Box, graphics: list[FoundGraphic], text: str) -> None:
    """Drawings at/after the option labels, when the options themselves have (almost) no text,
    are image options - e.g. 'Which graph is correct? (1) [graph] (2) [graph] ...'."""
    diagrams = [g for g in graphics if g.kind == RegionKind.diagram]
    if len(diagrams) < 2:
        return
    label_ys = sorted(w.box.y0 + box.y0 for w in res.words if OPTION_LABEL.match(w.text))
    if len(label_ys) < 2:
        return
    options = parse_question(text)["options"]
    textless = not options or sum(len(re.sub(r"^\(?\w[.)]\)?\s*", "", o)) < 4 for o in options) >= 2
    cand = [g for g in diagrams if g.box.y1 >= label_ys[0]]
    if textless and len(cand) >= 2:
        for g in cand:
            g.kind = RegionKind.option_image


def assemble_question(pr: Project, q: QuestionRecord) -> QuestionRecord:
    """(Re)build text, options, statements, images, type and review flags from the question's
    parts and its child crops. Reviewer decisions (edited text, confirmed labels) are kept."""
    cfg = settings()
    parts = q.parts
    q.text = "\n".join(p.text for p in parts if p.text)
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

    parsed = parse_question(q.text)
    q.options, q.statements = parsed["options"], parsed["statements"]
    if not q.options and opt_imgs:
        q.options = [f"[image option {i + 1}]" for i in range(len(opt_imgs))]

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
    if q.q_no is None or q.q_no_inferred:
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
    q.review_reasons = reasons
    q.needs_review = bool(reasons)
    return q

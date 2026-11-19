"""Turn page layouts into blocks: header, section, passage, question (with continuations).

A question runs from its marker to the next marker in reading order. Markers are matched
fuzzily (Q1, Q.1, Que 1, 1., 1), OCR slips like O.1 / 0.1), must sit at the left margin,
and must follow the numbering sequence - which stops option lists "1. 2. 3. 4." or
statement lists from being mistaken for new questions.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from rapidfuzz import fuzz

from .layout.types import Graphic, Line, PageLayout, Segment
from .schemas import Box, RegionKind

MARKER_STRONG = re.compile(
    r"^\s*(?:(?:Q|Que|Ques|Question|O|0)\s*(?:No)?\s*[.\-:]?\s*)(\d{1,3})\s*[.):\]]?(?=\s|$)", re.I)
MARKER_PUNCT = re.compile(r"^\s*(\d{1,3})\s*[.):\]](?=\s|$|[A-Za-z(])")
MARKER_BARE = re.compile(r"^\s*(\d{1,3})\s+(?=[A-Z])")
PASSAGE_RE = re.compile(r"^\s*(paragraph|passage|comprehension|read the following)", re.I)
PASSAGE_RANGE = re.compile(r"(\d{1,3})\s*(?:-|–|to|and|&)\s*(\d{1,3})")
SECTION_RE = re.compile(r"^\s*(section|part)\s*[-:]?\s*([A-Z0-9IVX]+)\b", re.I)
INSTRUCTION_RE = re.compile(r"instruction|read carefully|marking scheme|duration|max(?:imum)? marks",
                            re.I)


@dataclass
class Part:
    page: int
    box: Box                     # working coords on that page
    graphics: list[Graphic] = field(default_factory=list)
    lines: list[Line] = field(default_factory=list)


@dataclass
class Block:
    kind: RegionKind
    parts: list[Part]
    q_no: int | None = None
    subject: str | None = None
    passage_id: str | None = None
    passage_range: tuple[int, int] | None = None
    marker_text: str = ""
    running: bool = False        # running page header/footer

    @property
    def page(self) -> int:
        return self.parts[0].page


# OCR often reads a question number's "1" as i / l / I / | / ! and "0" as o / O.
# B->8, S->5, Z->2 are NOT mapped: option lines like "B. ..." would turn into questions.
_CONFUSE = str.maketrans({"i": "1", "l": "1", "I": "1", "|": "1", "!": "1", "o": "0", "O": "0"})
_LEAD = re.compile(r"^(\s*)([0-9ilI|!oO]{1,3})(?=\s*[.):\]])")


def fix_marker_digits(text: str) -> str:
    """'i. System in...' -> '1. System in...' (at most one confusable letter, so 'ii.' stays)."""
    m = _LEAD.match(text)
    if not m or m.group(2).isdigit():
        return text
    tok = m.group(2)
    if sum(not c.isdigit() for c in tok) > 1:
        return text
    return text[:m.start(2)] + tok.translate(_CONFUSE) + text[m.end(2):]


def match_marker(text: str, expected: int | None, after_section: bool) -> int | None:
    text = fix_marker_digits(text)
    for rx, strength in ((MARKER_STRONG, 2), (MARKER_PUNCT, 1), (MARKER_BARE, 0)):
        m = rx.match(text)
        if not m:
            continue
        n = int(m.group(1))
        if expected is None:
            return n if strength >= 1 and n <= 200 else None
        if n == expected:
            return n
        if after_section and n == 1:
            return n
        if strength >= 1 and expected < n <= expected + 4:
            return n
        return None
    return None


def match_section(text: str, subjects: list[str]) -> str | None:
    words = text.split()
    if not words or len(words) > 6:
        return None
    for s in subjects:
        if fuzz.partial_ratio(s.lower(), text.lower()) >= 88 and len(text) <= len(s) + 25:
            return s
    m = SECTION_RE.match(text)
    if m:
        return None if len(words) > 6 else "__section__"
    return None


def _looks_like_option_run(lines: list[Line], idx: int, n: int) -> bool:
    """'2.' followed within 3 lines by '3.' and '4.' -> it's an option list, not questions."""
    hits, want = 0, n + 1
    for ln in lines[idx + 1: idx + 4]:
        m = MARKER_PUNCT.match(ln.text) or MARKER_BARE.match(ln.text)
        if m and int(m.group(1)) == want:
            hits += 1
            want += 1
    # also catches "1. foo 2. bar" packed on one line
    packed = len(re.findall(r"(?:^|\s)\d\s*[.)]\s", lines[idx].text)) >= 2
    return hits >= 2 or packed


def segment_document(layouts: list[PageLayout], subjects: list[str]) -> list[Block]:
    blocks: list[Block] = []
    current: Block | None = None
    last_q: int | None = None
    subject: str | None = None
    after_section = False
    passage_count = 0

    def start(kind: RegionKind, seg: Segment, line: Line | None, **kw) -> Block:
        nonlocal current
        b = Block(kind=kind, parts=[Part(page=seg.page, box=(line.box if line else seg.box))], **kw)
        blocks.append(b)
        current = b
        return b

    def attach_line(seg: Segment, ln: Line):
        nonlocal current
        if current is None:
            start(RegionKind.header, seg, ln)
        part = current.parts[-1]
        if part.page != seg.page or not _same_segment(part, seg):
            current.parts.append(Part(page=seg.page, box=ln.box))   # continuation
            part = current.parts[-1]
        part.box = part.box.union(ln.box)
        part.lines.append(ln)

    for layout in layouts:
        top_band = layout.height * 0.04
        bottom_band = layout.height * 0.96
        for seg in layout.segments:
            starts: list[tuple[float, Block]] = []
            for i, ln in enumerate(seg.lines):
                text = ln.text.strip()
                at_margin = ln.box.x0 - seg.margin_x <= 2.5 * layout.median_char_h
                # Running headers/footers (page numbers, coaching names) -> their own header crop.
                if (ln.box.y1 < top_band or ln.box.y0 > bottom_band) and len(text) < 60 \
                        and not MARKER_STRONG.match(text):
                    blk = Block(kind=RegionKind.header, running=True,
                                parts=[Part(page=seg.page, box=ln.box, lines=[ln])])
                    blocks.append(blk)
                    continue

                sec = match_section(text, subjects)
                if sec:
                    if sec != "__section__":
                        subject = sec
                    b = start(RegionKind.section, seg, ln, subject=subject)
                    b.parts[0].lines.append(ln)
                    after_section = True
                    starts.append((ln.box.y0, b))
                    continue

                if at_margin and PASSAGE_RE.match(text):
                    passage_count += 1
                    rng = PASSAGE_RANGE.search(text)
                    b = start(RegionKind.passage, seg, ln, subject=subject,
                              passage_id=f"P{passage_count}",
                              passage_range=(int(rng.group(1)), int(rng.group(2))) if rng else None)
                    b.parts[0].lines.append(ln)
                    starts.append((ln.box.y0, b))
                    continue

                expected = None if last_q is None else last_q + 1
                n = match_marker(text, expected, after_section) if at_margin else None
                if n is not None and current is not None and current.kind == RegionKind.question \
                        and _looks_like_option_run(seg.lines, i, n):
                    n = None
                if n is not None:
                    b = start(RegionKind.question, seg, ln, q_no=n, subject=subject, marker_text=text[:12])
                    b.parts[0].lines.append(ln)
                    last_q, after_section = n, False
                    starts.append((ln.box.y0, b))
                    continue

                if current is not None and current.kind == RegionKind.header and \
                        INSTRUCTION_RE.search(text):
                    current.kind = RegionKind.instructions
                attach_line(seg, ln)

            _assign_graphics(seg, blocks, starts, current)

    return [b for b in blocks if b.parts]


def _same_segment(part: Part, seg: Segment) -> bool:
    return part.box.intersection(seg.box) > 0 or (
        seg.box.x0 <= (part.box.x0 + part.box.x1) / 2 < seg.box.x1
        and seg.box.y0 <= part.box.y0 < seg.box.y1)


def _assign_graphics(seg: Segment, blocks: list[Block], starts: list[tuple[float, Block]],
                     current: Block | None):
    """Each diagram/table goes to the block whose marker is the last one above its top edge."""
    for g in seg.graphics:
        owner = None
        for y, b in starts:
            if y <= g.box.y0 + 4:
                owner = b
        if owner is None:
            # Above the first marker in this column: belongs to the question carried over.
            owner = _carried_block(blocks, starts, seg) or current
        if owner is None:
            continue
        part = next((p for p in reversed(owner.parts) if p.page == seg.page and _same_segment(p, seg)),
                    None)
        if part is None:
            part = Part(page=seg.page, box=g.box)
            owner.parts.append(part)
        part.box = part.box.union(g.box)
        part.graphics.append(g)


def _carried_block(blocks: list[Block], starts, seg: Segment) -> Block | None:
    first_start = starts[0][1] if starts else None
    idx = blocks.index(first_start) if first_start in blocks else len(blocks)
    for b in reversed(blocks[:idx]):
        if b.running:
            continue
        if b.kind in (RegionKind.question, RegionKind.passage, RegionKind.instructions,
                      RegionKind.header) and b.parts[-1].box.h > 0:
            return b
    return None

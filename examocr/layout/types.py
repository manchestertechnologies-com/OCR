"""Layout structures. All boxes here are in WORKING-COPY coordinates."""
from __future__ import annotations

from dataclasses import dataclass, field

from ..schemas import Box, RegionKind, Word


@dataclass
class Line:
    box: Box
    words: list[Word]
    source_type: str = "printed"

    @property
    def text(self) -> str:
        return " ".join(w.text for w in sorted(self.words, key=lambda w: w.box.x0))

    @property
    def conf(self) -> float:
        return sum(w.conf for w in self.words) / max(1, len(self.words))


@dataclass
class Graphic:
    kind: RegionKind           # diagram | table | equation
    box: Box
    labels: list[Word] = field(default_factory=list)
    score: float = 1.0


@dataclass
class Segment:
    """One stretch of reading order: a column inside a horizontal zone of the page."""
    page: int
    order: int
    box: Box
    lines: list[Line] = field(default_factory=list)
    graphics: list[Graphic] = field(default_factory=list)

    @property
    def margin_x(self) -> int:
        return min((l.box.x0 for l in self.lines), default=self.box.x0)


@dataclass
class PageLayout:
    page: int
    width: int
    height: int
    columns: int
    segments: list[Segment]
    median_char_h: float

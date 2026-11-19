from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import numpy as np

from ..schemas import Word


@dataclass
class OCRResult:
    words: list[Word] = field(default_factory=list)

    def text(self, unreadable_conf: float = 0.3) -> str:
        """Verbatim text with original line breaks. Low-confidence words become [?], never guessed."""
        lines: dict[int, list[Word]] = {}
        for w in self.words:
            lines.setdefault(w.line_id, []).append(w)
        out = []
        for lid in sorted(lines, key=lambda k: min(w.box.y0 for w in lines[k])):
            ws = sorted(lines[lid], key=lambda w: w.box.x0)
            out.append(" ".join("[?]" if w.conf < unreadable_conf else w.text for w in ws))
        return "\n".join(out)

    @property
    def confidence(self) -> float:
        if not self.words:
            return 0.0
        total = sum(max(1, len(w.text)) for w in self.words)
        return sum(w.conf * max(1, len(w.text)) for w in self.words) / total


class OCREngine(ABC):
    name = "base"

    @abstractmethod
    def read(self, image: np.ndarray) -> OCRResult:
        """Read text from an image (BGR or grey). Word boxes are relative to the image."""

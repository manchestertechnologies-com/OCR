"""Split verbatim question text into stem, options and statements.

Splitting never alters characters: every option/statement is an exact substring of `text`.
"""
from __future__ import annotations

import re

OPT_PATTERNS = [
    re.compile(r"(?:^|(?<=\s))\(([A-D])\)\s*"),          # (A)
    re.compile(r"(?:^|(?<=\s))\(([1-4])\)\s*"),          # (1)
    re.compile(r"(?:^|(?<=\s))\(([a-d])\)\s*"),          # (a)
    re.compile(r"(?:^|(?<=\s))([A-D])[.)]\s+"),          # A.  A)
    re.compile(r"(?:^|(?<=\s))([1-4])[.)]\s+"),          # 1.  1)
    re.compile(r"(?:^|(?<=\s))([a-d])\)\s*"),            # a)  (common in handwritten papers)
]
STATEMENT_START = re.compile(
    r"^\s*(statement\s*[-–]?\s*(?:I{1,3}|IV|\d)\b|assertion\b|reason\b|\(?(?:i{1,3}|iv|v)\)|"
    r"[A-E]\s*[.)]\s|\(?[P-S]\)|column\s*[-–]?\s*I|list\s*[-–]?\s*I)", re.I)


def split_options(text: str) -> tuple[str, list[str]]:
    """Find the label scheme that yields a complete ordered run (A,B,C,D / 1,2,3,4)."""
    best: tuple[str, list[str]] | None = None
    for pat in OPT_PATTERNS:
        ms = list(pat.finditer(text))
        if len(ms) < 2:
            continue
        labels = [m.group(1).upper() for m in ms]
        seq = "ABCD" if labels[0] in "ABCD" else "1234"
        # take the LAST complete run (options come at the end of a question)
        run_start = None
        for i in range(len(ms) - 1, -1, -1):
            if labels[i] == seq[0]:
                tail = labels[i:i + 4]
                if tail == list(seq[:len(tail)]) and len(tail) >= 3:
                    run_start = i
                    break
        if run_start is None:
            continue
        run = ms[run_start:run_start + 4]
        opts = []
        for j, m in enumerate(run):
            end = run[j + 1].start() if j + 1 < len(run) else len(text)
            opts.append(text[m.start():end].strip())
        stem = text[:run[0].start()].rstrip()
        if best is None or len(opts) > len(best[1]):
            best = (stem, opts)
    return best if best else (text, [])


def split_statements(stem: str) -> list[str]:
    out: list[str] = []
    for line in stem.splitlines():
        if STATEMENT_START.match(line):
            out.append(line.strip())
        elif out and line.strip() and not line.strip().endswith("?") and len(out) < 12 \
                and line[:1].isspace():
            out[-1] = out[-1] + "\n" + line.strip()
    return out


def parse_question(text: str) -> dict:
    stem, options = split_options(text)
    statements = split_statements(stem)
    return {"stem": stem, "options": options, "statements": statements}

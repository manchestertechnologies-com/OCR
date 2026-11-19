"""Final output: every question as it looks on the page - its text (the reviewer's edited copy,
else the page layout) with each "[diagram 1]" / "[option image 2]" marker replaced by that
locked crop image, at the spot it was placed.

  questions.html   open in a browser (images next to it in crops/)
  questions.docx   Word, images embedded
"""
from __future__ import annotations

import html
import io
import re
from pathlib import Path

from .schemas import Project, QuestionRecord

FIG_RE = re.compile(r"\[(diagram|table|option image|figure) (\d+)\]")
DPI = 300          # pages are rendered at >= 300 DPI: 300 px of crop = 1 inch on paper


def final_text(q: QuestionRecord) -> str:
    return q.text_edited if q.text_edited is not None else (q.display_text or q.text)


def _pieces(q: QuestionRecord):
    """Per line: [("text", str) | ("img", crop path)] in order."""
    figs = {f["label"]: f["path"] for f in q.figures}
    for line in final_text(q).splitlines():
        out, pos = [], 0
        for m in FIG_RE.finditer(line):
            path = figs.get(f"{m.group(1)} {m.group(2)}")
            if path is None:
                continue                                   # unknown marker: keep it as text
            out.append(("text", line[pos:m.start()]))
            out.append(("img", path))
            pos = m.end()
        out.append(("text", line[pos:]))
        yield [p for p in out if p[0] == "img" or p[1]]


def _title(q: QuestionRecord) -> str:
    return f"Q{q.q_no if q.q_no is not None else '?'} · page {q.page}" + (f" · {q.subject}" if q.subject else "")


def build_html(pr: Project) -> str:
    rows = []
    for q in pr.questions:
        body = "\n".join("".join(html.escape(v) if k == "text" else f'<img src="{html.escape(v)}" alt="">'
                                 for k, v in line) for line in _pieces(q))
        rows.append(f'<section><h2>{html.escape(_title(q))} <small>{q.question_type.value}</small></h2>'
                    f'<div class="q">{body}</div></section>')
    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{html.escape(pr.source_name)} - questions</title>
<style>
body {{ font-family: system-ui, sans-serif; margin: 24px; color: #111; background: #fff; }}
h1 {{ font-size: 20px; }} h2 {{ font-size: 15px; margin: 0 0 6px; }} small {{ color: #666; font-weight: normal; }}
section {{ border: 1px solid #ddd; border-radius: 8px; padding: 12px 14px; margin: 0 0 16px; break-inside: avoid; }}
.q {{ font: 14px/1.6 Consolas, ui-monospace, monospace; white-space: pre; overflow-x: auto; }}
.q img {{ vertical-align: middle; max-width: 48%; max-height: 260px; }}
</style></head><body>
<h1>{html.escape(pr.source_name)} · {html.escape(pr.exam)} · {len(pr.questions)} questions</h1>
{chr(10).join(rows)}
</body></html>"""


def build_docx(pr: Project, base: Path) -> bytes | None:
    """Word file with the images embedded at their place in the text. None if python-docx
    is not installed (the rest of the export still works)."""
    try:
        from docx import Document
        from docx.shared import Inches, Pt
    except ImportError:
        return None
    from PIL import Image
    doc = Document()
    doc.add_heading(f"{pr.source_name} · {pr.exam}", level=1)
    for q in pr.questions:
        doc.add_heading(f"{_title(q)}  ({q.question_type.value})", level=3)
        for line in _pieces(q):
            p = doc.add_paragraph()
            p.paragraph_format.space_after = Pt(2)
            for kind, v in line:
                if kind == "text":
                    r = p.add_run(v)
                    r.font.name, r.font.size = "Consolas", Pt(10.5)   # monospace keeps the alignment
                else:
                    f = base / v
                    if not f.exists():
                        continue
                    with Image.open(f) as im:
                        width = min(im.width / DPI, 3.0 if len(line) > 1 else 6.0)   # side by side: smaller
                    p.add_run().add_picture(str(f), width=Inches(width))
    buf = io.BytesIO()
    doc.save(buf)
    return buf.getvalue()

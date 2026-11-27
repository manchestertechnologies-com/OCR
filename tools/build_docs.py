"""Build the downloadable documents from docs/*.md:  Word (.docx) and PDF.

    python tools/build_docs.py docs/Mathpix-Fast-Reading.md

Word: python-docx (headings, paragraphs, bold, lists, tables, code).
PDF:  the same content as a styled HTML page, printed by Chrome / Edge (headless).
Handles the Markdown these documents use - not a general Markdown converter.
"""
from __future__ import annotations

import html
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


# ------------------------------------------------------------------ parse
def blocks(md: str) -> list[tuple]:
    """-> [("h", level, text) | ("p", text) | ("ul", [items]) | ("ol", [items]) | ("table", rows) | ("code", text)]."""
    out: list[tuple] = []
    lines = md.replace("\r\n", "\n").split("\n")
    i = 0
    while i < len(lines):
        ln = lines[i]
        if not ln.strip():
            i += 1
            continue
        if ln.startswith("```"):
            j = i + 1
            while j < len(lines) and not lines[j].startswith("```"):
                j += 1
            out.append(("code", "\n".join(lines[i + 1:j])))
            i = j + 1
            continue
        m = re.match(r"^(#{1,4})\s+(.*)", ln)
        if m:
            out.append(("h", len(m.group(1)), m.group(2).strip()))
            i += 1
            continue
        if ln.lstrip().startswith("|"):
            rows = []
            while i < len(lines) and lines[i].lstrip().startswith("|"):
                cells = [c.strip() for c in lines[i].strip().strip("|").split("|")]
                if not all(re.fullmatch(r":?-{3,}:?", c) for c in cells):
                    rows.append(cells)
                i += 1
            out.append(("table", rows))
            continue
        for kind, rx in (("ul", r"^\s*[-*]\s+(.*)"), ("ol", r"^\s*\d+\.\s+(.*)")):
            if re.match(rx, ln):
                items = []
                while i < len(lines) and (re.match(rx, lines[i]) or (items and lines[i].startswith("   ") and lines[i].strip())):
                    mm = re.match(rx, lines[i])
                    if mm:
                        items.append(mm.group(1).strip())
                    else:
                        items[-1] += "\n" + lines[i].strip()
                    i += 1
                out.append((kind, items))
                break
        else:
            para = [ln.strip()]
            i += 1
            while i < len(lines) and lines[i].strip() and not re.match(r"^(#{1,4}\s|\||```|\s*[-*]\s|\s*\d+\.\s)", lines[i]):
                para.append(lines[i].strip())
                i += 1
            out.append(("p", " ".join(para)))
    return out


INLINE = re.compile(r"(\*\*[^*]+\*\*|`[^`]+`)")


def runs(text: str) -> list[tuple[str, str]]:
    """'a **b** `c`' -> [('t','a '), ('b','b'), ('t',' '), ('c','c')]."""
    out = []
    for part in INLINE.split(text):
        if not part:
            continue
        if part.startswith("**"):
            out.append(("b", part[2:-2]))
        elif part.startswith("`"):
            out.append(("c", part[1:-1]))
        else:
            out.append(("t", part))
    return out


# ------------------------------------------------------------------ Word
def to_docx(md: str, out: Path) -> None:
    from docx import Document
    from docx.enum.table import WD_TABLE_ALIGNMENT
    from docx.oxml import OxmlElement
    from docx.oxml.ns import qn
    from docx.shared import Pt, RGBColor

    doc = Document()
    st = doc.styles["Normal"]
    st.font.name, st.font.size = "Calibri", Pt(11)
    accent = RGBColor(0x1E, 0x40, 0xAF)

    def add_runs(p, text, size=None):
        for kind, t in runs(text):
            r = p.add_run(t)
            if kind == "b":
                r.bold = True
            elif kind == "c":
                r.font.name = "Consolas"
                r.font.size = Pt(10)
            if size:
                r.font.size = Pt(size)

    def shade(cell, hexcolor):
        tcPr = cell._tc.get_or_add_tcPr()
        shd = OxmlElement("w:shd")
        shd.set(qn("w:val"), "clear")
        shd.set(qn("w:color"), "auto")
        shd.set(qn("w:fill"), hexcolor)
        tcPr.append(shd)

    for b in blocks(md):
        if b[0] == "h":
            h = doc.add_heading(level=min(b[1], 3))
            add_runs(h, b[2])
            for r in h.runs:
                r.font.color.rgb = accent
        elif b[0] == "p":
            add_runs(doc.add_paragraph(), b[1])
        elif b[0] in ("ul", "ol"):
            for item in b[1]:
                first, *rest = item.split("\n")
                p = doc.add_paragraph(style="List Bullet" if b[0] == "ul" else "List Number")
                add_runs(p, first)
                for extra in rest:
                    q = doc.add_paragraph()
                    q.paragraph_format.left_indent = Pt(36)
                    add_runs(q, extra)
        elif b[0] == "table":
            rows = b[1]
            t = doc.add_table(rows=len(rows), cols=max(len(r) for r in rows))
            t.style = "Table Grid"
            t.alignment = WD_TABLE_ALIGNMENT.LEFT
            for ri, row in enumerate(rows):
                for ci, val in enumerate(row):
                    cell = t.cell(ri, ci)
                    cell.text = ""
                    add_runs(cell.paragraphs[0], val, size=10)
                    if ri == 0:
                        for r in cell.paragraphs[0].runs:
                            r.bold = True
                        shade(cell, "DBEAFE")
            doc.add_paragraph()
        elif b[0] == "code":
            for line in b[1].split("\n"):
                p = doc.add_paragraph()
                p.paragraph_format.space_after = Pt(0)
                r = p.add_run(line)
                r.font.name, r.font.size = "Consolas", Pt(10)
    doc.save(str(out))


# ------------------------------------------------------------------ HTML -> PDF
def inline_html(text: str) -> str:
    out = ""
    for kind, t in runs(text):
        t = html.escape(t)
        out += f"<b>{t}</b>" if kind == "b" else f"<code>{t}</code>" if kind == "c" else t
    return out


def to_html(md: str, title: str) -> str:
    body = []
    for b in blocks(md):
        if b[0] == "h":
            body.append(f"<h{b[1]}>{inline_html(b[2])}</h{b[1]}>")
        elif b[0] == "p":
            body.append(f"<p>{inline_html(b[1])}</p>")
        elif b[0] in ("ul", "ol"):
            items = "".join("<li>" + "<br>".join(inline_html(x) for x in it.split("\n")) + "</li>" for it in b[1])
            body.append(f"<{b[0]}>{items}</{b[0]}>")
        elif b[0] == "table":
            rows = b[1]
            head = "".join(f"<th>{inline_html(c)}</th>" for c in rows[0])
            rest = "".join("<tr>" + "".join(f"<td>{inline_html(c)}</td>" for c in r) + "</tr>" for r in rows[1:])
            body.append(f"<table><thead><tr>{head}</tr></thead><tbody>{rest}</tbody></table>")
        elif b[0] == "code":
            body.append(f"<pre>{html.escape(b[1])}</pre>")
    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><title>{html.escape(title)}</title>
<style>
@page {{ size: A4; margin: 16mm 15mm; }}
body {{ font: 10.5pt/1.5 "Segoe UI", Calibri, Arial, sans-serif; color: #1f2937; }}
h1 {{ font-size: 20pt; color: #1e40af; margin: 0 0 4pt; }}
h2 {{ font-size: 14pt; color: #1e40af; border-bottom: 1.5px solid #bfdbfe; padding-bottom: 2pt; margin: 16pt 0 6pt; }}
h3 {{ font-size: 11.5pt; color: #1e3a8a; margin: 10pt 0 4pt; }}
h1, h2, h3 {{ break-after: avoid; page-break-after: avoid; }}
p, li {{ margin: 3pt 0; }}
table {{ border-collapse: collapse; width: 100%; margin: 6pt 0 10pt; font-size: 9.5pt; page-break-inside: avoid; }}
th, td {{ border: 1px solid #cbd5e1; padding: 4pt 6pt; text-align: left; vertical-align: top; }}
th {{ background: #dbeafe; }}
tr:nth-child(even) td {{ background: #f8fafc; }}
code, pre {{ font-family: Consolas, monospace; font-size: 9pt; background: #f1f5f9; border-radius: 3px; }}
code {{ padding: 0 3px; }}
pre {{ padding: 6pt 8pt; white-space: pre-wrap; }}
</style></head><body>{"".join(body)}</body></html>"""


def browser() -> str | None:
    for c in (r"C:\Program Files\Google\Chrome\Application\chrome.exe",
              r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
              r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
              shutil.which("chrome"), shutil.which("google-chrome"), shutil.which("chromium"), shutil.which("msedge")):
        if c and Path(c).exists():
            return str(c)
    return None


def to_pdf(md: str, title: str, out: Path) -> bool:
    exe = browser()
    if exe is None:
        print("no Chrome / Edge found: PDF skipped (the Word file is made)")
        return False
    with tempfile.TemporaryDirectory() as td:
        page = Path(td) / "doc.html"
        page.write_text(to_html(md, title), encoding="utf-8")
        subprocess.run([exe, "--headless=new", "--disable-gpu", "--no-first-run", f"--user-data-dir={td}/profile",
                        "--no-pdf-header-footer", f"--print-to-pdf={out}", page.as_uri()],
                       check=True, capture_output=True, timeout=120)
    return out.exists()


def main(src: str) -> int:
    path = Path(src)
    md = path.read_text(encoding="utf-8")
    title = next((b[2] for b in blocks(md) if b[0] == "h"), path.stem)
    docx_out, pdf_out = path.with_suffix(".docx").resolve(), path.with_suffix(".pdf").resolve()
    to_docx(md, docx_out)
    print("Word:", docx_out)
    if to_pdf(md, title, pdf_out):
        print("PDF: ", pdf_out)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else "docs/Mathpix-Fast-Reading.md"))

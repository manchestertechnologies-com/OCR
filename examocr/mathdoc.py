"""Formulas for the Word export: LaTeX -> native, editable Word equations (OMML).

  LaTeX --latex2mathml--> MathML --mathml2omml--> OMML (<m:oMath>), offline.
  mhchem \\ce{...} is first turned into standard LaTeX (ce_to_latex): latex2mathml does not
  know \\ce.
A formula that cannot be converted is written as plain Unicode text (K₂MnO₄, M/3) - the
export never fails because of one formula.
"""
from __future__ import annotations

import re

M_NS = "http://schemas.openxmlformats.org/officeDocument/2006/math"
W_NS = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
MATH_SPLIT = re.compile(r"(\$\$.+?\$\$|\$.+?\$|\\\(.+?\\\)|\\\[.+?\\\])", re.S)
_GROUP = r"\{((?:[^{}]|\{(?:[^{}]|\{[^{}]*\})*\})*)\}"


def split_math(line: str) -> list[tuple[str, str]]:
    """'a) $\\frac{M}{3}$ or x' -> [('text', 'a) '), ('math', '\\frac{M}{3}'), ('text', ' or x')]."""
    out = []
    for i, part in enumerate(MATH_SPLIT.split(line)):
        if not part:
            continue
        if i % 2:
            body = part[2:-2] if part.startswith(("$$", "\\[", "\\(")) else part[1:-1]
            out.append(("math", body.strip()))
        else:
            out.append(("text", part))
    return out


# ------------------------------------------------------------------ mhchem -> LaTeX
_ARROWS = [("<=>>", r"\rightleftharpoons"), ("<<=>", r"\rightleftharpoons"), ("<=>", r"\rightleftharpoons"),
           ("<->", r"\leftrightarrow"), ("<-", r"\leftarrow"), ("->", r"\rightarrow")]


def _ce_formula(tok: str) -> str:
    """'K2MnO4' -> 'K_{2}MnO_{4}'; 'Hg^2+' / 'Hg^{2+}' -> 'Hg^{2+}'; '2H2O' keeps the coefficient."""
    m = re.match(r"^(\d+(?:\.\d+)?|\d*/\d+)?(.*)$", tok)
    coef, body = (m.group(1) or ""), m.group(2)
    body = re.sub(r"\^\{?([0-9]*[+-])\}?", r"^{\1}", body)                  # charges
    body = re.sub(r"(?<=[A-Za-z)\]])(\d+)(?![^{]*\})", r"_{\1}", body)       # subscripts
    body = re.sub(r"(?<=[A-Za-z)\]])([+-])$", r"^{\1}", body)               # "Na+" -> Na^{+}
    return f"{coef}\\mathrm{{{body}}}" if body else coef


def ce_to_latex(s: str) -> str:
    """Replace every \\ce{...} with plain LaTeX: formulas upright with subscripts, arrows with
    their conditions (->[above][below]), "+" kept. The common exam subset; anything unusual
    is passed through as \\mathrm text, never dropped."""
    def one(body: str) -> str:
        out = []
        for tok in re.split(r"(\s+)", body.strip()):
            if not tok or tok.isspace():
                out.append(" ")
                continue
            arrow = next(((a, cmd) for a, cmd in _ARROWS if tok.startswith(a)), None)
            if arrow:
                a, cmd = arrow
                conds = re.findall(r"\[([^\]]*)\]", tok[len(a):])
                above = one(conds[0]) if conds else ""
                below = one(conds[1]) if len(conds) > 1 else ""
                if cmd == r"\rightarrow" and (above or below):
                    out.append(rf"\xrightarrow[{below}]{{{above}}}")
                elif above or below:
                    out.append(rf"\overset{{{above}}}{{{cmd}}}")
                else:
                    out.append(cmd)
            elif tok in ("+", "=", "^", "v"):
                out.append({"^": r"\uparrow", "v": r"\downarrow"}.get(tok, tok))
            else:
                out.append(_ce_formula(tok))
        return "".join(out).strip()
    prev = None
    while prev != s:
        prev = s
        s = re.sub(r"\\ce\s*" + _GROUP, lambda m: one(m.group(1)), s)
    return s


# ------------------------------------------------------------------ LaTeX -> OMML
def latex_to_omml(tex: str):
    """LaTeX formula -> lxml <m:oMath> element (raises on failure)."""
    from latex2mathml.converter import convert as to_mathml
    from lxml import etree
    import mathml2omml
    tex = ce_to_latex(tex)
    om = mathml2omml.convert(to_mathml(tex))
    if not om.lstrip().startswith("<m:oMath"):
        om = f"<m:oMath>{om}</m:oMath>"
    om = om.replace("<m:oMath", f'<m:oMath xmlns:m="{M_NS}" xmlns:w="{W_NS}"', 1)
    return etree.fromstring(om)


def add_math_runs(paragraph, line: str, font: str = "Consolas", size_pt: float = 10.5) -> None:
    """Add one line of the question to a python-docx paragraph: text as runs, every $...$ as a
    native Word equation (plain Unicode text if that one formula cannot be converted)."""
    from docx.shared import Pt

    from .ocr.latex import latex_to_text, math_spaces, no_backslash

    def text_run(t: str):
        r = paragraph.add_run(t)
        r.font.name, r.font.size = font, Pt(size_pt)
    for kind, v in split_math(line):
        if kind == "text":
            text_run(no_backslash(v))
            continue
        try:
            paragraph._p.append(latex_to_omml(v))
        except Exception:
            text_run(latex_to_text(math_spaces(ce_to_latex(v))))

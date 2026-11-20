"""LaTeX / Markdown from the OCR models -> plain, readable, editable text.

The models write formulas as LaTeX ($H_{2}O$, \\frac{M}{3}) and the page as Markdown. The
question text stays plain Unicode: H₂O, M/3, K₂MnO₄ → MnO₂, 6.27×10⁴. Nothing is invented:
an unknown command is kept as written.
"""
from __future__ import annotations

import re

SUP = dict(zip("0123456789+-=()niaxyk", "⁰¹²³⁴⁵⁶⁷⁸⁹⁺⁻⁼⁽⁾ⁿⁱᵃˣʸᵏ"))
SUB = dict(zip("0123456789+-=()aeoxhklmnpst", "₀₁₂₃₄₅₆₇₈₉₊₋₌₍₎ₐₑₒₓₕₖₗₘₙₚₛₜ"))
SYMBOLS = {
    r"\times": "×", r"\cdot": "·", r"\div": "÷", r"\pm": "±", r"\mp": "∓",
    r"\rightleftharpoons": "⇌", r"\leftrightharpoons": "⇌", r"\rightleftarrows": "⇄",
    r"\longrightarrow": "→", r"\rightarrow": "→", r"\to": "→", r"\leftarrow": "←", r"\gets": "←",
    r"\longleftarrow": "←", r"\Rightarrow": "⇒", r"\Leftarrow": "⇐", r"\Leftrightarrow": "⇔",
    r"\leftrightarrow": "↔", r"\uparrow": "↑", r"\downarrow": "↓",
    r"\leq": "≤", r"\le": "≤", r"\geq": "≥", r"\ge": "≥", r"\neq": "≠", r"\ne": "≠",
    r"\gg": "≫", r"\ll": "≪", r"\approx": "≈", r"\sim": "~", r"\simeq": "≃", r"\propto": "∝",
    r"\equiv": "≡", r"\infty": "∞", r"\therefore": "∴", r"\because": "∵",
    r"\degree": "°", r"\circ": "°", r"\prime": "′", r"\partial": "∂", r"\nabla": "∇",
    r"\int": "∫", r"\oint": "∮", r"\sum": "Σ", r"\prod": "Π", r"\sqrt": "√",
    r"\in": "∈", r"\notin": "∉", r"\subset": "⊂", r"\cup": "∪", r"\cap": "∩", r"\perp": "⊥",
    r"\parallel": "∥", r"\angle": "∠", r"\triangle": "△", r"\hbar": "ℏ", r"\AA": "Å",
    r"\alpha": "α", r"\beta": "β", r"\gamma": "γ", r"\delta": "δ", r"\epsilon": "ε",
    r"\varepsilon": "ε", r"\theta": "θ", r"\vartheta": "θ", r"\lambda": "λ", r"\mu": "μ",
    r"\nu": "ν", r"\xi": "ξ", r"\pi": "π", r"\rho": "ρ", r"\sigma": "σ", r"\tau": "τ",
    r"\phi": "φ", r"\varphi": "φ", r"\omega": "ω", r"\eta": "η", r"\kappa": "κ", r"\chi": "χ",
    r"\psi": "ψ", r"\zeta": "ζ", r"\iota": "ι", r"\upsilon": "υ",
    r"\Delta": "Δ", r"\Gamma": "Γ", r"\Theta": "Θ", r"\Lambda": "Λ", r"\Sigma": "Σ",
    r"\Phi": "Φ", r"\Omega": "Ω", r"\Psi": "Ψ", r"\Pi": "Π", r"\Xi": "Ξ",
    r"\%": "%", r"\{": "{", r"\}": "}", r"\_": "_", r"\&": "&", r"\#": "#", r"\$": "$",
    r"\ldots": "...", r"\dots": "...", r"\cdots": "...", r"\dagger": "†", r"\ddagger": "‡",
}
_GROUP = r"\{((?:[^{}]|\{(?:[^{}]|\{[^{}]*\})*\})*)\}"
_OPT = r"\[((?:[^\[\]]|\[[^\[\]]*\])*)\]"


def latex_to_text(s: str) -> str:
    """Readable plain text from LaTeX. Superscripts/subscripts become Unicode when every
    character has a Unicode form, otherwise ^(...) / _(...)."""
    s = s.strip().strip("$").strip()
    for _ in range(6):
        s2 = re.sub(r"\\(?:mathrm|mathbf|mathit|mathsf|mathtt|mathcal|text|textrm|textbf|textit|"
                    r"operatorname|mbox|boldsymbol|ce|pu|displaystyle|textstyle)\s*" + _GROUP, r"\1", s)
        s2 = re.sub(r"\\[dt]?frac\s*" + _GROUP + r"\s*" + _GROUP,
                    lambda m: f"{_paren(m.group(1))}/{_paren(m.group(2))}", s2)
        s2 = re.sub(r"\\sqrt\s*" + _OPT + r"\s*" + _GROUP,
                    lambda m: f"{_script(m.group(1), SUP, '^')}√{_paren(m.group(2))}", s2)
        s2 = re.sub(r"\\sqrt\s*" + _GROUP, lambda m: "√" + _paren(m.group(1)), s2)
        # reaction arrows with conditions: \xrightarrow[below]{above} -> →(above; below)
        s2 = re.sub(r"\\x(right|left)arrow\s*(?:" + _OPT + r")?\s*" + _GROUP,
                    lambda m: _arrow("→" if m.group(1) == "right" else "←", m.group(3), m.group(2)), s2)
        s2 = re.sub(r"\\(?:overset|stackrel)\s*" + _GROUP + r"\s*" + _GROUP,
                    lambda m: _over(m.group(2), m.group(1)), s2)
        s2 = re.sub(r"\\underset\s*" + _GROUP + r"\s*" + _GROUP,
                    lambda m: _over(m.group(2), None, m.group(1)), s2)
        # a circled option label (handwritten Ⓐ) -> "A)"
        s2 = re.sub(r"\\(?:textcircled|circled)\s*(?:" + _GROUP + r"|(\w))",
                    lambda m: f"{m.group(1) or m.group(2)})", s2)
        # accents the model adds to handwriting (\vec n, \overline{j}): keep the letter
        s2 = re.sub(r"\\(?:vec|overline|underline|hat|widehat|bar|dot|ddot|tilde|overrightarrow)\s*"
                    + _GROUP, r"\1", s2)
        s2 = re.sub(r"\\(?:vec|overline|underline|hat|widehat|bar|dot|ddot|tilde)\s*(\w)", r"\1", s2)
        if s2 == s:
            break
        s = s2
    s = re.sub(r"\\begin\{[a-z*]+\}(\{[^}]*\})?|\\end\{[a-z*]+\}", " ", s)
    s = s.replace("&", " ").replace("\\\\", "\n")
    s = re.sub(r"\\(left|right|big|Big|bigg|Bigg)\b\s*", "", s)
    s = re.sub(r"\\(?:,|;|:|!|quad|qquad|space)|~", " ", s)
    s = re.sub(r"\^\s*\{\s*\\circ\s*\}|\^\s*\\circ", "°", s)
    for cmd in sorted(SYMBOLS, key=len, reverse=True):
        s = re.sub(re.escape(cmd) + r"(?![a-z])", lambda _m, v=SYMBOLS[cmd]: v, s)
    s = re.sub(r"\^\s*" + _GROUP, lambda m: _script(m.group(1), SUP, "^"), s)
    s = re.sub(r"\^\s*(\S)", lambda m: _script(m.group(1), SUP, "^"), s)
    s = re.sub(r"_\s*" + _GROUP, lambda m: _script(m.group(1), SUB, "_"), s)
    s = re.sub(r"_\s*(\S)", lambda m: _script(m.group(1), SUB, "_"), s)
    s = s.replace("{", "").replace("}", "")
    s = re.sub(r"\s*([=+×÷⇌⇄→←⇒≤≥≠≈≫≪])\s*", r" \1 ", s)      # space around operators
    s = re.sub(r"(?<=[\d)])\s*×\s*(?=10)", "×", s)              # but keep 6.27×10⁴ together
    s = re.sub(r"(^|[(\s])\s*-\s+", r"\1-", s)                  # leading minus sign
    s = re.sub(r"([→←]) \(", r"\1(", s)                          # →(Heat; CCl₄) stays one token
    return "\n".join(re.sub(r"[ \t]+", " ", ln).strip() for ln in s.split("\n")).strip()


def _paren(x: str) -> str:
    x = latex_to_text(x)
    return x if re.fullmatch(r"[\w.]+", x) else f"({x})"


def _arrow(sym: str, above: str | None, below: str | None) -> str:
    parts = [latex_to_text(p) for p in (above, below) if p and latex_to_text(p)]
    return f" {sym}({'; '.join(parts)}) " if parts else f" {sym} "


def _over(base: str, above: str | None, below: str | None = None) -> str:
    b = latex_to_text(base)
    if b in ("→", "←", "⇌", "⇄", "=", "->"):
        return _arrow(b if b != "->" else "→", above, below)
    extra = [latex_to_text(p) for p in (above, below) if p]
    return b + "".join(f"({e})" for e in extra if e)


def _script(body: str, table: dict, mark: str) -> str:
    body = latex_to_text(body).replace(" ", "")
    if mark == "^":                     # a handwritten "+" in a charge is often read as a dagger
        body = body.replace("†", "+").replace("‡", "+")
    if body and all(c in table for c in body):
        return "".join(table[c] for c in body)
    if mark == "_" and body.isalpha():   # "H_g": the g of Hg sitting low, not a subscript
        return body
    return f"{mark}({body})" if len(body) > 1 else f"{mark}{body}"


# ------------------------------------------------------------------ Markdown page -> text
_MATH = re.compile(r"\$\$(.+?)\$\$|\\\[(.+?)\\\]|\$(.+?)\$|\\\((.+?)\\\)", re.S)
_TAG = re.compile(r"</?(?:table|thead|tbody|tr|td|th|br|p|div|span|b|i|u|sup|sub)\b[^>]*>", re.I)


def markdown_to_text(md: str) -> str:
    """A page as the model wrote it (Markdown + LaTeX + maybe an HTML table) -> plain text
    with the same lines. Formulas become Unicode; headings / bold markers are dropped."""
    md = md.replace("\r\n", "\n")
    md = re.sub(r"\\(?:textcircled|circled)\s*(?:\{(\w)\}|(\w))", lambda m: f"{m.group(1) or m.group(2)})", md)
    md = re.sub(r"<sup>(.*?)</sup>", lambda m: _script(m.group(1), SUP, "^"), md, flags=re.I | re.S)
    md = re.sub(r"<sub>(.*?)</sub>", lambda m: _script(m.group(1), SUB, "_"), md, flags=re.I | re.S)
    md = _html_table(md)
    md = _MATH.sub(lambda m: _math(next(g for g in m.groups() if g is not None),
                                   display=m.group(1) is not None or m.group(2) is not None), md)
    out = []
    for ln in md.split("\n"):
        ln = re.sub(r"^\s{0,3}#{1,6}\s+", "", ln)                  # heading
        ln = re.sub(r"(\*\*|__)(.+?)\1", r"\2", ln)                 # bold
        ln = re.sub(r"(?<![\w*])\*(?!\s)(.+?)(?<!\s)\*(?![\w*])", r"\1", ln)   # italic
        ln = re.sub(r"^\s*[-*]\s+(?=\S)", "", ln) if re.match(r"^\s*[-*]\s+\(?[A-Da-d1-4][.)]", ln) else ln
        if re.fullmatch(r"\s*\|?\s*:?-{3,}:?\s*(\|\s*:?-{3,}:?\s*)*\|?\s*", ln):
            continue                                                # markdown table rule
        if ln.strip().startswith("|") and ln.strip().endswith("|"):
            ln = "   ".join(c.strip() for c in ln.strip().strip("|").split("|"))
        if re.search(r"\\[A-Za-z]{2,}|[_^]\{", ln):                 # LaTeX written outside $...$
            ln = latex_to_text(ln)
        out.append(ln.rstrip())
    text = "\n".join(out)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip("\n")


def math_spaces(body: str) -> str:
    """In LaTeX math, spaces in the source mean nothing ("C H_3" is printed CH₃): drop them,
    except inside \\text{...} and after a command name (\\alpha x)."""
    keep: list[str] = []

    def hold(m):
        keep.append(m.group(0))
        return f"\x00{len(keep) - 1}\x00"
    s = re.sub(r"\\(?:text|textrm|textbf|textit|mbox)\s*" + _GROUP, hold, body)
    # "\equiv C" -> "\equiv{}C" (the symbol needs no space); other commands keep one
    s = re.sub(r"(\\[A-Za-z]+)\s+(?=[A-Za-z0-9])",
               lambda m: m.group(1) + "{}" if m.group(1) in SYMBOLS else hold(m), s)
    s = re.sub(r"\s+", "", s)
    return re.sub(r"\x00(\d+)\x00", lambda m: keep[int(m.group(1))], s)


def _math(body: str, display: bool) -> str:
    t = latex_to_text(math_spaces(body))
    return f"\n{t}\n" if display else t


def _html_table(md: str) -> str:
    """<table><tr><td>..</td></tr></table> -> one line per row, cells separated by 3 spaces."""
    def table(m):
        rows = re.findall(r"<tr[^>]*>(.*?)</tr>", m.group(0), flags=re.I | re.S)
        lines = []
        for r in rows:
            cells = re.findall(r"<t[dh][^>]*>(.*?)</t[dh]>", r, flags=re.I | re.S)
            lines.append("   ".join(_TAG.sub("", c).strip() for c in cells))
        return "\n" + "\n".join(lines) + "\n"
    md = re.sub(r"<table\b.*?</table>", table, md, flags=re.I | re.S)
    return _TAG.sub("", md)

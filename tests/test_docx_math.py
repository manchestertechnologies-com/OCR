"""Formulas in the exports: LaTeX source -> native Word equations; mhchem -> LaTeX."""
import io
import zipfile

from examocr.mathdoc import ce_to_latex, latex_to_omml, split_math


def test_split_math():
    assert split_math(r"a) $\frac{M}{3}$ or $$x^2$$ end") == [
        ("text", "a) "), ("math", r"\frac{M}{3}"), ("text", " or "), ("math", "x^2"), ("text", " end")]
    assert split_math("no maths here") == [("text", "no maths here")]


def test_ce_to_latex():
    assert ce_to_latex(r"\ce{K2MnO4}") == r"\mathrm{K_{2}MnO_{4}}"
    assert ce_to_latex(r"\ce{Hg^2+}") == r"\mathrm{Hg^{2+}}"
    assert ce_to_latex(r"\ce{2KMnO4 ->[\Delta] K2MnO4 + MnO2 + O2}") == (
        r"2\mathrm{KMnO_{4}} \xrightarrow[]{\mathrm{\Delta}} \mathrm{K_{2}MnO_{4}} + \mathrm{MnO_{2}} + \mathrm{O_{2}}")
    assert r"\rightleftharpoons" in ce_to_latex(r"\ce{N2 + 3H2 <=> 2NH3}")


def test_latex_to_omml():
    for tex in (r"\frac{M}{3}", r"\sec^{-1} mol^{-1} L", r"\sqrt{2gh}", r"\ce{K2MnO4 + MnO2}",
                r"-6.27\times10^{4}\,J"):
        el = latex_to_omml(tex)
        assert el.tag.endswith("}oMath"), tex
    xml = __import__("lxml.etree", fromlist=["tostring"]).tostring(latex_to_omml(r"\frac{M}{3}")).decode()
    assert "<m:f>" in xml and "<m:num>" in xml                      # a real fraction, not text


def test_docx_has_native_equations(tmp_path):
    from examocr.export import build_docx
    from examocr.schemas import Box, Project, QuestionRecord
    q = QuestionRecord(id="q1", exam="NEET", page=1, box=Box(x0=0, y0=0, x1=10, y1=10), crop_path="",
                       crop_id="", text="3) mass M/3", original_sha256="",
                       display_latex="3) Equivalent mass is $\\frac{M}{3}$\n"
                                     "$$\\ce{2KMnO4 ->[\\Delta] K2MnO4 + MnO2 + O2}$$\n"
                                     "broken: $\\frac{M$ stays text")
    pr = Project(id="p1", exam="NEET", source_name="t.pdf", created="now", questions=[q])
    data = build_docx(pr, tmp_path)
    doc = zipfile.ZipFile(io.BytesIO(data)).read("word/document.xml").decode("utf-8")
    assert doc.count("<m:oMath") >= 2                               # fraction + reaction
    assert "<m:f>" in doc
    assert "\\frac" not in doc and "\\ce" not in doc                # no LaTeX left in the file
    assert "Equivalent mass is" in doc and "stays text" in doc

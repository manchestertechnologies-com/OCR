"""GLM-OCR reading path and the layout-model figure finder: the logic around the models.
None of these load a model (the end-to-end tests in test_manual_crop.py use the real ones
when they are installed)."""
import numpy as np

from examocr.analyze import FoundGraphic, contains_next, find_q_no, place_figures
from examocr.layout.docmodel import has_arrow, select_figures
from examocr.ocr.glmocr import prepare, repeat_tail, squeeze_blank, split_bands
from examocr.ocr.latex import latex_to_text, markdown_to_text
from examocr.schemas import Box, RegionKind


def test_markdown_and_latex_to_plain_text():
    md = ("2) Match the items\n**a)** $Q = k$\n"
          "$$\\text{NBS} / \\text{AIBN} \\xrightarrow[\\text{CCl}_4]{\\text{Heat}} \\text{A} + \\text{B}$$\n"
          "| A | B |\n| --- | --- |\n| 1 | 2 |\n"
          "$2KMnO_{4} \\overset{\\Delta}{\\longrightarrow} K_{2}MnO_{4}+O_{2}$ and $\\frac{M}{3}$")
    out = markdown_to_text(md).splitlines()
    assert out[0] == "2) Match the items"
    assert out[1] == "a) Q = k"
    assert "NBS/AIBN →(Heat; CCl₄) A + B" in out          # math: source spaces don't count
    assert "A   B" in out and "1   2" in out
    assert out[-1] == "2KMnO₄ →(Δ) K₂MnO₄ + O₂ and M/3"


def test_math_spaces_and_glued_commands():
    # in LaTeX math, source spaces mean nothing: "C H_3" is printed CH₃
    assert markdown_to_text(r"$\mathrm{C H_{3} - C \equiv C H}$") == "CH₃-C≡CH"
    assert markdown_to_text(r"$CH_{3}-C\equivCH$") == "CH₃-C≡CH"          # command glued to a capital
    assert markdown_to_text(r"\textcircledA 1 2 3 4") == "A) 1 2 3 4"     # circled option label
    assert markdown_to_text(r"Reaction at $\text{equal rates}$") == "Reaction at equal rates"
    assert latex_to_text(r"\infty \int x") == "∞ ∫ x"


def test_select_figures():
    lh = 30.0
    dets = [
        ("image", 0.8, Box(x0=100, y0=100, x1=200, y1=220)),          # a drawing
        ("image", 0.4, Box(x0=105, y0=95, x1=210, y1=230)),           # same drawing again -> merged
        ("figure_title", 0.6, Box(x0=140, y0=235, x1=160, y1=260)),   # its caption "A" -> joins it
        ("image", 0.3, Box(x0=90, y0=90, x1=500, y1=240)),            # weak group around 2 -> dropped
        ("image", 0.75, Box(x0=300, y0=100, x1=400, y1=220)),
        ("image", 0.33, Box(x0=0, y0=300, x1=120, y1=360)),           # weak and 2 lines tall: text
        ("formula", 0.5, Box(x0=0, y0=400, x1=250, y1=520)),          # 4 lines tall: a drawn structure
        ("formula", 0.7, Box(x0=300, y0=400, x1=360, y1=460)),        # a fraction: text
        ("text", 0.9, Box(x0=0, y0=0, x1=600, y1=30)),
    ]
    figs = select_figures(dets, lh, 600, 600)
    boxes = [(b.x0, b.y0, b.x1, b.y1) for _, b, _ in figs]
    assert (100, 95, 210, 260) in boxes                                # merged + caption
    assert (300, 100, 400, 220) in boxes
    assert (0, 400, 250, 520) in boxes and [f for f in figs if f[1].y0 == 400][0][2]   # formula_only
    assert len(figs) == 3


def test_reaction_arrow_is_text():
    reagents = np.zeros((90, 300), bool)
    reagents[10:30, 20:280:30] = True                 # letters above
    reagents[44:47, 10:290] = True                    # the arrow
    reagents[60:80, 20:280:30] = True                 # letters below
    assert has_arrow(reagents)
    structure = np.zeros((120, 200), bool)
    for k in range(60):                               # diagonal bonds and a short "="
        structure[20 + k, 20 + k] = structure[100 - k, 120 + k] = True
    structure[58:60, 90:110] = structure[64:66, 90:110] = True
    assert not has_arrow(structure)


def test_squeeze_blank_keeps_lines_and_drops_paper():
    img = np.full((600, 800, 3), 255, np.uint8)
    img[100:120, 100:500] = 0                         # line 1
    img[400:420, 100:300] = 0                         # line 2, after a blanked drawing
    out = squeeze_blank(img, 12)
    dark = (out[:, :, 0] < 128).any(axis=1)
    assert out.shape[0] < 80 and out.shape[1] <= 404
    assert dark.sum() == 40                           # both lines kept whole
    p = prepare(img, 12, target=16)
    assert p.shape[0] >= 56 and p.shape[1] >= 56


def test_split_bands_only_at_blank_rows():
    img = np.full((1200, 400, 3), 255, np.uint8)
    for y in range(20, 1180, 40):
        img[y:y + 20, 20:380] = 0                     # 29 lines with blank rows between
    bands = split_bands(img, 12, max_lines=7)
    assert len(bands) > 1
    assert sum(int((b[:, :, 0] < 128).any(axis=1).sum()) for b in bands) == 29 * 20
    tilted = np.full((1200, 400, 3), 255, np.uint8)
    tilted[:, 10] = 0                                 # ink in every row: never cut
    assert len(split_bands(tilted, 12, max_lines=7)) == 1


def test_repeat_tail():
    ids = list(range(10)) + [5, 6, 7] * 20
    assert repeat_tail(ids) == 57
    assert repeat_tail(list(range(80))) == 0


def test_question_number_and_next_question():
    assert find_q_no("1 In the following reaction") == 1
    assert find_q_no("[1] In the following") == 1
    assert find_q_no("c) 2:1:1\nd) 3:2:1\n2) Match the items") == 2
    assert find_q_no("Q.12 What is") == 12
    assert find_q_no("The major product") is None
    match = "2) Match\ncolumn I\nA) Q=k\n1. Reaction a\n2. Reaction b\n3. Reaction c\n4. eq"
    assert not contains_next(match, 2)                # a numbered list, not question 3
    assert contains_next("5) What\nis x\na) 1 b) 2\n6) Which", 5)


def test_option_labels_pair_with_drawings():
    ocr_in = np.full((400, 800, 3), 255, np.uint8)
    lines = ["5) The major product Q is", "A)", "B)", "C)", "D)"]
    boxes = [Box(x0=100, y0=100, x1=300, y1=200), Box(x0=500, y0=100, x1=700, y1=200),
             Box(x0=100, y0=250, x1=300, y1=350), Box(x0=500, y0=250, x1=700, y1=350)]
    graphics = [FoundGraphic(kind=RegionKind.diagram, box=b) for b in boxes]
    out = place_figures(lines, ocr_in, graphics, boxes)
    assert out == ["5) The major product Q is", "A) {{fig:1}}", "B) {{fig:2}}", "C) {{fig:3}}", "D) {{fig:4}}"]
    assert all(g.kind == RegionKind.option_image for g in graphics)
    # a scheme above the options stays a diagram: labels take the LAST drawings
    scheme = Box(x0=100, y0=10, x1=700, y1=60)
    graphics = [FoundGraphic(kind=RegionKind.diagram, box=b) for b in [scheme] + boxes]
    out = place_figures(lines, ocr_in, graphics, [scheme] + boxes)
    assert graphics[0].kind == RegionKind.diagram and out[1] == "A) {{fig:2}}"

"""Handwriting path: segmentation, formula text, option labels, print/handwriting detection.
None of these need the TrOCR models (those are tested by hand on real papers)."""
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from examocr.ocr.handseg import binarize, segment
from examocr.ocr.trocr import _fill_sequence, choose_math, latex_to_text, normalize_label
from examocr.parse import parse_question


def _font(size):
    for f in ("C:/Windows/Fonts/arial.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(f, size)
        except OSError:
            continue
    return ImageFont.load_default(size)


def test_latex_to_plain_text():
    assert latex_to_text(r"H_{2}+Hg^{2+}\rightleftharpoons2Hg+2H^{+}") == "H₂ + Hg²⁺ ⇌ 2Hg + 2H⁺"
    assert latex_to_text(r"-6.27\times10^{4}J") == "-6.27×10⁴ J"            # unit after a power: spaced
    assert latex_to_text(r"K_{2}MnO_{4}+MnO_{2}+O_{2}") == "K₂MnO₄ + MnO₂ + O₂"
    assert latex_to_text(r"\frac{M}{3}") == "M/3"
    assert latex_to_text(r"25^{\circ}C") == "25°C"
    assert latex_to_text(r"2H_{g}") == "2Hg"                 # a low letter, not a subscript
    assert latex_to_text(r"Hg^{2\dagger}") == "Hg²⁺"         # handwritten + read as a dagger
    assert latex_to_text(r"\vec{n}") == "n"


def test_formula_or_words():
    # measured on real pieces: formulas -> math model sure and clearly surer
    assert choose_math("2.Hg.t2H.", 0.72, r"2H_{g}+2H^{-1}", 0.90)
    assert not choose_math("with the bell", 0.96, r"w_{1}H_{1}H_{u}\cdot U_{1}", 0.77)
    assert not choose_math("fractions but else", 0.30, r"mentionedi\zeta_{i}^{i}", 0.85)


def test_option_labels():
    assert normalize_label("a )") == "a)"
    assert normalize_label("7]") == "7]"
    assert normalize_label("d", alone=True) == "d)"
    assert normalize_label("d") is None
    assert normalize_label("by", alone=True) is None
    labels = {1: "a)", 2: None, 3: "c)", 4: None}
    _fill_sequence(labels, [1, 2, 3, 4])
    assert labels == {1: "a)", 2: "b)", 3: "c)", 4: "d)"}
    labels = {1: None, 2: "B)", 3: "97.", 4: "D)"}          # "97." among letters is a misread
    _fill_sequence(labels, [1, 2, 3, 4])
    assert labels[1] == "A)" and labels[3] == "C)"
    labels = {1: "a)", 2: None, 3: None}                     # one hit is not enough
    _fill_sequence(labels, [1, 2, 3])
    assert labels[2] is None


def test_lowercase_options_parse():
    opts = parse_question("7) The work done will be\na) -6.27×10⁴J  b) -6.27×10⁵J\nc) -3.13×10⁴J  d) -1.25×10⁴J")["options"]
    assert [o[:2] for o in opts] == ["a)", "b)", "c)", "d)"]


def _tilted_page():
    img = Image.new("L", (1400, 360), 255)
    d = ImageDraw.Draw(img)
    f, small = _font(44), _font(26)
    d.text((40, 40), "Hg", font=f, fill=0)
    d.text((100, 28), "2+", font=small, fill=0)              # exponent above the line
    d.text((160, 40), "+ 2H", font=f, fill=0)
    d.text((300, 40), "is the cell reaction at", font=f, fill=0)
    d.text((40, 160), "a) 6.27 x 10", font=f, fill=0)
    d.text((900, 160), "b) 1.25 x 10", font=f, fill=0)       # a second column
    arr = np.array(img)
    m = cv2.getRotationMatrix2D((700, 180), 1.5, 1.0)        # a tilted photo
    return cv2.warpAffine(arr, m, (1400, 360), borderValue=255)


def test_segment_lines_scripts_and_columns():
    lay = segment(binarize(_tilted_page()))
    lines = {}
    for p in lay.pieces:
        lines.setdefault(p.line, []).append(p)
    # the exponent "2+" belongs to the first line: not a third line, not a piece of its own
    assert len(lines) == 2, [(p.line, p.box) for p in lay.pieces]
    assert all(p.box.h >= 0.6 * lay.line_h for p in lay.pieces), [(p.line, p.box) for p in lay.pieces]
    bottom = max(lines.values(), key=lambda ps: min(p.box.y0 for p in ps))
    assert len(bottom) == 2                                  # the a) and b) columns stay apart


def test_printed_page_is_not_handwritten():
    from examocr.analyze import looks_handwritten
    from examocr.ocr.tesseract import TesseractEngine
    img = np.full((300, 1600, 3), 255, np.uint8)
    pil = Image.fromarray(img)
    d = ImageDraw.Draw(pil)
    for i, t in enumerate(["1. A body of mass 2 kg moves with velocity 3 m/s.",
                           "Its kinetic energy is", "(1) 6 J   (2) 9 J   (3) 12 J   (4) 18 J"]):
        d.text((40, 30 + 80 * i), t, font=_font(40), fill=(0, 0, 0))
    img = np.array(pil)
    assert not looks_handwritten(img, TesseractEngine().read(img).confidence)


def test_photo_background_is_not_ink():
    """A phone photo: dark keyboard on the right, cloth on top. Not paper -> not ink."""
    from examocr.paper import background_mask, paper_only
    img = np.full((1200, 900, 3), 235, np.uint8)
    img[:, 760:] = (40, 40, 45)                       # keyboard
    img[:90, 500:] = (60, 40, 150)                    # red cloth
    pil = Image.fromarray(img)
    ImageDraw.Draw(pil).text((60, 300), "1] In the following reaction", font=_font(40), fill=(30, 30, 30))
    img = np.array(pil)
    m = background_mask(img)
    assert m[600, 850] and m[40, 700]                 # keyboard and cloth found
    assert not m[300:360, 60:700].any()               # the writing is paper
    clean = paper_only(img, m)
    assert (clean[:, 780:] == 255).all() and (clean[320, 60:700] < 120).any()
    flat = np.full((800, 600, 3), 240, np.uint8)      # a scan: nothing to remove
    assert not background_mask(flat).any()

"""Rough user boxes on a two-column page - the cases from a real CET scan:
heading sliced by the box top, green question number left of the box, dotted column divider,
and a box that runs into the next question."""
import shutil

import numpy as np
import pytest
from PIL import Image, ImageDraw

from make_sample_paper import font

pytestmark = pytest.mark.skipif(
    not shutil.which("tesseract") and not __import__("os").path.exists(
        r"C:\Program Files\Tesseract-OCR\tesseract.exe"), reason="tesseract not installed")

GREEN = (40, 150, 70)


def two_column_page():
    W, H = 2600, 2000
    img = Image.new("RGB", (W, H), "white")
    d = ImageDraw.Draw(img)
    f, fh = font(42), font(60)
    d.text((70, 100), "6.1 Thermodynamic Terms", fill=GREEN, font=fh)          # heading
    left = [(1, ["System in which there is no exchange of matter,",
                 "work or energy from surroundings is",
                 "(a) closed            (b) adiabatic",
                 "(c) isolated          (d) isothermal."]),
            (2, ["When the system does not exchange heat with",
                 "the surroundings, the process is",
                 "(a) isothermal        (b) adiabatic",
                 "(c) thermal           (d) isochoric."])]
    y = 200
    for n, lines in left:
        d.text((70, y), f"{n}.", fill=GREEN, font=f)                            # green number
        for ln in lines:
            d.text((150, y), ln, fill="black", font=f)
            y += 58
        y += 40
    for yy in range(80, 1900, 14):                                              # dotted divider
        d.ellipse([1255, yy, 1260, yy + 5], fill=(110, 110, 110))
    y = 200
    for n, lines in [(8, ["Work done on an ideal gas in a cylinder",
                          "when it is compressed is shown below:",
                          "(a) 10 J      (b) 20 J",
                          "(c) 30 J      (d) 40 J"]),
                     (9, ["For an isothermal reversible expansion,",
                          "the value of q is given by",
                          "(a) 1 J       (b) 2 J",
                          "(c) 3 J       (d) 4 J"])]:
        d.text((1300, y), f"{n}.", fill=GREEN, font=f)
        for ln in lines:
            d.text((1380, y), ln, fill="black", font=f)
            y += 58
        y += 40
    return np.array(img)[:, :, ::-1].copy()        # RGB -> BGR like cv2


@pytest.fixture(scope="module")
def page():
    return two_column_page()


def _snap_and_read(page, box):
    from examocr.analyze import analyze_part
    from examocr.schemas import Box
    from examocr.snap import snap_box
    b, notes = snap_box(page, Box(x0=box[0], y0=box[1], x1=box[2], y1=box[3]))
    return b, notes, analyze_part(page, 1, b)


def test_heading_slice_divider_and_number(page):
    # box starts inside the heading, misses the "1." and ends on the dotted divider
    b, notes, pa = _snap_and_read(page, (140, 140, 1262, 435))
    assert "trimmed a partial line above the question" in notes
    assert "removed column divider on the right" in notes
    assert "included the question number on the left" in notes
    assert b.x0 <= 80 and b.x1 < 1255
    gray = page[b.y0:b.y1, b.x0:b.x1].min(axis=2)
    assert b.y0 > 166, "no ink of the heading (incl. the 'y' descender) may remain in the crop"
    assert (gray[:5] < 180).sum() == 0
    assert pa.q_no == 1
    assert "Thermo" not in pa.text and "|" not in pa.text and "[?]" not in pa.text
    assert pa.text.splitlines()[0].endswith("exchange of matter,")


def test_box_into_next_question_is_cut(page):
    # bottom edge cuts through "2. When the system..." -> that line is the next question
    b, notes, pa = _snap_and_read(page, (60, 190, 1240, 500))     # Q2's line is at y 472-515
    assert "left out the next question's first line" in notes or "trimmed a partial line below the question" in notes
    assert pa.q_no == 1 and "When the system" not in pa.text


def test_right_column_box_keeps_its_own_first_line(page):
    # top edge cuts most of "8. Work done..." away; the number line must be kept,
    # and nothing from the left column / divider may come in
    b, notes, pa = _snap_and_read(page, (1360, 222, 2500, 435))
    assert pa.q_no == 8
    assert b.x0 > 1262
    assert pa.text.splitlines()[0].startswith("8.") or "Work done" in pa.text.splitlines()[0]
    assert "isothermal" not in pa.text and "|" not in pa.text


def test_snap_never_breaks_a_clean_box(page):
    b, notes, pa = _snap_and_read(page, (60, 190, 1240, 425))
    assert pa.q_no == 1 and len(pa.text.splitlines()) == 4

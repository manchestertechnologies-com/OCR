"""Generate a synthetic NEET-style test page (A4 @ 300 DPI) with MCQs, an assertion-reason
question, a match-the-following question and a diagram with dashed lines, arrows and a '?'.
For smoke tests only - always validate on REAL papers.

  python tools/make_sample_paper.py data/samples/sample_neet.png
"""
from __future__ import annotations

import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


def font(size: int):
    for f in ("C:/Windows/Fonts/arial.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
              "/System/Library/Fonts/Supplemental/Arial.ttf"):
        if Path(f).exists():
            return ImageFont.truetype(f, size)
    return ImageFont.load_default(size)


def make(path: Path) -> Path:
    W, H = 2480, 3508
    img = Image.new("RGB", (W, H), "white")
    d = ImageDraw.Draw(img)
    f, fb, fh = font(40), font(46), font(56)
    x, y = 180, 160

    def line(t, fnt=f, dx=0, gap=62):
        nonlocal y
        d.text((x + dx, y), t, fill="black", font=fnt)
        y += gap

    line("NEET MOCK TEST - 2026", fh, dx=650, gap=90)
    line("Instructions: Each question carries 4 marks. Read carefully.", f, gap=110)
    line("PHYSICS", fb, dx=950, gap=100)

    line("1. A body of mass 2 kg moves with velocity 3 m/s. Its kinetic energy is")
    line("(1) 6 J        (2) 9 J        (3) 12 J        (4) 18 J", dx=60, gap=110)

    line("2. Given below are two statements:")
    line("Assertion (A): The speed of light in vacuum is constant.", dx=60)
    line("Reason (R): Light is an electromagnetic wave.", dx=60)
    line("(1) Both A and R are true and R explains A", dx=60)
    line("(2) Both A and R are true but R does not explain A", dx=60)
    line("(3) A is true but R is false", dx=60)
    line("(4) A is false but R is true", dx=60, gap=110)

    line("3. In the figure shown below, find the direction of the force on the block.")
    # Diagram: block on incline, dashed normal, arrows, labels and a question mark
    ox, oy = x + 400, y + 40
    d.polygon([(ox, oy + 420), (ox + 700, oy + 420), (ox, oy + 20)], outline="black", width=4)
    d.rectangle([ox + 150, oy + 170, ox + 290, oy + 280], outline="black", width=4)
    for k in range(0, 156, 26):                              # dashed normal line
        d.line([(ox + 220, oy + 170 - k), (ox + 220, oy + 170 - k - 13)], fill="black", width=3)
    d.line([(ox + 290, oy + 225), (ox + 520, oy + 225)], fill="black", width=4)   # arrow
    d.polygon([(ox + 520, oy + 205), (ox + 560, oy + 225), (ox + 520, oy + 245)], fill="black")
    d.text((ox + 580, oy + 200), "F = ?", fill="black", font=f)
    d.text((ox + 40, oy + 360), "30°", fill="black", font=f)
    d.text((ox + 200, oy + 190), "m", fill="black", font=f)
    y = oy + 480
    line("(1) Up the incline     (2) Down the incline", dx=60)
    line("(3) Perpendicular       (4) Zero", dx=60, gap=110)

    line("4. Match Column I with Column II")
    line("Column I                         Column II", dx=60)
    line("(P) Force                        (i) Joule", dx=60)
    line("(Q) Work                         (ii) Newton", dx=60)
    line("(R) Power                        (iii) Watt", dx=60)
    line("(1) P-ii, Q-i, R-iii      (2) P-i, Q-ii, R-iii", dx=60)
    line("(3) P-iii, Q-i, R-ii      (4) P-ii, Q-iii, R-i", dx=60, gap=110)

    line("5. The SI unit of electric charge is")
    line("(1) Ampere      (2) Coulomb      (3) Volt      (4) Ohm", dx=60)

    path.parent.mkdir(parents=True, exist_ok=True)
    img.save(path)  # PNG, lossless
    return path


if __name__ == "__main__":
    out = Path(sys.argv[1] if len(sys.argv) > 1 else "data/samples/sample_neet.png")
    print(make(out))

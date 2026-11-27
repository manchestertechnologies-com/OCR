"""Trial: read the questions you already cropped with Mathpix, and compare speed and text.

    python tools/mathpix_trial.py                    # lists your papers
    python tools/mathpix_trial.py <paper id>         # every question box of that paper
    python tools/mathpix_trial.py <paper id> --max 5 # only the first 5 boxes

For each box: seconds Mathpix took (network included), Mathpix's text, and the text the app
read on this computer - so you can decide on Fast mode with your own papers. Nothing in the
paper is changed. Each box is one Mathpix image (≈ $0.002) and counts in the usage counter.
Needs the key: setx MATHPIX_APP_ID "..." and setx MATHPIX_APP_KEY "..."
"""
from __future__ import annotations

import argparse
import statistics
import sys
import textwrap
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def main() -> int:
    from examocr import store, usage
    from examocr.crop import cut
    from examocr.ocr import mathpix
    from examocr.ocr.latex import markdown_to_text
    from examocr.paper import paper_only

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("paper", nargs="?")
    ap.add_argument("--max", type=int, default=0)
    a = ap.parse_args()
    if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
        sys.stdout.reconfigure(encoding="utf-8")
    if a.paper is None:
        for p in store.list_projects():
            print(f"{p['id']}  {p['source_name']:40s} {p['questions']} questions")
        return 0
    if mathpix.credentials() is None:
        print("No Mathpix key. In PowerShell:  setx MATHPIX_APP_ID \"...\"   setx MATHPIX_APP_KEY \"...\"")
        return 1
    pr = store.load(a.paper)
    boxes = [(q, p) for q in pr.questions for p in q.parts]
    if a.max:
        boxes = boxes[:a.max]
    times = []
    for q, part in boxes:
        page = store.page_image(pr.id, store._page(pr, part.page))
        img = cut(paper_only(page), part.box)
        print("=" * 100)
        print(f"Q{q.q_no if q.q_no is not None else '?'}  (page {part.page}, box {part.box.w}x{part.box.h} px)")
        try:
            r = mathpix.read_image(img)
        except mathpix.MathpixError as e:
            print("  Mathpix failed:", e)
            continue
        usage.record(1)
        times.append(r.seconds)
        mtext = markdown_to_text(mathpix.IMAGE_LINK.sub("[diagram]", mathpix.SMILES_TAG.sub(r"[structure: \1]", r.text)))
        print(f"  Mathpix: {r.seconds:.2f} s · confidence {r.confidence:.2f} · {len(r.figures)} figure(s)"
              + (" · handwritten" if r.handwritten else ""))
        print(textwrap.indent(mtext.strip() or "(no text)", "    | "))
        print("  Read on this computer before:")
        print(textwrap.indent((part.text or "(no text)").strip(), "    : "))
    if times:
        print("=" * 100)
        print(f"{len(times)} boxes · Mathpix seconds: median {statistics.median(times):.2f} · "
              f"fastest {min(times):.2f} · slowest {max(times):.2f}")
        st = usage.status()
        print(f"Usage this month: {st['images']} images · ${st['usd']} (≈ ₹{st['inr']})")
    return 0


if __name__ == "__main__":
    sys.exit(main())

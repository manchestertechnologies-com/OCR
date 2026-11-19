"""Pick blur thresholds from YOUR papers instead of a fixed number.

  training/datasets/quality/good/*.png|jpg     pages a human says are readable
  training/datasets/quality/blurry/*.png|jpg   pages a human says are too blurry

  python training/eval/calibrate_quality.py
Prints the threshold that best separates the two sets, to put in configs/default.yaml.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from examocr.ingest import load_pages  # noqa: E402
from examocr.quality import blur_score  # noqa: E402


def scores(folder: Path) -> list[float]:
    out = []
    for f in sorted(folder.glob("*")):
        if f.suffix.lower() in {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".pdf"}:
            out += [blur_score(p) for p in load_pages(f)]
    return out


def main():
    base = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "training/datasets/quality"
    good, bad = scores(base / "good"), scores(base / "blurry")
    if not good or not bad:
        sys.exit(f"Put sample pages in {base}/good and {base}/blurry")
    cands = np.unique(np.concatenate([good, bad]))
    best_t, best_acc = None, -1.0
    for t in cands:
        acc = (np.mean(np.array(good) >= t) + np.mean(np.array(bad) < t)) / 2
        if acc > best_acc:
            best_t, best_acc = float(t), acc
    print(f"good:   n={len(good)} median={np.median(good):.1f} min={min(good):.1f}")
    print(f"blurry: n={len(bad)} median={np.median(bad):.1f} max={max(bad):.1f}")
    print(f"suggested quality.blur_page_min = {best_t:.1f}  (balanced accuracy {best_acc:.1%})")
    print(f"suggested quality.blur_region_min = {best_t * 0.7:.1f}")
    print(f"suggested quality.rescan_page_min = {float(np.percentile(bad, 25)):.1f}")


if __name__ == "__main__":
    main()

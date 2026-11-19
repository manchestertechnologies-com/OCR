"""Turn reviewed projects into a YOLO dataset (page images + normalised boxes).

Only projects that are 'ready' are used; superseded crops are skipped, so the boxes are the
reviewer-corrected ones. Pages are split train/val by paper (not by page) to avoid leakage.

  python training/layout/export_yolo_dataset.py --out training/datasets/exam_layout --val 0.15
"""
from __future__ import annotations

import argparse
import random
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import yaml  # noqa: E402

from examocr import store  # noqa: E402

CLASSES = ["question", "diagram", "table", "passage", "section", "header", "option_image"]
MAP = {"question": 0, "diagram": 1, "table": 2, "passage": 3, "section": 4, "header": 5,
       "instructions": 5, "option_image": 6}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=ROOT / "training/datasets/exam_layout")
    ap.add_argument("--val", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    projects = [p for p in store.list_projects() if p["status"] == "ready"]
    if not projects:
        sys.exit("No ready projects. Process and review some papers first.")
    random.Random(a.seed).shuffle(projects)
    n_val = max(1, int(len(projects) * a.val)) if len(projects) > 1 else 0
    for split in ("train", "val"):
        (a.out / "images" / split).mkdir(parents=True, exist_ok=True)
        (a.out / "labels" / split).mkdir(parents=True, exist_ok=True)

    count = 0
    for i, meta in enumerate(projects):
        split = "val" if i < n_val else "train"
        pr = store.load(meta["id"])
        for pg in pr.pages:
            lines = []
            for c in pr.crops:
                if c.page != pg.page or c.superseded or c.kind.value not in MAP:
                    continue
                b = c.box
                cx, cy = (b.x0 + b.x1) / 2 / pg.width, (b.y0 + b.y1) / 2 / pg.height
                lines.append(f"{MAP[c.kind.value]} {cx:.6f} {cy:.6f} {b.w / pg.width:.6f} {b.h / pg.height:.6f}")
            if not lines:
                continue
            name = f"{pr.id}_p{pg.page:03d}"
            shutil.copy2(store.project_dir(pr.id) / pg.path, a.out / "images" / split / f"{name}.png")
            (a.out / "labels" / split / f"{name}.txt").write_text("\n".join(lines))
            count += 1

    cfg = {"path": str(a.out.resolve()), "train": "images/train", "val": "images/val",
           "names": dict(enumerate(CLASSES))}
    (a.out / "data.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
    print(f"Exported {count} pages from {len(projects)} papers to {a.out}")


if __name__ == "__main__":
    main()

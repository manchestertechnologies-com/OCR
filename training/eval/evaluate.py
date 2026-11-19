"""Accuracy report for one processed paper against hand-made ground truth.

Ground truth JSON (one per paper):
  {"questions": [{"q_no": 1, "page": 1, "box": [x0, y0, x1, y1], "text": "...", "type": "mcq"}, ...]}

Metrics (from the project plan):
  - CER / WER on OCR text (jiwer)
  - crop accuracy: share of questions whose crop has IoU >= 0.9 with the true box and
    covers >= 99% of it (holds the whole question, nothing much extra)
  - type accuracy
  - review stats: share flagged, and how many flagged ones actually needed a fix

  python training/eval/evaluate.py <project_id> ground_truth.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import jiwer  # noqa: E402

from examocr import store  # noqa: E402
from examocr.schemas import Box  # noqa: E402
from examocr.verify import roundtrip  # noqa: E402


def iou(a: Box, b: Box) -> float:
    inter = a.intersection(b)
    return inter / max(1, a.area + b.area - inter)


def main():
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    pr = store.load(sys.argv[1])
    gt = json.loads(Path(sys.argv[2]).read_text(encoding="utf-8"))["questions"]
    by_no = {q.q_no: q for q in pr.questions if q.q_no is not None}

    refs, hyps, crop_ok, type_ok, missing = [], [], 0, 0, []
    flagged_needed = flagged = 0
    for g in gt:
        q = by_no.get(g["q_no"])
        if q is None:
            missing.append(g["q_no"])
            continue
        refs.append(g["text"])
        hyps.append(q.text)
        tb = Box(x0=g["box"][0], y0=g["box"][1], x1=g["box"][2], y1=g["box"][3])
        if q.page == g["page"] and iou(q.box, tb) >= 0.9 and q.box.intersection(tb) >= 0.99 * tb.area:
            crop_ok += 1
        type_ok += q.question_type.value == g.get("type", q.question_type.value)
        if q.needs_review:
            flagged += 1
            flagged_needed += q.text.strip() != g["text"].strip()

    n = len(gt)
    rt = roundtrip(pr)
    report = {
        "questions_gt": n,
        "detected": n - len(missing),
        "missing_q_nos": missing,
        "extra_detections": len(pr.questions) - (n - len(missing)),
        "CER": round(jiwer.cer(refs, hyps), 4) if refs else None,
        "WER": round(jiwer.wer(refs, hyps), 4) if refs else None,
        "crop_accuracy": round(crop_ok / n, 4) if n else None,
        "type_accuracy": round(type_ok / max(1, n - len(missing)), 4),
        "review_rate": round(sum(q.needs_review for q in pr.questions) / max(1, len(pr.questions)), 4),
        "review_precision": round(flagged_needed / flagged, 4) if flagged else None,
        "roundtrip_ok": rt["ok"],
        "lost_ink_regions": sum(p["lost_ink_count"] for p in rt["pages"]),
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

"""YOLOv8 layout detector (optional). Train it with training/layout/train_yolo.py.

Class names expected (from training/layout/data.yaml):
  0 question, 1 option, 2 diagram, 3 table, 4 equation, 5 passage, 6 section, 7 header
Only graphic classes (diagram, table, equation-as-image) are used here; question
segmentation still runs on text markers, which is more reliable for numbering.
"""
from __future__ import annotations

import logging
from functools import lru_cache
from pathlib import Path

import cv2

from ..config import ROOT, settings
from ..preprocess import WorkingCopy
from ..schemas import Box, RegionKind, Word
from .types import Graphic

log = logging.getLogger(__name__)

_GRAPHIC = {"diagram": RegionKind.diagram, "picture": RegionKind.diagram,
            "figure": RegionKind.diagram, "table": RegionKind.table}


@lru_cache(maxsize=1)
def _model():
    weights = ROOT / settings()["layout"]["yolo_weights"]
    if not Path(weights).exists():
        log.warning("YOLO weights not found at %s, using heuristic layout", weights)
        return None
    try:
        from ultralytics import YOLO
    except ImportError:
        log.warning("ultralytics not installed, using heuristic layout")
        return None
    return YOLO(str(weights))


def detect_graphics_yolo(wc: WorkingCopy, words: list[Word]) -> list[Graphic] | None:
    model = _model()
    if model is None:
        return None
    img = cv2.cvtColor(wc.gray, cv2.COLOR_GRAY2BGR)
    res = model.predict(img, conf=settings()["layout"]["yolo_conf"], verbose=False)[0]
    names = res.names
    out = []
    for b, c, s in zip(res.boxes.xyxy.tolist(), res.boxes.cls.tolist(), res.boxes.conf.tolist()):
        kind = _GRAPHIC.get(names[int(c)].lower())
        if kind is None:
            continue
        box = Box(x0=int(b[0]), y0=int(b[1]), x1=int(b[2]) + 1, y1=int(b[3]) + 1)
        labels = [w for w in words if box.intersection(w.box) > 0.5 * w.box.area]
        out.append(Graphic(kind=kind, box=box, labels=labels, score=float(s)))
    return out

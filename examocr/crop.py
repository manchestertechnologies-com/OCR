"""Cut crops from the ORIGINAL page and save them losslessly. No resize, no filter, no JPEG."""
from __future__ import annotations

import hashlib
import os
import stat
from pathlib import Path

import cv2
import numpy as np

from .schemas import Box


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def save_png(img: np.ndarray, path: str | Path, lock: bool = True) -> str:
    """Write a lossless PNG and return its SHA-256. Locked files are made read-only."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, buf = cv2.imencode(".png", img, [cv2.IMWRITE_PNG_COMPRESSION, 6])  # zlib = lossless
    if not ok:
        raise IOError(f"PNG encode failed for {path}")
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite locked original {path}")
    path.write_bytes(buf.tobytes())
    if lock:
        os.chmod(path, stat.S_IREAD | stat.S_IRGRP | stat.S_IROTH)
    return sha256_file(path)


def load_png(path: str | Path) -> np.ndarray:
    data = np.fromfile(str(path), dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise IOError(f"Cannot read {path}")
    return img


def cut(original: np.ndarray, box: Box) -> np.ndarray:
    """Pure array slice - the crop is bit-identical to that area of the original page."""
    b = box.clip(original.shape[1], original.shape[0])
    return original[b.y0:b.y1, b.x0:b.x1].copy()

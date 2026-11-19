import numpy as np
import pytest

from examocr.crop import cut, load_png, save_png, sha256_file
from examocr.preprocess import make_working_copy
from examocr.schemas import Box


def _page(seed=0):
    rng = np.random.default_rng(seed)
    return rng.integers(0, 256, size=(400, 300, 3), dtype=np.uint8)


def test_crop_is_bit_identical_after_png_roundtrip(tmp_path):
    page = _page()
    box = Box(x0=17, y0=33, x1=211, y1=377)
    sha = save_png(cut(page, box), tmp_path / "c.png")
    back = load_png(tmp_path / "c.png")
    assert np.array_equal(back, page[33:377, 17:211])
    assert sha == sha256_file(tmp_path / "c.png")


def test_locked_crop_cannot_be_overwritten(tmp_path):
    save_png(_page(), tmp_path / "c.png")
    with pytest.raises(FileExistsError):
        save_png(_page(1), tmp_path / "c.png")


def test_working_copy_box_maps_back_and_covers_rotated_content():
    page = np.full((1200, 900, 3), 255, np.uint8)
    for r in range(100, 1100, 60):                 # fake text rows
        page[r:r + 18, 100:800] = 0
    import cv2
    m = cv2.getRotationMatrix2D((450, 600), 3.0, 1.0)
    skewed = cv2.warpAffine(page, m, (900, 1200), borderValue=(255, 255, 255))
    wc = make_working_copy(skewed)
    assert abs(abs(wc.skew_deg) - 3.0) < 0.6
    ob = wc.to_original(Box(x0=90, y0=90, x1=810, y1=1110))
    ink = np.argwhere(skewed[..., 0] < 128)
    inside = ((ink[:, 0] >= ob.y0) & (ink[:, 0] < ob.y1) & (ink[:, 1] >= ob.x0) & (ink[:, 1] < ob.x1))
    assert inside.mean() > 0.99

"""The real Mathpix service on the sample paper (no OCR model runs on this computer).
Needs the key (setx MATHPIX_APP_ID / MATHPIX_APP_KEY) and costs 2 images ≈ $0.004.
Run only this:  python -m pytest tests -m mathpix"""
import pytest

from make_sample_paper import make


def _mathpix_key() -> bool:
    try:
        from examocr.ocr.mathpix import credentials
        return credentials() is not None
    except Exception:
        return False


@pytest.mark.mathpix
@pytest.mark.skipif(not _mathpix_key(), reason="no Mathpix key (setx MATHPIX_APP_ID / MATHPIX_APP_KEY)")
def test_mathpix_real(tmp_data, tmp_path):
    import time

    from examocr import pipeline, store
    from examocr.schemas import Box
    pr = pipeline.run(pipeline.create_project(make(tmp_path / "s.png"), "NEET").id)
    t = time.perf_counter()
    q1 = store.add_manual_crop(pr.id, 1, Box(x0=150, y0=440, x1=1500, y1=600), "question")["question"]
    took = time.perf_counter() - t
    assert q1["read_by"] == "Mathpix", q1["review_reasons"]
    assert q1["q_no"] == 1 and len(q1["options"]) == 4
    assert "kinetic energy" in q1["text"]
    print(f"Mathpix question 1: {took:.2f} s")
    q3 = store.add_manual_crop(pr.id, 1, Box(x0=150, y0=1100, x1=1500, y1=1830), "question")["question"]
    assert q3["read_by"] == "Mathpix" and len(q3["diagram_paths"]) == 1 and len(q3["options"]) == 4

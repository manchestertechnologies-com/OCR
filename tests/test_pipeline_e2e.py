"""End-to-end on a synthetic page. Needs Tesseract installed."""
import shutil

import pytest

from make_sample_paper import make

pytestmark = pytest.mark.skipif(
    not shutil.which("tesseract") and not __import__("os").path.exists(
        r"C:\Program Files\Tesseract-OCR\tesseract.exe"), reason="tesseract not installed")


def test_sample_paper(tmp_data, tmp_path):
    from examocr import pipeline, store
    from examocr.schemas import QuestionType

    src = make(tmp_path / "sample.png")
    pr = pipeline.create_project(src, "NEET", "auto")
    pr = pipeline.run(pr.id)
    assert pr.status == "ready", pr.error

    qs = {q.q_no: q for q in pr.questions}
    assert set(qs) == {1, 2, 3, 4, 5}, [(q.q_no, q.text[:40]) for q in pr.questions]
    assert qs[1].options and len(qs[1].options) == 4
    assert qs[2].question_type == QuestionType.assertion_reason
    assert qs[4].question_type == QuestionType.match
    assert qs[3].diagram_path, "diagram should be cropped and linked to Q3"
    assert all(q.subject == "Physics" for q in pr.questions)
    assert not qs[2].needs_review, qs[2].review_reasons

    # every mark of the diagram (dashed line, arrowhead, '?') is inside the diagram crop
    from examocr.crop import load_png
    from examocr.quality import touches_edge
    diag = load_png(store.project_dir(pr.id) / qs[3].diagram_path)
    assert not touches_edge(diag)
    assert "\nI\n" not in qs[3].text      # no dash fragments leaking into question text

    rt = pr.roundtrip
    assert rt["hash_errors"] == []
    assert all(p["pixel_equal"] for p in rt["pages"])
    assert rt["ok"], rt["pages"][0]["lost_ink_regions"][:5]

    # editing never touches the original
    q = store.edit_text(pr.id, qs[1].id, "edited text")
    assert q.text == qs[1].text and q.text_edited == "edited text"
    assert store.verify_hash(pr.id, q.crop_path, q.original_sha256)
    q = store.reset(pr.id, q.id)
    assert q.text_edited is None

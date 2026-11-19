"""Undo must restore every kind of reviewer change, not just drop the history entry."""
import shutil

import pytest

from make_sample_paper import make

pytestmark = pytest.mark.skipif(
    not shutil.which("tesseract") and not __import__("os").path.exists(
        r"C:\Program Files\Tesseract-OCR\tesseract.exe"), reason="tesseract not installed")

PNG = b"\x89PNG\r\n\x1a\n" + b"\x00" * 16


@pytest.fixture
def question(tmp_data, tmp_path):
    from examocr import pipeline, store
    from examocr.schemas import Box
    pr = pipeline.run(pipeline.create_project(make(tmp_path / "sample.png"), "NEET").id)
    q = store.add_manual_crop(pr.id, 1, Box(x0=150, y0=440, x1=1500, y1=600), "question")["question"]
    return pr.id, q


def test_undo_label_change(question):
    from examocr import store
    pid, q = question
    store.set_fields(pid, q["id"], question_type="numerical", q_no=42, subject="Chemistry")
    back = store.undo(pid, q["id"])
    assert back.question_type.value == q["question_type"]
    assert back.q_no == q["q_no"] and back.subject == q["subject"]
    assert back.labels_reviewed is False
    assert back.type_signals == q["type_signals"]


def test_undo_reset_restores_edits(question):
    from examocr import store
    pid, q = question
    store.edit_text(pid, q["id"], "fixed text")
    store.edit_diagram(pid, q["id"], PNG)
    r = store.reset(pid, q["id"])
    assert r.text_edited is None and r.diagram_edited_path is None and not r.edited
    back = store.undo(pid, q["id"])
    assert back.text_edited == "fixed text"
    assert back.diagram_edited_path is not None and back.edited


def test_undo_recrop_updates_superseded_list(question):
    from examocr import store
    from examocr.schemas import Box
    pid, q = question
    moved = store.recrop(pid, q["id"], Box(x0=150, y0=430, x1=1520, y1=610))
    assert q["crop_path"] in moved.superseded_crops
    back = store.undo(pid, q["id"])
    assert back.crop_path == q["crop_path"]
    assert q["crop_path"] not in back.superseded_crops          # it is the live crop again
    assert moved.crop_path in back.superseded_crops

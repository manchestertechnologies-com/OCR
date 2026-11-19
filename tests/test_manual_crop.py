"""The USER draws the boxes; the app must extract text, images and type from each one."""
import shutil

import pytest
from PIL import Image, ImageDraw

from make_sample_paper import font, make

pytestmark = pytest.mark.skipif(
    not shutil.which("tesseract") and not __import__("os").path.exists(
        r"C:\Program Files\Tesseract-OCR\tesseract.exe"), reason="tesseract not installed")


def _image_option_page(path):
    """Q6: question text, then four small graphs as the options (1)-(4)."""
    W, H = 2480, 3508
    img = Image.new("RGB", (W, H), "white")
    d = ImageDraw.Draw(img)
    f = font(40)
    d.text((180, 200), "6. Which graph shows velocity of a body in free fall?", fill="black", font=f)
    for i in range(4):
        x0, y0 = 240 + (i % 2) * 900, 330 + (i // 2) * 520
        d.text((x0 - 60, y0), f"({i + 1})", fill="black", font=f)
        ax, ay = x0 + 40, y0 + 380
        d.line([(ax, ay), (ax + 420, ay)], fill="black", width=4)            # x axis
        d.line([(ax, ay), (ax, ay - 360)], fill="black", width=4)            # y axis
        d.polygon([(ax + 420, ay - 12), (ax + 450, ay), (ax + 420, ay + 12)], fill="black")
        d.polygon([(ax - 12, ay - 360), (ax, ay - 390), (ax + 12, ay - 360)], fill="black")
        curve = [(ax + t * 4, ay - (t * 3 if i == 0 else 150 if i == 1 else t * t // 30 if i == 2 else 300 - t * 3))
                 for t in range(0, 100)]
        d.line(curve, fill="black", width=4)
        d.text((ax + 430, ay + 10), "t", fill="black", font=f)
        d.text((ax - 40, ay - 400), "v", fill="black", font=f)
    img.save(path)
    return path


@pytest.fixture
def project(tmp_data, tmp_path):
    from examocr import pipeline
    pr = pipeline.create_project(make(tmp_path / "sample.png"), "NEET")     # default = manual mode
    pr = pipeline.run(pr.id)
    assert pr.status == "ready" and pr.mode == "manual"
    assert pr.questions == [] and len(pr.pages) == 1                        # nothing auto-cropped
    return pr


def _draw(pid, box, kind="question", **kw):
    from examocr import store
    from examocr.schemas import Box
    return store.add_manual_crop(pid, 1, Box(x0=box[0], y0=box[1], x1=box[2], y1=box[3]), kind, **kw)


def test_user_crop_mcq(project):
    q = _draw(project.id, (150, 440, 1500, 600))["question"]
    assert q["q_no"] == 1                                 # number read from the marker
    assert len(q["options"]) == 4
    assert q["question_type"] == "mcq"
    assert q["diagram_paths"] == []
    assert q["subject"] is None or q["subject"] == "Physics"


def test_user_crop_with_diagram_inside(project):
    from examocr import store
    from examocr.crop import load_png
    from examocr.quality import touches_edge
    q = _draw(project.id, (150, 1100, 1500, 1830), subject="Physics")["question"]
    assert q["q_no"] == 3
    assert len(q["diagram_paths"]) == 1, "diagram inside the user's box must be extracted"
    diag = load_png(store.project_dir(project.id) / q["diagram_paths"][0])
    assert not touches_edge(diag)                         # dashed line, arrow, labels all inside
    assert "F = ?" not in q["text"]                       # diagram labels kept out of the question text
    assert len(q["options"]) == 4
    assert q["question_type"] == "diagram"
    assert any("diagram" in s for s in q["type_signals"])


def test_user_crop_match_and_assertion(project):
    q4 = _draw(project.id, (150, 1855, 1300, 2310))["question"]
    assert q4["question_type"] == "match" and len(q4["statements"]) >= 3
    q2 = _draw(project.id, (150, 620, 1300, 1075))["question"]
    assert q2["question_type"] == "assertion_reason"
    assert q2["type_confidence"] >= 0.8 and not q2["needs_review"], q2["review_reasons"]


def test_question_split_over_two_boxes(project):
    from examocr import store
    q = _draw(project.id, (150, 620, 1300, 820))["question"]         # stem + A/R lines only
    assert q["options"] == []
    q = _draw(project.id, (150, 820, 1300, 1075), kind="part", parent_id=q["id"])["question"]
    assert len(q["parts"]) == 2
    assert len(q["options"]) == 4
    assert q["question_type"] == "assertion_reason"
    q = store.undo(project.id, q["id"]).model_dump()                  # undo removes the part again
    assert len(q["parts"]) == 1


def test_recrop_reanalyzes_and_keeps_old(project):
    from examocr import store
    from examocr.schemas import Box
    q = _draw(project.id, (150, 1100, 1500, 1300))["question"]       # box too small: diagram cut
    q2 = store.recrop(project.id, q["id"], Box(x0=150, y0=1100, x1=1500, y1=1830))
    assert len(q2.diagram_paths) == 1 and len(q2.options) == 4
    assert q["crop_path"] in q2.superseded_crops
    q3 = store.undo(project.id, q["id"])
    assert q3.crop_id == q["crop_id"]


def test_image_options(tmp_data, tmp_path):
    from examocr import pipeline
    pr = pipeline.run(pipeline.create_project(_image_option_page(tmp_path / "img_opts.png"), "NEET").id)
    q = _draw(pr.id, (120, 180, 2200, 1300))["question"]
    assert q["q_no"] == 6
    assert len(q["option_images"]) == 4, (q["option_images"], q["diagram_paths"])
    assert len(q["options"]) == 4


def test_passage_link_and_label_fix(project):
    from examocr import store
    p = _draw(project.id, (150, 120, 2300, 320), kind="passage")["crop"]
    assert p["passage_id"] == "P1" and p["text"]
    q = _draw(project.id, (150, 2335, 1500, 2480), passage_id="P1")["question"]
    assert q["question_type"] == "paragraph" and q["passage_id"] == "P1"
    q = store.set_fields(project.id, q["id"], question_type="mcq", passage_id="")
    assert q.question_type.value == "mcq" and q.labels_reviewed and q.passage_id is None

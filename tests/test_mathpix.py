"""Every box is read by Mathpix - tested with Mathpix-style answers (no network, no cost).
A real run with your key: tests/test_real_models.py::test_mathpix_real (marked `mathpix`)."""
import time

import pytest

from make_sample_paper import make

pytestmark = pytest.mark.mathpix_path       # the real Mathpix code, not the offline stand-in

Q3_BOX = (150, 1100, 1500, 1830)            # sample page: question 3 with a diagram, 4 options


def _fake_q3():
    """What Mathpix returns for the sample's question 3 (crop coordinates)."""
    return {
        "request_id": "test-1",
        "text": "3. In the figure shown below, find the direction of the force on the block.\n\n"
                "![](https://cdn.mathpix.com/cropped/fake.jpg?height=480&width=725)\n\n"
                "(1) Up the incline (2) Down the incline\n(3) Perpendicular (4) Zero",
        "confidence": 0.97,
        "line_data": [
            {"type": "text", "cnt": [[10, 10], [1300, 10], [1300, 60], [10, 60]], "is_printed": True},
            {"type": "diagram", "cnt": [[418, 124], [1143, 124], [1143, 604], [418, 604]], "included": True},
            {"type": "text", "cnt": [[10, 620], [900, 620], [900, 700], [10, 700]], "is_printed": True},
        ],
    }


@pytest.fixture
def fast(monkeypatch, tmp_data, tmp_path):
    """A project and a fake Mathpix that answers with `answer` (a dict or an exception)."""
    from examocr import pipeline
    from examocr.config import settings
    from examocr.ocr import mathpix
    calls = []
    state = {"answer": _fake_q3(), "status": 200}

    def post(url, headers, body, timeout):
        calls.append(body)
        assert url.endswith("/v3/text") and headers["app_id"] == "id" and body["src"].startswith("data:image/")
        assert body["include_line_data"] and body["math_inline_delimiters"] == ["$", "$"]
        if isinstance(state["answer"], Exception):
            raise state["answer"]
        return state["status"], state["answer"]
    monkeypatch.setattr(mathpix, "credentials", lambda: ("id", "key"))
    monkeypatch.setattr(mathpix, "_http_post", post)
    monkeypatch.setitem(settings()["mathpix"], "monthly_limit_usd", 20)
    pr = pipeline.run(pipeline.create_project(make(tmp_path / "s.png"), "NEET").id)
    assert pr.reader == "fast"
    return pr, state, calls


def _draw(pid, box, **kw):
    from examocr import store
    from examocr.schemas import Box
    return store.add_manual_crop(pid, 1, Box(x0=box[0], y0=box[1], x1=box[2], y1=box[3]), "question", **kw)["question"]


def test_fast_reads_text_diagram_and_options(fast):
    pr, state, calls = fast
    t = time.perf_counter()
    q = _draw(pr.id, Q3_BOX)
    took = time.perf_counter() - t
    assert len(calls) == 1                                       # one request per box
    assert q["read_by"] == "Mathpix"
    assert q["q_no"] == 3 and len(q["options"]) == 4
    assert "find the direction of the force on the block" in q["text"]
    assert "cdn.mathpix.com" not in q["text"] + q["display_text"]   # Mathpix's own image links dropped
    assert len(q["diagram_paths"]) == 1                          # cut from the locked original
    lines = q["display_text"].splitlines()
    assert lines.index("[diagram 1]") == 1                       # exactly where Mathpix put it
    assert took < 1.5, took                                      # the app's own time on top of Mathpix


def test_formulas_and_smiles_suggestion(fast):
    pr, state, calls = fast
    state["answer"] = {
        "text": "5) $CH_{3}-C\\equiv CH$ gives $\\frac{M}{3}$\nA) <smiles>C/C=C/CC</smiles> B) <smiles>CC=C(C)C</smiles>",
        "confidence": 0.9,
        "line_data": [
            {"type": "chemistry", "cnt": [[200, 300], [500, 300], [500, 500], [200, 500]], "is_handwritten": True},
            {"type": "chemistry", "cnt": [[800, 300], [1100, 300], [1100, 500], [800, 500]], "is_handwritten": True},
        ],
    }
    q = _draw(pr.id, Q3_BOX)
    assert "CH₃-C≡CH" in q["text"] and "M/3" in q["text"]         # plain copy: no backslashes
    assert "\\" not in q["text"]
    assert "$\\frac{M}{3}$" in q["latex_text"]                    # LaTeX source for MathJax / Word
    assert len(q["option_images"]) == 2                          # "A) [drawing] B) [drawing]"
    from examocr import store
    crops = {c.id: c for c in store.load(pr.id).crops}
    sugg = [crops[f["crop_id"]].smiles_suggested for f in q["figures"]]
    assert sugg == ["C/C=C/CC", "CC=C(C)C"]                      # suggestions only
    assert all(crops[f["crop_id"]].smiles is None for f in q["figures"])
    assert q["source_type"] == "handwritten"


@pytest.mark.parametrize("problem, reason", [
    ("offline", "cannot reach Mathpix"),
    ("refused", "key was refused"),
    ("no_key", "no Mathpix key"),
    ("limit", "limit is reached"),
])
def test_not_read_box_says_why_then_reads_again(fast, monkeypatch, problem, reason):
    """No OCR runs on this computer: a box Mathpix cannot read stays, marked with the reason,
    and is read when Read again sends the same box once the problem is gone."""
    from examocr import store
    from examocr.config import settings
    from examocr.ocr import mathpix
    from examocr.schemas import Box
    pr, state, calls = fast
    good = state["answer"]
    if problem == "offline":
        state["answer"] = mathpix.MathpixError("cannot reach Mathpix (no internet?): ConnectError")
    elif problem == "refused":
        state["status"], state["answer"] = 401, {"error": "Invalid credentials"}
    elif problem == "no_key":
        monkeypatch.setattr(mathpix, "credentials", lambda: None)
    else:
        monkeypatch.setitem(settings()["mathpix"], "monthly_limit_usd", 0.0001)
    q = _draw(pr.id, Q3_BOX)
    assert q["status"] == "error" and reason in q["read_error"], q["read_error"]
    assert q["review_reasons"] == [f"not read: {q['read_error']} - press Read again"]
    assert q["text"] == "" and q["diagram_paths"] == [] and len(q["parts"]) == 1   # nothing guessed
    state["answer"], state["status"] = good, 200                 # internet / key / limit fixed
    monkeypatch.setattr(mathpix, "credentials", lambda: ("id", "key"))
    monkeypatch.setitem(settings()["mathpix"], "monthly_limit_usd", 20)
    again = store.recrop(pr.id, q["id"], Box(**q["parts"][0]["box"]), 0)   # what Read again sends
    assert again.status == "ready" and again.read_by == "Mathpix" and again.q_no == 3
    assert again.read_error is None and len(again.diagram_paths) == 1


def test_usage_counter_and_limit(fast, monkeypatch):
    from examocr import usage
    from examocr.config import settings
    pr, state, calls = fast
    before = usage.status()
    assert before["configured"] and before["images"] == 0
    _draw(pr.id, Q3_BOX)
    _draw(pr.id, Q3_BOX)
    after = usage.status()
    assert after["images"] == 2 and after["usd"] == pytest.approx(0.004)
    monkeypatch.setitem(settings()["mathpix"], "monthly_limit_usd", 0.004)
    assert usage.status()["limit_reached"] and not usage.can_spend()
    q = _draw(pr.id, Q3_BOX)
    assert q["status"] == "error" and "limit" in q["read_error"] and len(calls) == 2   # no request past it


def test_background_fast_box(fast):
    """In the web app the box is saved at once and read in the background."""
    from examocr import store, worker
    pr, state, calls = fast
    q = _draw(pr.id, Q3_BOX, background=True)
    assert q["status"] == "reading"
    assert worker.wait_idle(timeout=30)
    done = next(x for x in store.load(pr.id).questions if x.id == q["id"])
    assert done.status == "ready" and done.read_by == "Mathpix" and done.q_no == 3


def test_drawings_left_out_of_the_text_are_still_figures(fast):
    """Mathpix answers graphs and drawings with "included": false and no image link in the
    text (seen on a real paper: 6 drawings, 0 figures before this was fixed)."""
    from examocr.ocr import mathpix
    sq = lambda x0, y0, x1, y1: [[x0, y0], [x1, y0], [x1, y1], [x0, y1]]
    res = mathpix.parse_response({"text": "8. Which graph? (a) (b)", "line_data": [
        {"type": "text", "cnt": sq(0, 0, 900, 40), "included": True},
        {"type": "diagram", "cnt": sq(50, 60, 400, 300), "included": False},
        {"type": "chart", "subtype": "analytical", "cnt": sq(500, 60, 850, 300), "included": False},
    ]}, 1.0, 900, 320)
    assert [f.kind for f in res.figures] == ["diagram", "chart"]
    pr, state, calls = fast
    ans = _fake_q3()
    ans["text"] = ans["text"].replace("![](https://cdn.mathpix.com/cropped/fake.jpg?height=480&width=725)\n\n", "")
    ans["line_data"][1]["included"] = False
    for ln, t in zip(ans["line_data"], ["3. In the figure shown below, find the direction of the force on the block.",
                                         None, "(1) Up the incline (2) Down the incline (3) Perpendicular (4) Zero"]):
        ln["text"] = t
    state["answer"] = ans
    q = _draw(pr.id, Q3_BOX)
    assert len(q["diagram_paths"]) == 1
    assert q["display_text"].splitlines().index("[diagram 1]") == 1    # where Mathpix read it


def test_question_drawing_and_image_options_in_reading_order(fast):
    """Q8-like: a drawing under the question, then four graphs as options (a)-(d) - each
    placed where Mathpix read it (its line_data order), with no image links in the text."""
    pr, state, calls = fast
    sq = lambda x0, y0, x1, y1: [[x0, y0], [x1, y0], [x1, y1], [x0, y1]]
    lines = [{"type": "text", "text": "8. Work done on a gas is shown below:", "cnt": sq(10, 5, 1300, 40)},
             {"type": "diagram", "included": False, "cnt": sq(400, 60, 900, 250)},
             {"type": "text", "text": "\n\nWhich graph shows the work done?", "cnt": sq(10, 270, 1300, 300)}]
    for n, lab in enumerate("abcd"):
        x = 20 + 330 * n
        lines += [{"type": "text", "text": f"\n({lab})", "cnt": sq(x, 330, x + 40, 360)},
                  {"type": "chart", "included": False, "cnt": sq(x + 50, 330, x + 300, 600)}]
    state["answer"] = {"text": "8. Work done on a gas is shown below:\n\nWhich graph shows the work done?\n(a)\n(b)\n(c)\n(d)",
                       "confidence": 0.99, "line_data": lines}
    q = _draw(pr.id, Q3_BOX)
    assert q["display_text"].splitlines() == [
        "8. Work done on a gas is shown below:", "[diagram 1]", "Which graph shows the work done?",
        "(a) [option image 1]", "(b) [option image 2]", "(c) [option image 3]", "(d) [option image 4]"]
    assert len(q["diagram_paths"]) == 1 and len(q["option_images"]) == 4


def test_recognise_structure_with_mathpix(fast):
    """"Recognise drawing" sends the figure to Mathpix (no MolScribe model on this computer)."""
    from examocr import chem, store
    from examocr.ocr import mathpix
    pr, state, calls = fast
    q = _draw(pr.id, Q3_BOX)
    cid = q["figures"][0]["crop_id"]
    state["answer"] = {"text": "<smiles>OC(=O)c1ccccc1</smiles>", "confidence": 0.3, "line_data": []}
    r = store.suggest_structure(pr.id, cid)
    assert r["smiles"] == chem.canonical_smiles(chem.mol_from(smiles="OC(=O)c1ccccc1"))
    assert "M  END" in r["molfile"] and len(calls) == 2
    assert next(c for c in store.load(pr.id).crops if c.id == cid).smiles is None   # a suggestion only
    state["answer"] = {"text": "no structure here", "confidence": 0.1, "line_data": []}
    with pytest.raises(ValueError, match="structure editor"):
        store.suggest_structure(pr.id, cid)
    state["answer"] = mathpix.MathpixError("cannot reach Mathpix (no internet?): ConnectError")
    with pytest.raises(RuntimeError, match="cannot reach Mathpix"):
        store.suggest_structure(pr.id, cid)


def test_key_from_env_file(tmp_path, monkeypatch):
    """The key can sit in .env (a copy of .env.example) - and git never uploads that file."""
    import shutil
    import subprocess
    from pathlib import Path

    from examocr import config
    from examocr.ocr import mathpix
    for name in ("MATHPIX_APP_ID", "MATHPIX_APP_KEY"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(mathpix, "_registry_user_env", lambda name: None)   # never the real key
    monkeypatch.setattr(config, "CONFIG_DIR", tmp_path)                     # no secrets.yaml
    env = tmp_path / ".env"
    env.write_text('﻿# a comment\n\nexport MATHPIX_APP_ID = "my-id"\nMATHPIX_APP_KEY=abc123  # the key\n',
                   encoding="utf-8")                                        # BOM, as Notepad saves it
    monkeypatch.setattr(mathpix, "ENV_FILE", env)
    assert mathpix.credentials() == ("my-id", "abc123")
    monkeypatch.setattr(mathpix, "ENV_FILE", tmp_path / "missing.env")
    assert mathpix.credentials() is None
    if shutil.which("git"):
        root = Path(mathpix.__file__).resolve().parents[2]
        assert subprocess.run(["git", "check-ignore", "-q", ".env"], cwd=root).returncode == 0


def test_big_crop_is_resized_under_the_limit():
    import numpy as np

    from examocr.ocr import mathpix
    img = np.random.default_rng(0).integers(0, 255, (5000, 3500, 3), dtype=np.uint8)   # noisy = large PNG
    src, scale = mathpix.encode(img)
    assert len(src) < 5_000_000 and scale < 1
    res = mathpix.parse_response({"text": "x", "line_data": [
        {"type": "diagram", "cnt": [[100, 100], [300, 100], [300, 300], [100, 300]]}]}, scale, 3500, 5000)
    assert res.figures[0].box.x1 == pytest.approx(300 / scale, abs=2)   # back in crop pixels


def test_word_upload_needs_libreoffice(tmp_data, tmp_path, monkeypatch):
    from examocr import ingest
    f = tmp_path / "paper.docx"
    f.write_bytes(b"PK\x03\x04 not really a docx")
    monkeypatch.setattr(ingest, "soffice", lambda: None)
    with pytest.raises(ValueError, match="LibreOffice"):
        ingest.load_pages(f)


def test_api_upload_and_status(tmp_data, tmp_path, monkeypatch):
    from fastapi.testclient import TestClient

    from examocr.api.app import app
    from examocr.ocr import mathpix
    monkeypatch.setattr(mathpix, "credentials", lambda: None)
    with TestClient(app) as c:
        st = c.get("/api/mathpix").json()
        assert st["configured"] is False and "limit_inr" in st and "word_upload" in st
        img = make(tmp_path / "u.png").read_bytes()
        r = c.post("/api/projects", data={"exam": "NEET"}, files={"file": ("u.png", img, "image/png")})
        assert r.status_code == 200
        assert c.get(f"/api/projects/{r.json()['id']}").json()["reader"] == "fast"
        bad = c.post("/api/projects", data={"exam": "NEET", "mode": "turbo"}, files={"file": ("u.png", img, "image/png")})
        assert bad.status_code == 400
        doc = c.post("/api/projects", data={"exam": "NEET"}, files={"file": ("p.docx", b"x", "application/octet-stream")})
        assert doc.status_code == 200                              # accepted; converted in the background
        for pid in (r.json()["id"], doc.json()["id"]):             # let the background work finish HERE,
            for _ in range(200):                                   # inside the test's temporary folder
                if c.get(f"/api/projects/{pid}").json()["status"] != "processing":
                    break
                time.sleep(0.05)
        p = c.get(f"/api/projects/{doc.json()['id']}").json()
        assert p["status"] == "error" and "LibreOffice" in (p["error"] or "")   # clear reason, no crash

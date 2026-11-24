"""Background reading: drawing a box answers at once; the text arrives when the reader is done."""
import time

from make_sample_paper import make


def test_box_returns_at_once_and_is_read_in_background(tmp_data, tmp_path):
    from examocr import pipeline, store, worker
    from examocr.schemas import Box
    pr = pipeline.run(pipeline.create_project(make(tmp_path / "s.png"), "NEET").id)
    t = time.perf_counter()
    q = store.add_manual_crop(pr.id, 1, Box(x0=150, y0=440, x1=1500, y1=600), "question",
                              background=True)["question"]
    took = time.perf_counter() - t
    assert q["status"] == "reading" and q["text"] == ""
    assert q["parts"][0]["box"] == {"x0": 150, "y0": 440, "x1": 1500, "y1": 600}   # the box as drawn
    assert (store.project_dir(pr.id) / q["parts"][0]["crop_path"]).exists()      # crop saved now
    assert took < 1.0, took                                    # no OCR in the request
    assert worker.wait_idle(timeout=120)
    done = next(x for x in store.load(pr.id).questions if x.id == q["id"])
    assert done.status == "ready" and done.q_no == 1 and len(done.options) == 4
    assert done.read_seconds is not None


def test_number_on_the_paper_beats_a_guess(tmp_data, tmp_path):
    """A box is assembled before it is read: its number must come from the paper, not be
    guessed from the question before it (Q1 drawn, then Q3 -> must be 3, not 'inferred 2')."""
    from examocr import pipeline, store, worker
    from examocr.schemas import Box
    pr = pipeline.run(pipeline.create_project(make(tmp_path / "s.png"), "NEET").id)
    store.add_manual_crop(pr.id, 1, Box(x0=150, y0=440, x1=1500, y1=600), "question", background=True)
    store.add_manual_crop(pr.id, 1, Box(x0=150, y0=1100, x1=1500, y1=1830), "question", background=True)
    assert worker.wait_idle(timeout=120)
    qs = sorted(store.load(pr.id).questions, key=lambda q: q.box.y0)
    assert [(q.q_no, q.q_no_inferred) for q in qs] == [(1, False), (3, False)]


def test_deleted_while_reading_is_dropped(tmp_data, tmp_path):
    from examocr import pipeline, store, worker
    from examocr.schemas import Box
    pr = pipeline.run(pipeline.create_project(make(tmp_path / "s.png"), "NEET").id)
    q = store.add_manual_crop(pr.id, 1, Box(x0=150, y0=440, x1=1500, y1=600), "question",
                              background=True)["question"]
    store.remove_question(pr.id, q["id"])
    assert worker.wait_idle(timeout=120)
    assert all(x.id != q["id"] for x in store.load(pr.id).questions)   # not brought back

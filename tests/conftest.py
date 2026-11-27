import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))


@pytest.fixture
def tmp_data(tmp_path, monkeypatch):
    """Point project storage at a temp folder."""
    from examocr import store
    (tmp_path / "projects").mkdir()
    (tmp_path / "uploads").mkdir()
    monkeypatch.setattr(store, "data_dir", lambda: tmp_path)
    return tmp_path


def pytest_configure(config):
    config.addinivalue_line("markers", "mathpix_path: the real Mathpix code against a fake Mathpix server")
    config.addinivalue_line("markers", "mathpix: calls the real Mathpix service (needs the key; costs a few images)")


@pytest.fixture(autouse=True)
def offline_reader(request, monkeypatch):
    """The app reads every box with Mathpix (online, paid). Most tests check the rules around
    reading - locked crops, question assembly, undo, background reading - so here Mathpix is
    played by the old offline reader with Tesseract: no network, no cost.
    Tests marked `mathpix_path` / `mathpix` run the real Mathpix code instead."""
    if request.node.get_closest_marker("mathpix_path") or request.node.get_closest_marker("mathpix"):
        yield
        return
    from examocr import analyze
    from examocr.config import settings
    cfg = settings()
    monkeypatch.setitem(cfg["ocr"], "printed_engine", "tesseract")
    monkeypatch.setitem(cfg["ocr"], "handwritten_engine", "tesseract")
    monkeypatch.setitem(cfg["layout"], "figure_model", "")
    monkeypatch.setattr(analyze, "_analyze_mathpix", lambda original, page, box, find_graphics:
                        analyze._analyze_local(original, page, box, find_graphics))
    yield

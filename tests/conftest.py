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
    config.addinivalue_line("markers", "models: runs the real offline OCR / layout models (slow)")


@pytest.fixture(autouse=True)
def fast_engines(request, monkeypatch):
    """Most tests check the rules around OCR, so they use Tesseract and the ink rules (fast).
    Tests marked `models` use the configured models (GLM-OCR + PP-DocLayoutV3)."""
    if request.node.get_closest_marker("models"):
        yield
        return
    from examocr.config import settings
    cfg = settings()
    monkeypatch.setitem(cfg["ocr"], "printed_engine", "tesseract")
    monkeypatch.setitem(cfg["ocr"], "handwritten_engine", "tesseract")
    monkeypatch.setitem(cfg["layout"], "figure_model", "")
    yield

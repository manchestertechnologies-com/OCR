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

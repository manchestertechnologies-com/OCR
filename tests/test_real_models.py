"""The real offline models on user-drawn boxes: GLM-OCR reads, PP-DocLayoutV3 finds figures.
Slow (~1-2 minutes per box on a laptop CPU). Skipped when the models are not installed:
    python -m examocr.cli download-models
Run only these:  python -m pytest tests -m models"""
import pytest

from make_sample_paper import make
from test_manual_crop import _draw, _image_option_page


def _installed() -> bool:
    try:
        import torchvision  # noqa: F401  (GLM-OCR's processor needs it)
        from huggingface_hub import try_to_load_from_cache

        from examocr.config import settings
        cfg = settings()
        return all(isinstance(try_to_load_from_cache(m, "config.json"), str)
                   for m in (cfg["ocr"]["glm_model"], cfg["layout"]["figure_model"]))
    except Exception:
        return False


pytestmark = [pytest.mark.models,
              pytest.mark.skipif(not _installed(), reason="OCR models not downloaded")]


@pytest.fixture
def project(tmp_data, tmp_path):
    from examocr import pipeline
    pr = pipeline.run(pipeline.create_project(make(tmp_path / "sample.png"), "NEET").id)
    assert pr.status == "ready"
    return pr


def test_glm_reads_question_with_diagram_inside(project):
    """The user's box is used as drawn; the drawing is cut out, the options under it are read."""
    q = _draw(project.id, (150, 1100, 1500, 1830))["question"]
    assert q["parts"][0]["box"] == {"x0": 150, "y0": 1100, "x1": 1500, "y1": 1830}   # not moved
    assert q["q_no"] == 3
    assert len(q["diagram_paths"]) == 1
    assert "F = ?" not in q["text"]                       # diagram labels stay in the image
    assert "find the direction of the force on the block" in q["text"]
    assert len(q["options"]) == 4 and "Down the incline" in q["text"]
    assert q["display_text"].index("[diagram 1]") < q["display_text"].index("(1) Up the incline")


def test_glm_mcq_text_is_exact(project):
    q = _draw(project.id, (150, 440, 1500, 600))["question"]
    assert q["q_no"] == 1 and len(q["options"]) == 4 and q["question_type"] == "mcq"
    assert "A body of mass 2 kg moves with velocity 3 m/s" in q["text"]
    assert q["diagram_paths"] == []


def test_layout_model_finds_image_options(tmp_data, tmp_path):
    from examocr import pipeline, store
    from examocr.schemas import Box
    pr = pipeline.run(pipeline.create_project(_image_option_page(tmp_path / "img_opts.png"), "NEET").id)
    r = store.add_manual_crop(pr.id, 1, Box(x0=100, y0=170, x1=2200, y1=1400), "question")
    q = r["question"]
    assert q["q_no"] == 6
    assert len(q["option_images"]) == 4, q["display_text"]


def _molscribe_ok() -> bool:
    try:
        from huggingface_hub import try_to_load_from_cache
        import importlib.util
        if importlib.util.find_spec("molscribe") is None:
            return False
        return isinstance(try_to_load_from_cache("yujieq/MolScribe", "swin_base_char_aux_1m.pth"), str)
    except Exception:
        return False


@pytest.mark.skipif(not _molscribe_ok(), reason="MolScribe not installed")
def test_molscribe_reads_a_printed_structure():
    """Printed drawings are what MolScribe reads reliably (hand-drawn ones often fail - the
    app then asks for the structure to be drawn in Ketcher)."""
    from examocr import chem
    for smiles in ("CC/C=C/C", "OC(=O)c1ccccc1"):
        img = chem.png_bytes_to_array(chem.render_png(chem.mol_from(smiles=smiles), 400))
        got, molfile, conf = chem.recognise(img)
        assert got == chem.canonical_smiles(chem.mol_from(smiles=smiles)), (smiles, got, conf)
        assert conf > 0.5 and "M  END" in molfile

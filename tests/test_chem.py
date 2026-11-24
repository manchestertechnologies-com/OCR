"""Chemical structures: RDKit checks, drawings, 3D; saving a structure for a figure + undo."""
import pytest

from examocr import chem


def test_mol_from_smiles_and_molfile():
    mol = chem.mol_from(smiles="CC/C=C/C")                         # trans-pent-2-ene
    canon = chem.canonical_smiles(mol)
    assert chem.canonical_smiles(chem.mol_from(smiles=canon)) == canon
    again = chem.mol_from(molfile=chem.molfile(mol))              # molfile round trip keeps E/Z
    assert chem.canonical_smiles(again) == canon and "/" in canon
    with pytest.raises(ValueError):
        chem.mol_from(smiles="C1CC")                                 # unclosed ring
    with pytest.raises(ValueError):
        chem.mol_from()


def test_render_png_and_3d():
    png = chem.render_png(chem.mol_from(smiles="c1ccccc1"), 300)
    assert png[:8] == b"\x89PNG\r\n\x1a\n"
    img = chem.png_bytes_to_array(png)
    assert img.shape[1] == 300 and img.mean() > 200               # black lines on white
    block = chem.to_3d_molblock("C=C")
    assert "V2000" in block or "V3000" in block
    zs = [float(ln.split()[2]) for ln in block.splitlines()[4:10] if len(ln.split()) > 3]
    assert len(zs) == 6                                            # 2 C + 4 H with 3D coordinates
    with pytest.raises(ValueError):
        chem.to_3d_molblock("not a molecule")


def test_structure_on_a_figure_and_undo(tmp_data, tmp_path):
    from make_sample_paper import make

    from examocr import pipeline, store
    from examocr.schemas import Box
    pr =pipeline.run(pipeline.create_project(make(tmp_path / "s.png"), "NEET").id)
    q = store.add_manual_crop(pr.id, 1, Box(x0=150, y0=1100, x1=1500, y1=1830), "question")["question"]
    fig = q["figures"][0]
    cid = fig["crop_id"]
    assert fig["path"].startswith("clean/")
    r = store.set_structure(pr.id, cid, None, "CC=CC")
    assert r["crop"]["smiles"] == "CC=CC" and r["crop"]["structure_path"].startswith("structures/")
    assert r["question"]["figures"][0]["path"] == r["crop"]["structure_path"]   # shown + exported
    assert (store.project_dir(pr.id) / r["crop"]["structure_path"]).exists()
    with pytest.raises(ValueError):
        store.set_structure(pr.id, cid, None, "C1CC")
    q2 = store.undo(pr.id, q["id"])
    assert q2.figures[0]["path"].startswith("clean/")              # back to the drawing
    c = next(c for c in store.load(pr.id).crops if c.id == cid)
    assert c.smiles is None and c.structure_path is None
    from examocr.verify import hash_check
    assert hash_check(store.load(pr.id)) == []                     # locked crops untouched

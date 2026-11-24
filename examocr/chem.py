"""Chemistry: molecules from drawings, clean structure images, 3D coordinates. All offline.

  RDKit (BSD)       check a molecule, draw it as a clean image, make 3D coordinates
  MolScribe (MIT)   read a structure drawing -> SMILES + molfile (a suggestion the reviewer
                    accepts or fixes in Ketcher; handwriting is read less reliably than print)

MolScribe is loaded on first use only (1.1 GB of weights; GLM-OCR already takes 3.6 GB of
this 8 GB laptop's RAM). Two compatibility shims are applied here, inside our code, without
editing any installed package:
  * MolScribe imports OpenNMT-py, whose package __init__ imports torchtext (training-only,
    discontinued, not installable with current torch). torchtext gets an empty stand-in;
    MolScribe uses only OpenNMT's plain-torch layers.
  * MolScribe post-processes with multiprocessing.Pool(16). On Windows every pool process
    re-imports everything (16 x the memory, ~40 s each). For one drawing at a time the work
    runs in-process instead.
"""
from __future__ import annotations

import importlib.abc
import importlib.machinery
import io
import logging
import sys
import threading
import types

import numpy as np

log = logging.getLogger(__name__)


# ------------------------------------------------------------------ RDKit
def mol_from(molfile: str | None = None, smiles: str | None = None):
    """A checked RDKit molecule (2D coordinates kept from a molfile), or ValueError."""
    from rdkit import Chem
    from rdkit.Chem import AllChem
    mol = None
    if molfile and molfile.strip():
        mol = Chem.MolFromMolBlock(molfile, sanitize=True, removeHs=False)
    if mol is None and smiles and smiles.strip():
        mol = Chem.MolFromSmiles(smiles.strip())
        if mol is not None:
            AllChem.Compute2DCoords(mol)
    if mol is None or mol.GetNumAtoms() == 0:
        raise ValueError("not a valid structure")
    if mol.GetNumConformers() == 0:
        AllChem.Compute2DCoords(mol)
    return mol


def canonical_smiles(mol) -> str:
    from rdkit import Chem
    return Chem.MolToSmiles(mol)


def molfile(mol) -> str:
    from rdkit import Chem
    return Chem.MolToMolBlock(mol)


def render_png(mol, width: int = 360) -> bytes:
    """A clean structure drawing (black on white) for display and export."""
    from rdkit.Chem.Draw import rdMolDraw2D
    d = rdMolDraw2D.MolDraw2DCairo(width, int(width * 0.75))
    opts = d.drawOptions()
    opts.useBWAtomPalette()
    opts.clearBackground = True
    opts.padding = 0.08
    rdMolDraw2D.PrepareAndDrawMolecule(d, mol)
    d.FinishDrawing()
    return d.GetDrawingText()


def to_3d_molblock(smiles: str) -> str:
    """3D coordinates for the 3D viewer: hydrogens added, embedded (ETKDG v3), MMFF-relaxed."""
    from rdkit import Chem
    from rdkit.Chem import AllChem
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError("not a valid SMILES")
    mol = Chem.AddHs(mol)
    params = AllChem.ETKDGv3()
    params.randomSeed = 7                       # the same molecule always looks the same
    if AllChem.EmbedMolecule(mol, params) != 0:
        raise ValueError("could not build 3D coordinates for this molecule")
    try:
        AllChem.MMFFOptimizeMolecule(mol, maxIters=500)
    except Exception:
        pass                                    # embedded geometry is still fine to view
    return Chem.MolToMolBlock(mol)


# ------------------------------------------------------------------ MolScribe
class _Stub:
    def __init__(self, *a, **k):
        pass


def _stub_module(name: str) -> types.ModuleType:
    m = types.ModuleType(name)
    m.__path__ = []

    def ga(attr):
        if attr.startswith("__"):
            raise AttributeError(attr)
        if attr[:1].isupper():
            return _Stub                        # Field, Vocab, Iterator ... (never used here)
        sub = f"{name}.{attr}"
        return sys.modules.setdefault(sub, _stub_module(sub))
    m.__getattr__ = ga
    return m


class _TorchtextFinder(importlib.abc.MetaPathFinder, importlib.abc.Loader):
    def find_spec(self, fullname, path, target=None):
        if fullname == "torchtext" or fullname.startswith("torchtext."):
            return importlib.machinery.ModuleSpec(fullname, self, is_package=True)
        return None

    def create_module(self, spec):
        return _stub_module(spec.name)

    def exec_module(self, module):
        pass


class _InProcessPool:
    """multiprocessing.Pool replacement: same map / starmap, run here."""

    def __init__(self, *a, **k):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def map(self, fn, it, chunksize=None):
        return [fn(x) for x in it]

    def starmap(self, fn, it, chunksize=None):
        return [fn(*x) for x in it]


_MS = None
_MS_FAILED: str | None = None
_MS_LOCK = threading.Lock()


def _load_molscribe():
    global _MS, _MS_FAILED
    with _MS_LOCK:
        if _MS is not None or _MS_FAILED:
            return _MS
        try:
            import warnings
            warnings.filterwarnings("ignore", module="molscribe")
            try:
                import torchtext  # noqa: F401  (a real install is fine too)
            except Exception:
                sys.meta_path.insert(0, _TorchtextFinder())
            import torch
            from huggingface_hub import hf_hub_download
            from molscribe import MolScribe
            from molscribe import chemistry as ms_chem

            from .config import settings
            ms_chem.multiprocessing = types.SimpleNamespace(Pool=_InProcessPool)
            cfg = settings().get("chem", {})
            if not cfg.get("molscribe_repo"):
                raise RuntimeError("structure recognition is switched off (chem.molscribe_repo)")
            path = hf_hub_download(cfg["molscribe_repo"], cfg["molscribe_file"])
            log.info("loading structure recogniser %s", cfg["molscribe_repo"])
            _MS = MolScribe(path, device=torch.device("cpu"))
        except Exception as e:                  # not installed / no weights: say why, once
            _MS_FAILED = f"{type(e).__name__}: {e}"
            log.warning("structure recognition unavailable: %s", _MS_FAILED)
        return _MS


def recognition_status() -> str | None:
    """None when recognition works (or has not been tried yet), else the reason it cannot."""
    return _MS_FAILED


def recognise(img: np.ndarray) -> tuple[str, str, float]:
    """A structure drawing (BGR or grey) -> (SMILES, molfile, confidence 0-1).
    Raises RuntimeError when MolScribe is unavailable, ValueError when nothing valid is read."""
    import cv2
    ms = _load_molscribe()
    if ms is None:
        raise RuntimeError(f"structure recognition is not available ({_MS_FAILED})")
    rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB) if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    with _MS_LOCK:
        r = ms.predict_image(rgb, return_confidence=True)
    smiles = (r.get("smiles") or "").strip()
    if not smiles or smiles == "<invalid>":
        raise ValueError("no structure could be read from this drawing")
    mol = mol_from(molfile=r.get("molfile"), smiles=smiles)
    return canonical_smiles(mol), molfile(mol), float(r.get("confidence") or 0.0)


def png_bytes_to_array(data: bytes) -> np.ndarray:
    from PIL import Image
    with Image.open(io.BytesIO(data)) as im:
        return np.array(im.convert("RGB"))[:, :, ::-1].copy()

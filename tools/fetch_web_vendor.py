"""Download the browser libraries once into web/vendor/ (the app then works offline).

    python tools/fetch_web_vendor.py          (also run by: python -m examocr.cli download-models)

  MathJax 3.2.2 (Apache-2.0)      maths + mhchem chemistry rendering   web/vendor/mathjax/
  smiles-drawer 2.4.2 (MIT)       draws a structure from SMILES        web/vendor/smiles-drawer.min.js
  3Dmol.js 3.0.0 (BSD-3-Clause)   3D molecule viewer                   web/vendor/3Dmol-min.js
  Ketcher 3.21.0 (Apache-2.0)     2D structure editor                  web/vendor/ketcher/

Versions are pinned: an update is a deliberate change here, never a surprise.
"""
from __future__ import annotations

import io
import shutil
import sys
import tarfile
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
VENDOR = ROOT / "web" / "vendor"

NPM = "https://registry.npmjs.org/{name}/-/{base}-{ver}.tgz"
KETCHER = "https://github.com/epam/ketcher/releases/download/v{ver}/ketcher-standalone-{ver}.zip"

PACKAGES = [
    # name, version, {path in package: path under web/vendor}
    ("mathjax", "3.2.2", {"es5/": "mathjax/"}),
    ("smiles-drawer", "2.4.2", {"dist/smiles-drawer.min.js": "smiles-drawer.min.js"}),
    ("3dmol", "3.0.0", {"build/3Dmol-min.js": "3Dmol-min.js"}),
]
KETCHER_VERSION = "3.21.0"


def _get(url: str) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "examocr-vendor-fetch"})
    with urllib.request.urlopen(req, timeout=300) as r:
        return r.read()


def _npm(name: str, ver: str, files: dict[str, str]) -> None:
    base = name.split("/")[-1]
    data = _get(NPM.format(name=name, base=base, ver=ver))
    with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as tar:
        for member in tar.getmembers():
            if not member.isfile():
                continue
            rel = member.name.split("/", 1)[1] if "/" in member.name else member.name   # strip "package/"
            for src, dst in files.items():
                if rel == src or (src.endswith("/") and rel.startswith(src)):
                    out = VENDOR / (dst + rel[len(src):] if src.endswith("/") else dst)
                    if ".." in out.relative_to(VENDOR).parts:
                        continue
                    out.parent.mkdir(parents=True, exist_ok=True)
                    out.write_bytes(tar.extractfile(member).read())


def _ketcher(ver: str) -> None:
    data = _get(KETCHER.format(ver=ver))
    target = VENDOR / "ketcher"
    shutil.rmtree(target, ignore_errors=True)
    with zipfile.ZipFile(io.BytesIO(data)) as z:
        names = [n for n in z.namelist() if not n.endswith("/")]
        # the release zip has one top folder holding index.html: strip it
        top = min((n for n in names if n.endswith("index.html")), key=len)
        prefix = top[: -len("index.html")]
        for n in names:
            if not n.startswith(prefix):
                continue
            out = target / n[len(prefix):]
            if ".." in Path(n[len(prefix):]).parts:
                continue
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_bytes(z.read(n))


def installed() -> dict[str, bool]:
    return {
        "mathjax": (VENDOR / "mathjax" / "tex-chtml-full.js").exists(),
        "smiles-drawer": (VENDOR / "smiles-drawer.min.js").exists(),
        "3dmol": (VENDOR / "3Dmol-min.js").exists(),
        "ketcher": (VENDOR / "ketcher" / "index.html").exists(),
    }


def main(force: bool = False) -> int:
    VENDOR.mkdir(parents=True, exist_ok=True)
    have = installed()
    for name, ver, files in PACKAGES:
        if have[name] and not force:
            print(f"{name} {ver}: already there")
            continue
        print(f"{name} {ver}: downloading")
        _npm(name, ver, files)
    if have["ketcher"] and not force:
        print(f"ketcher {KETCHER_VERSION}: already there")
    else:
        print(f"ketcher {KETCHER_VERSION}: downloading")
        _ketcher(KETCHER_VERSION)
    missing = [k for k, v in installed().items() if not v]
    if missing:
        print("missing after download:", ", ".join(missing), file=sys.stderr)
        return 1
    print("web libraries ready in", VENDOR)
    return 0


if __name__ == "__main__":
    sys.exit(main(force="--force" in sys.argv))

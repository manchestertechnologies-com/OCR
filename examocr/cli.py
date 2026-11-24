"""Command line:
  python -m examocr.cli process paper.pdf --exam NEET [--script handwritten]
  python -m examocr.cli download-models          (offline OCR, layout, structure models + web libraries, ~4 GB)
  python -m examocr.cli verify <project_id>
  python -m examocr.cli serve
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from . import pipeline, store
from .verify import hash_check, roundtrip


def main(argv=None):
    ap = argparse.ArgumentParser(prog="examocr")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("process")
    p.add_argument("file", type=Path)
    p.add_argument("--exam", required=True, choices=["JEE", "NEET", "CET", "jee", "neet", "cet"])
    p.add_argument("--mode", default="auto", choices=["auto", "manual"],
                   help="auto: model crops everything; manual: only load pages for cropping in the web app")
    p.add_argument("--script", default="auto", choices=["auto", "printed", "handwritten"],
                   help="paper is printed or handwritten (auto: detect per crop)")
    sub.add_parser("download-models", help="fetch the offline OCR and figure-finding models once")
    v = sub.add_parser("verify")
    v.add_argument("project")
    s = sub.add_parser("serve")
    s.add_argument("--host", default="127.0.0.1")
    s.add_argument("--port", type=int, default=8000)
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    if a.cmd == "process":
        pr = pipeline.create_project(a.file, a.exam, a.mode, a.script)
        pr = pipeline.run(pr.id, lambda m, f: print(f"[{f:5.0%}] {m}", file=sys.stderr))
        print(json.dumps({"project": pr.id, "status": pr.status, "error": pr.error,
                          "dir": str(store.project_dir(pr.id)),
                          "questions": len(pr.questions),
                          "needs_review": sum(q.needs_review for q in pr.questions),
                          "roundtrip_ok": (pr.roundtrip or {}).get("ok")}, indent=2))
        return 0 if pr.status == "ready" else 1
    if a.cmd == "download-models":
        from huggingface_hub import snapshot_download

        from .config import settings
        cfg = settings()
        ocr = cfg["ocr"]
        names = [cfg["layout"].get("figure_model")]
        if "glm" in (ocr["printed_engine"], ocr["handwritten_engine"]):
            names.append(ocr.get("glm_model", "zai-org/GLM-OCR"))
        if ocr["handwritten_engine"] == "trocr":
            names += [ocr.get("trocr_model"), ocr.get("trocr_math_model")]
        for m in filter(None, names):
            print("downloading", m, "->", snapshot_download(m, allow_patterns=["*.json", "*.txt", "*.jinja", "*.safetensors"]))
        chem = cfg.get("chem", {})
        if chem.get("molscribe_repo"):                     # structure drawing -> molecule (1.1 GB)
            from huggingface_hub import hf_hub_download
            print("downloading", chem["molscribe_repo"], "->",
                  hf_hub_download(chem["molscribe_repo"], chem["molscribe_file"]))
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))
        from fetch_web_vendor import main as fetch_vendor  # MathJax, SmilesDrawer, 3Dmol, Ketcher
        return fetch_vendor()
    if a.cmd == "verify":
        pr = store.load(a.project)
        print(json.dumps({"hash_errors": hash_check(pr), **roundtrip(pr)}, indent=2))
        return 0
    if a.cmd == "serve":
        import uvicorn
        print(f"Open http://{a.host}:{a.port} in your browser. Press Ctrl+C to stop.", file=sys.stderr)
        try:
            # Ctrl+C: a crop still being read is given 3 s, then the server stops (the crops
            # already saved are safe - each save is written in one step)
            uvicorn.run("examocr.api.app:app", host=a.host, port=a.port, timeout_graceful_shutdown=3)
        except KeyboardInterrupt:
            pass
        print("Server stopped.", file=sys.stderr)
        return 0


if __name__ == "__main__":
    sys.exit(main())

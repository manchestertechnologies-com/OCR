"""FastAPI server: upload a paper, watch it process, review/edit crops, export.

Run:  uvicorn examocr.api.app:app --reload     then open http://127.0.0.1:8000
"""
from __future__ import annotations

import io
import json
import tempfile
import threading
import zipfile
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from .. import pipeline, store
from ..config import ROOT, available_exams
from ..export import build_docx, build_html
from ..ingest import IMAGE_EXT
from ..schemas import Box
from ..verify import hash_check, roundtrip

@asynccontextmanager
async def lifespan(_app):
    store.mark_interrupted()
    from ..ocr.router import preload_models
    preload_models()
    yield


app = FastAPI(title="Exam paper OCR", lifespan=lifespan)


class QuietShutdown:
    """Ctrl+C cancels the requests still running (a crop being read takes a minute). That is
    expected, not an error: end them quietly instead of printing a traceback for each one."""

    def __init__(self, inner):
        self.inner = inner

    async def __call__(self, scope, receive, send):
        import asyncio
        started = False

        async def track(msg):
            nonlocal started
            started = started or msg["type"] == "http.response.start"
            await send(msg)
        try:
            await self.inner(scope, receive, track)
        except (asyncio.CancelledError, KeyboardInterrupt):
            if scope["type"] != "http":
                return
            import logging
            logging.getLogger("uvicorn.error").info("stopped while %s %s was running",
                                                    scope.get("method"), scope.get("path"))
            if not started:
                try:
                    await send({"type": "http.response.start", "status": 503,
                                "headers": [(b"content-type", b"application/json")]})
                    await send({"type": "http.response.body",
                                "body": b'{"detail": "The server was stopped. Start it again and redraw this box."}'})
                except Exception:
                    pass


app.add_middleware(QuietShutdown)
WEB = ROOT / "web"
app.mount("/static", StaticFiles(directory=WEB), name="static")

PROGRESS: dict[str, dict] = {}
MAX_UPLOAD = 200 * 1024 * 1024


def _code_fingerprint() -> str:
    """Python code is loaded once when the server starts, but web/ files are re-read on every
    request - so after an update the page can look new while crops still use old code."""
    import hashlib
    h = hashlib.sha256()
    for f in sorted((ROOT / "examocr").rglob("*.py")):
        h.update(f"{f.relative_to(ROOT)}:{f.stat().st_mtime_ns}".encode())
    return h.hexdigest()


STARTED_WITH = _code_fingerprint()


@app.get("/", response_class=HTMLResponse)
def index():
    return (WEB / "index.html").read_text(encoding="utf-8")


@app.get("/favicon.ico", include_in_schema=False)
def favicon():
    return FileResponse(WEB / "favicon.svg", media_type="image/svg+xml")


@app.get("/api/version")
def version():
    """stale = the code on disk changed since this server started: restart it."""
    return {"stale": _code_fingerprint() != STARTED_WITH}


@app.get("/api/exams")
def exams():
    return available_exams()


@app.get("/api/exams/{exam}")
def exam_info(exam: str):
    from ..config import exam_profile
    try:
        return exam_profile(exam)
    except ValueError:
        raise HTTPException(404, "unknown exam")


@app.post("/api/projects")
async def upload(file: UploadFile = File(...), exam: str = Form(...), mode: str = Form("manual"),
                 script: str = Form("auto")):
    name = Path(file.filename or "upload").name
    ext = Path(name).suffix.lower()
    if ext != ".pdf" and ext not in IMAGE_EXT:
        raise HTTPException(400, f"Unsupported file type {ext}. Upload a PDF or image.")
    if exam.upper() not in available_exams():
        raise HTTPException(400, f"Unknown exam {exam}")
    data = await file.read()
    if len(data) > MAX_UPLOAD:
        raise HTTPException(413, "File too large")
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td) / name
        tmp.write_bytes(data)
        try:
            pr = pipeline.create_project(tmp, exam, mode, script)
        except ValueError as e:          # e.g. an unknown mode: a bad request, not a server crash
            raise HTTPException(400, str(e)) from None
    PROGRESS[pr.id] = {"message": "queued", "fraction": 0.0}

    def work():
        pipeline.run(pr.id, lambda m, f: PROGRESS.__setitem__(pr.id, {"message": m, "fraction": f}))

    threading.Thread(target=work, daemon=True).start()
    return {"id": pr.id}


@app.get("/api/projects")
def projects():
    return store.list_projects()


def _get(pid: str):
    try:
        return store.load(pid)
    except (FileNotFoundError, ValueError):
        raise HTTPException(404, "project not found")


@app.get("/api/projects/{pid}")
def project(pid: str):
    _get(pid)
    try:
        store.ensure_clean(pid)          # older projects: make the clean diagram copies once
    except Exception:
        pass
    pr = _get(pid)
    out = json.loads(pr.model_dump_json())
    out["progress"] = PROGRESS.get(pid, {"message": pr.status, "fraction": 1.0})
    return out


@app.get("/api/projects/{pid}/file/{rel:path}")
def project_file(pid: str, rel: str):
    base = store.project_dir(pid).resolve()
    path = (base / rel).resolve()
    if base not in path.parents or not path.is_file():
        raise HTTPException(404)
    return FileResponse(path)


class TextEdit(BaseModel):
    text: str | None


class BoxIn(BaseModel):
    x0: int
    y0: int
    x1: int
    y1: int


class Fields(BaseModel):
    question_type: str | None = None
    q_no: int | None = None
    subject: str | None = None
    passage_id: str | None = None


class ManualCrop(BaseModel):
    page: int
    box: BoxIn
    kind: str = "question"
    parent_id: str | None = None
    q_no: int | None = None
    subject: str | None = None
    passage_id: str | None = None
    snap: bool = False       # the user's box is final; True fits it to the content (old behaviour)


def _wrap(fn, *a):
    try:
        return fn(*a)
    except KeyError:
        raise HTTPException(404, "question not found")
    except (ValueError, StopIteration) as e:
        raise HTTPException(400, str(e))


@app.post("/api/projects/{pid}/questions/{qid}/text")
def edit_text(pid: str, qid: str, body: TextEdit):
    return _wrap(store.edit_text, pid, qid, body.text)


@app.post("/api/projects/{pid}/questions/{qid}/fields")
def set_fields(pid: str, qid: str, body: Fields):
    return _wrap(store.set_fields, pid, qid, body.question_type, body.q_no, body.subject, body.passage_id)


@app.get("/api/question-types")
def question_types():
    from ..schemas import QuestionType
    return [t.value for t in QuestionType]


@app.post("/api/projects/{pid}/questions/{qid}/diagram")
async def edit_diagram(pid: str, qid: str, file: UploadFile = File(...)):
    return _wrap(store.edit_diagram, pid, qid, await file.read())


@app.post("/api/projects/{pid}/questions/{qid}/undo")
def undo(pid: str, qid: str):
    return _wrap(store.undo, pid, qid)


@app.post("/api/projects/{pid}/questions/{qid}/reset")
def reset(pid: str, qid: str):
    return _wrap(store.reset, pid, qid)


@app.post("/api/projects/{pid}/questions/{qid}/recrop")
def recrop(pid: str, qid: str, body: BoxIn, part: int = 0, snap: bool = False):
    return _wrap(store.recrop, pid, qid, Box(**body.model_dump()), part, snap)


@app.delete("/api/projects/{pid}/questions/{qid}/parts/{index}")
def remove_part(pid: str, qid: str, index: int):
    return _wrap(store.remove_part, pid, qid, index)


@app.delete("/api/projects/{pid}/questions/{qid}")
def delete_question(pid: str, qid: str):
    _wrap(store.remove_question, pid, qid)
    return {"ok": True}


@app.post("/api/projects/{pid}/crops")
def manual_crop(pid: str, body: ManualCrop):
    return _wrap(store.add_manual_crop, pid, body.page, Box(**body.box.model_dump()), body.kind,
                 body.parent_id, body.q_no, body.subject, body.passage_id, body.snap)


@app.post("/api/projects/{pid}/verify")
def verify(pid: str):
    pr = _get(pid)
    pr.roundtrip = {"hash_errors": hash_check(pr), **roundtrip(pr)}
    store.save(pr)
    return pr.roundtrip


@app.get("/api/projects/{pid}/export")
def export(pid: str):
    pr = _get(pid)
    errors = hash_check(pr)
    if errors:
        raise HTTPException(409, {"error": "original crops changed - export blocked", "details": errors})
    base = store.project_dir(pid)
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_STORED) as z:   # PNGs are stored, not recompressed
        z.writestr("questions.json", json.dumps([json.loads(q.model_dump_json()) for q in pr.questions],
                                                indent=2, ensure_ascii=False))
        z.writestr("crops.json", json.dumps([json.loads(c.model_dump_json()) for c in pr.crops],
                                            indent=2, ensure_ascii=False))
        z.writestr("roundtrip.json", json.dumps(pr.roundtrip or {}, indent=2))
        # the finished questions: text with every diagram at its place
        z.writestr("questions.html", build_html(pr))
        docx = build_docx(pr, base)
        if docx:
            z.writestr("questions.docx", docx)
        for folder in ("crops", "clean", "edits"):
            for f in (base / folder).glob("*.png") if (base / folder).exists() else []:
                z.write(f, f"{folder}/{f.name}")
    buf.seek(0)
    return StreamingResponse(buf, media_type="application/zip",
                             headers={"Content-Disposition": f'attachment; filename="{pid}.zip"'})

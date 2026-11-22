"""Project storage with two layers per question: locked original + editable copy.

data/projects/<id>/
  project.json
  source/<uploaded file>
  pages/page_001.png          locked original pages
  crops/...png                locked original crops
  edits/...png                edited diagram copies (never overwrite originals)
"""
from __future__ import annotations

import json
import os
import re
import threading
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from .config import data_dir
from .crop import cut, load_png, save_png, sha256_file
from .quality import crop_quality
from .quality import touches_edge
from .snap import snap_box
from .schemas import (Box, Crop, EditEntry, PageInfo, Project, QuestionPart, QuestionRecord,
                      QuestionType, RegionKind)

_locks: dict[str, threading.Lock] = {}


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def new_id() -> str:
    return uuid.uuid4().hex[:12]


def project_dir(pid: str) -> Path:
    if not pid.isalnum():
        raise ValueError("bad project id")
    return data_dir() / "projects" / pid


def _lock(pid: str) -> threading.Lock:
    return _locks.setdefault(pid, threading.Lock())


def save(project: Project) -> None:
    d = project_dir(project.id)
    d.mkdir(parents=True, exist_ok=True)
    tmp = d / f"project.json.{uuid.uuid4().hex[:8]}.tmp"   # unique: two savers never share a temp file
    tmp.write_text(project.model_dump_json(indent=2), encoding="utf-8")
    # On Windows the swap fails while anything has project.json open (the web page polling
    # progress, antivirus scanning the new file). That lasts milliseconds, so retry briefly.
    for attempt in range(40):
        try:
            os.replace(tmp, d / "project.json")
            return
        except PermissionError:
            if attempt == 39:
                tmp.unlink(missing_ok=True)
                raise
            time.sleep(0.05)


def load(pid: str) -> Project:
    return Project.model_validate_json((project_dir(pid) / "project.json").read_text(encoding="utf-8"))


def list_projects() -> list[dict]:
    out = []
    for p in sorted((data_dir() / "projects").glob("*/project.json"), key=os.path.getmtime, reverse=True):
        try:
            pr = Project.model_validate_json(p.read_text(encoding="utf-8"))
            out.append({"id": pr.id, "exam": pr.exam, "source_name": pr.source_name,
                        "created": pr.created, "status": pr.status,
                        "questions": len(pr.questions),
                        "needs_review": sum(q.needs_review for q in pr.questions)})
        except Exception:
            continue
    return out


# Everything set_fields may change, so undo can put it back exactly.
_LABEL_FIELDS = ("question_type", "multi_answer", "type_signals", "type_confidence", "labels_reviewed",
                 "q_no", "q_no_inferred", "subject", "passage_id")


def _base_text(q: QuestionRecord) -> str:
    """What the editable copy starts from: the page layout with figure markers (older
    records without one fall back to the plain OCR text)."""
    return q.display_text or q.text


def _json_dict(s: str | None) -> dict | None:
    try:
        v = json.loads(s) if s else None
    except ValueError:
        return None
    return v if isinstance(v, dict) else None


def mark_interrupted() -> list[str]:
    """Call once at server start. A paper still 'processing' then lost its worker thread when
    the server stopped, so mark it failed instead of leaving the page polling forever."""
    out = []
    for f in (data_dir() / "projects").glob("*/project.json"):
        try:
            pr = Project.model_validate_json(f.read_text(encoding="utf-8"))
        except Exception:
            continue
        if pr.status == "processing":
            pr.status = "error"
            pr.error = "Processing was interrupted because the server stopped. Upload the paper again."
            save(pr)
            out.append(pr.id)
    return out


def _question(project: Project, qid: str) -> QuestionRecord:
    for q in project.questions:
        if q.id == qid:
            return q
    raise KeyError(qid)


# ---------------------------------------------------------------- edits (copy layer only)
def edit_text(pid: str, qid: str, new_text: str | None) -> QuestionRecord:
    with _lock(pid):
        pr = load(pid)
        q = _question(pr, qid)
        base = _base_text(q)
        before = q.text_edited if q.text_edited is not None else base
        q.text_edited = None if new_text is None or new_text == base else new_text
        q.edit_history.append(EditEntry(time=now(), field="text",
                                        description="text edited" if q.text_edited else "text reset",
                                        before=before, after=q.text_edited or base))
        q.edited = q.text_edited is not None or q.diagram_edited_path is not None
        save(pr)
        return q


def edit_diagram(pid: str, qid: str, png_bytes: bytes) -> QuestionRecord:
    """Store an edited COPY of the diagram. The original diagram file is never touched."""
    with _lock(pid):
        pr = load(pid)
        q = _question(pr, qid)
        if not png_bytes.startswith(b"\x89PNG"):
            raise ValueError("edited diagram must be PNG")
        n = sum(1 for e in q.edit_history if e.field == "diagram") + 1
        rel = f"edits/{q.id}_diagram_v{n}.png"
        path = project_dir(pid) / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(png_bytes)
        q.edit_history.append(EditEntry(time=now(), field="diagram", description=f"diagram edited (v{n})",
                                        before=q.diagram_edited_path, after=rel))
        q.diagram_edited_path = rel
        q.edited = True
        save(pr)
        return q


# ---------------------------------------------------------------- crop building blocks
def _unique(pr: Project, base: str) -> str:
    used = {c.id for c in pr.crops}
    if base not in used:
        return base
    k = 2
    while f"{base}_v{k}" in used:
        k += 1
    return f"{base}_v{k}"


def _page(pr: Project, page_no: int) -> PageInfo:
    for p in pr.pages:
        if p.page == page_no:
            return p
    raise ValueError(f"page {page_no} not found")


def _save_crop(pr: Project, original: np.ndarray, kind: RegionKind, page_no: int, box: Box,
               cid: str, parent: str | None = None, source_part: str | None = None) -> tuple[Crop, np.ndarray]:
    cid = _unique(pr, cid)
    img = cut(original, box)
    rel = f"crops/p{page_no:03d}_{cid}.png"
    sha = save_png(img, project_dir(pr.id) / rel)
    blur, flag = crop_quality(img)
    c = Crop(id=cid, kind=kind, page=page_no, box=box, path=rel, sha256=sha, blur_score=blur,
             quality_flag=flag, parent_id=parent, source_part=source_part)
    if kind in FIGURE_KINDS:
        c.clean_path = _save_clean(pr.id, rel, img)
    pr.crops.append(c)
    return c, img


FIGURE_KINDS = (RegionKind.diagram, RegionKind.table, RegionKind.option_image)


def _save_clean(pid: str, rel: str, img: np.ndarray) -> str | None:
    """The drawing alone (clean.py) next to the locked crop: clean/<same name>.png."""
    import cv2

    from .clean import clean_drawing
    out = "clean/" + rel.split("/", 1)[-1]
    path = project_dir(pid) / out
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(str(path), clean_drawing(img)):
            return None
    except Exception:
        return None                    # display falls back to the locked crop
    return out


def ensure_clean(pid: str) -> bool:
    """Projects made before clean copies existed: make them once (on open)."""
    from .analyze import refresh_figures
    pr = load(pid)
    if not any(c.kind in FIGURE_KINDS and not c.clean_path for c in pr.crops):
        return False
    with _lock(pid):
        pr = load(pid)
        changed = False
        for c in pr.crops:
            if c.kind in FIGURE_KINDS and not c.clean_path:
                f = project_dir(pid) / c.path
                if f.exists():
                    c.clean_path = _save_clean(pid, c.path, load_png(f))
                    changed |= c.clean_path is not None
        if changed:
            for q in pr.questions:
                refresh_figures(pr, q)
            save(pr)
        return changed


def attach_part(pr: Project, original: np.ndarray, q: QuestionRecord, page_no: int, box: Box,
                index: int | None = None, snap: bool = False) -> QuestionPart:
    """Analyze a box as (part of) question q: save the locked crop, extract every diagram /
    table / image option inside it as its own locked crop, and store the verbatim text.
    snap=True first fits a rough user box to the real content (examocr/snap.py)."""
    from .analyze import analyze_part
    user_box, notes = box, []
    if snap:
        box, notes = snap_box(original, box, include_marker=True)
    pa = analyze_part(original, page_no, box, script=pr.script)
    k = len([c for c in pr.crops if c.parent_id == q.id and c.kind == RegionKind.question]) + 1
    c, _ = _save_crop(pr, original, RegionKind.question, page_no, pa.box,
                      q.id if k == 1 else f"{q.id}_part{k}", parent=q.id)
    if snap:
        c.user_box, c.snap_notes = user_box, notes
    fig_ids: dict[int, str] = {}
    for j, g in enumerate(pa.graphics, start=1):
        gc, gimg = _save_crop(pr, original, g.kind, page_no, g.box, f"{c.id}_{g.kind.value}{j}",
                              parent=q.id, source_part=c.id)
        gc.text = g.labels
        # edge_cut checks the drawing's own ink against the question box edge. (A plain
        # "ink on the border" test on the padded crop fires on neighbouring text.)
        gc.needs_review = g.edge_cut
        fig_ids[j] = gc.id
    layout = re.sub(r"\{\{fig:(\d+)\}\}", lambda m: "{{crop:%s}}" % fig_ids[int(m.group(1))], pa.layout)
    part = QuestionPart(page=page_no, box=pa.box, crop_id=c.id, crop_path=c.path, text=pa.text,
                        layout_text=layout,
                        ocr_confidence=pa.confidence, source_type=pa.source_type,
                        blur_score=pa.blur, quality_flag=pa.quality_flag,
                        edge_warnings=pa.edge_warnings)
    if index is None:
        q.parts.append(part)
    else:
        q.parts.insert(index, part)
    if q.q_no is None and pa.q_no is not None and len(q.parts) == 1:
        q.q_no = pa.q_no
    q.original_sha256 = next(x.sha256 for x in pr.crops if x.id == q.parts[0].crop_id)
    return part


def _set_part_superseded(pr: Project, part: QuestionPart, value: bool) -> None:
    for c in pr.crops:
        if c.id == part.crop_id or c.source_part == part.crop_id:
            c.superseded = value


def _sort_questions(pr: Project) -> None:
    pr.questions.sort(key=lambda r: (r.page, r.box.y0 // 20, r.box.x0))


def _default_subject(pr: Project, page_no: int, box: Box) -> str | None:
    """Subject of the closest question above this box (users crop in reading order)."""
    before = [q for q in pr.questions if q.subject and (q.page, q.box.y0) <= (page_no, box.y0)]
    return before[-1].subject if before else None


# ---------------------------------------------------------------- user actions
def add_manual_crop(pid: str, page_no: int, box: Box, kind: str, parent_id: str | None = None,
                    q_no: int | None = None, subject: str | None = None,
                    passage_id: str | None = None, snap: bool = False) -> dict:
    """A box drawn by the user. It is used exactly as drawn (snap=False): the user crops,
    the app only reads what is inside.
      question            -> new question, fully analyzed (text, images inside, type, number)
      part                -> continuation of question parent_id (next column / next page)
      diagram/table/option_image -> extra image for question parent_id
      passage             -> shared passage; questions link to it with passage_id
      header/instructions/section -> labelled crop with its text
    """
    from .analyze import analyze_part, assemble_question
    from .config import settings
    with _lock(pid):
        pr = load(pid)
        page = _page(pr, page_no)
        original = load_png(project_dir(pid) / page.path)
        box = box.clip(page.width, page.height)
        if box.w < 8 or box.h < 8:
            raise ValueError("crop box too small")
        result: dict = {}

        if kind == "question":
            n = sum(1 for q in pr.questions if q.id.startswith("qm")) + 1
            qid = f"qm{n:04d}"
            while any(q.id == qid for q in pr.questions) or any(c.id == qid for c in pr.crops):
                n += 1
                qid = f"qm{n:04d}"
            q = QuestionRecord(id=qid, exam=pr.exam, q_no=q_no, page=page_no, box=box, crop_path="",
                               crop_id="", text="", original_sha256="",
                               subject=subject or _default_subject(pr, page_no, box), passage_id=passage_id)
            attach_part(pr, original, q, page_no, box, snap=snap)
            assemble_question(pr, q)
            pr.questions.append(q)
            _sort_questions(pr)
            result["question"] = q.model_dump()

        elif kind == "part":
            q = _question(pr, parent_id or "")
            attach_part(pr, original, q, page_no, box, snap=snap)
            q.edit_history.append(EditEntry(time=now(), field="part", description="continuation part added",
                                            after=q.parts[-1].crop_id))
            assemble_question(pr, q)
            result["question"] = q.model_dump()

        elif kind in ("diagram", "table", "option_image"):
            q = _question(pr, parent_id or "")
            user_box, notes = box, []
            if snap:
                box, notes = snap_box(original, box, include_marker=False)
            c, img = _save_crop(pr, original, RegionKind(kind), page_no, box, f"{q.id}_{kind}", parent=q.id)
            c.user_box, c.snap_notes = (user_box, notes) if snap else (None, [])
            c.needs_review = touches_edge(img)
            q.edit_history.append(EditEntry(time=now(), field="attach", description=f"{kind} crop added",
                                            after=c.id))
            assemble_question(pr, q)
            result["question"] = q.model_dump()
            result["crop"] = c.model_dump()

        elif kind in ("passage", "header", "instructions", "section"):
            k = RegionKind(kind)
            user_box, notes = box, []
            if snap:
                box, notes = snap_box(original, box, include_marker=False)
            c, img = _save_crop(pr, original, k, page_no, box, f"{kind}_{len(pr.crops) + 1:04d}")
            c.user_box, c.snap_notes = (user_box, notes) if snap else (None, [])
            pa = analyze_part(original, page_no, box, script=pr.script)
            c.text, c.ocr_confidence = pa.text, pa.confidence
            first_line = pa.text.strip().splitlines()[0][:40] if pa.text.strip() else None
            c.subject = subject or (first_line if k == RegionKind.section else None)
            c.needs_review = pa.confidence < settings()["ocr"]["confidence_review"]
            for j, g in enumerate(pa.graphics, start=1):
                _save_crop(pr, original, g.kind, page_no, g.box, f"{c.id}_{g.kind.value}{j}",
                           parent=c.id, source_part=c.id)
            if k == RegionKind.passage:
                c.passage_id = f"P{sum(1 for x in pr.crops if x.kind == RegionKind.passage)}"
            result["crop"] = c.model_dump()
        else:
            raise ValueError(f"unknown crop kind '{kind}'")
        save(pr)
        return result


def recrop(pid: str, qid: str, box: Box, part_index: int = 0, snap: bool = False) -> QuestionRecord:
    """Move/resize a question box: cut a NEW crop from the original and re-analyze it.
    The old crop (and the images extracted from it) is kept and can be restored with undo."""
    from .analyze import assemble_question
    with _lock(pid):
        pr = load(pid)
        q = _question(pr, qid)
        if not 0 <= part_index < len(q.parts):
            raise ValueError("bad part index")
        old = q.parts[part_index]
        page = _page(pr, old.page)
        original = load_png(project_dir(pid) / page.path)
        box = box.clip(page.width, page.height)
        if box.w < 8 or box.h < 8:
            raise ValueError("crop box too small")
        q.parts.pop(part_index)
        _set_part_superseded(pr, old, True)
        attach_part(pr, original, q, old.page, box, index=part_index, snap=snap)
        q.superseded_crops.append(old.crop_path)
        q.edit_history.append(EditEntry(time=now(), field="recrop", description="crop box moved",
                                        before=json.dumps({"index": part_index, "part": old.model_dump()}),
                                        after=q.parts[part_index].crop_id))
        assemble_question(pr, q)
        save(pr)
        return q


def remove_part(pid: str, qid: str, part_index: int) -> QuestionRecord:
    from .analyze import assemble_question
    with _lock(pid):
        pr = load(pid)
        q = _question(pr, qid)
        if len(q.parts) < 2 or not 0 <= part_index < len(q.parts):
            raise ValueError("a question needs at least one part")
        old = q.parts.pop(part_index)
        _set_part_superseded(pr, old, True)
        q.edit_history.append(EditEntry(time=now(), field="remove_part", description="part removed",
                                        before=json.dumps({"index": part_index, "part": old.model_dump()})))
        assemble_question(pr, q)
        save(pr)
        return q


def undo(pid: str, qid: str) -> QuestionRecord:
    from .analyze import assemble_question
    with _lock(pid):
        pr = load(pid)
        q = _question(pr, qid)
        if not q.edit_history:
            return q
        last = q.edit_history.pop()
        if last.field == "text":
            q.text_edited = None if last.before in (None, q.text, _base_text(q)) else last.before
        elif last.field == "diagram":
            q.diagram_edited_path = last.before
        elif last.field == "reset":
            snap = _json_dict(last.before)
            if snap is None:          # entries written before reset kept a snapshot: text only
                snap = {"text_edited": last.before}
            q.text_edited = snap.get("text_edited")
            q.diagram_edited_path = snap.get("diagram_edited_path")
        elif last.field == "labels":
            snap = _json_dict(last.before)
            if snap is not None:      # older entries have no snapshot and cannot be restored
                for k, v in snap.items():
                    setattr(q, k, QuestionType(v) if k == "question_type" else v)
        elif last.field in ("recrop", "remove_part"):
            info = json.loads(last.before)
            old = QuestionPart(**info["part"])
            if last.field == "recrop":
                new = q.parts.pop(info["index"])
                _set_part_superseded(pr, new, True)
                if old.crop_path in q.superseded_crops:
                    q.superseded_crops.remove(old.crop_path)
                q.superseded_crops.append(new.crop_path)
            q.parts.insert(info["index"], old)
            _set_part_superseded(pr, old, False)
            q.original_sha256 = next(c.sha256 for c in pr.crops if c.id == q.parts[0].crop_id)
        elif last.field == "part":
            new = next((p for p in q.parts if p.crop_id == last.after), None)
            if new is not None:
                q.parts.remove(new)
                _set_part_superseded(pr, new, True)
        elif last.field == "attach":
            for c in pr.crops:
                if c.id == last.after:
                    c.superseded = True
        q.edited = q.text_edited is not None or q.diagram_edited_path is not None
        assemble_question(pr, q)
        save(pr)
        return q


def reset(pid: str, qid: str) -> QuestionRecord:
    with _lock(pid):
        pr = load(pid)
        q = _question(pr, qid)
        q.edit_history.append(EditEntry(time=now(), field="reset", description="reset to original",
                                        before=json.dumps({"text_edited": q.text_edited,
                                                           "diagram_edited_path": q.diagram_edited_path}),
                                        after=None))
        q.text_edited = None
        q.diagram_edited_path = None
        q.edited = False
        save(pr)
        return q


def verify_hash(pid: str, rel: str, expected: str) -> bool:
    p = project_dir(pid) / rel
    return p.exists() and sha256_file(p) == expected


def remove_question(pid: str, qid: str) -> None:
    """Drop a question from the output. Its crop files are kept (superseded)."""
    with _lock(pid):
        pr = load(pid)
        q = _question(pr, qid)
        part_ids = {p.crop_id for p in q.parts}
        for c in pr.crops:
            if c.parent_id == qid or c.id in part_ids:
                c.superseded = True
        pr.questions = [x for x in pr.questions if x.id != qid]
        save(pr)


def set_fields(pid: str, qid: str, question_type: str | None = None, q_no: int | None = None,
               subject: str | None = None, passage_id: str | None = None) -> QuestionRecord:
    """Reviewer corrections (they become training data). passage_id='' unlinks."""
    from .analyze import assemble_question
    with _lock(pid):
        pr = load(pid)
        q = _question(pr, qid)
        snapshot = json.dumps({k: getattr(q, k) for k in _LABEL_FIELDS}, default=lambda e: e.value)
        changes = []
        if question_type and question_type != q.question_type.value:
            changes.append(f"type {q.question_type.value} -> {question_type}")
            q.question_type = QuestionType(question_type)
            q.multi_answer = q.question_type == QuestionType.multi_correct
            q.type_signals = ["set by reviewer"]
            q.type_confidence = 1.0
            q.labels_reviewed = True
        if q_no is not None and (q_no != q.q_no or q.q_no_inferred):
            changes.append(f"q_no {q.q_no} -> {q_no}")
            q.q_no, q.q_no_inferred = q_no, False
        if subject is not None and (subject or None) != q.subject:
            changes.append(f"subject {q.subject} -> {subject or None}")
            q.subject = subject or None
        if passage_id is not None and (passage_id or None) != q.passage_id:
            changes.append(f"passage {q.passage_id} -> {passage_id or None}")
            q.passage_id = passage_id or None
        if changes:
            q.edit_history.append(EditEntry(time=now(), field="labels", description="; ".join(changes),
                                            before=snapshot))
        assemble_question(pr, q)
        save(pr)
        return q

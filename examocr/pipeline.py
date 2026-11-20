"""Pipeline.

manual mode (default - the USER crops):
    upload -> original pages saved + locked -> blur/quality check -> ready for cropping
    Each box the user draws is analyzed by examocr.analyze (text, images inside, type).

auto mode (the model proposes crops; users can still fix them):
    ... -> working copy -> layout -> segment -> every proposed box goes through the SAME
    analyzer as a user-drawn box -> passages linked -> hash + round-trip check
"""
from __future__ import annotations

import logging
import shutil
import traceback
from pathlib import Path

import numpy as np

from . import store
from .analyze import analyze_part, assemble_question
from .config import exam_profile, settings
from .crop import save_png
from .ingest import load_pages
from .layout import analyze_page
from .preprocess import WorkingCopy, make_working_copy
from .quality import page_quality
from .schemas import PageInfo, Project, QuestionRecord, RegionKind
from .segment import Block, segment_document
from .verify import hash_check, roundtrip

log = logging.getLogger(__name__)
SCRIPTS = ("auto", "printed", "handwritten")


def create_project(src: Path, exam: str, mode: str = "manual", script: str = "auto") -> Project:
    exam_profile(exam)  # validate
    if mode not in ("manual", "auto"):
        raise ValueError("mode must be manual or auto")
    if script not in SCRIPTS:
        raise ValueError("script must be auto, printed or handwritten")
    pr = Project(id=store.new_id(), exam=exam.upper(), source_name=src.name, created=store.now(), mode=mode,
                 script=script)
    d = store.project_dir(pr.id)
    (d / "source").mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, d / "source" / src.name)
    store.save(pr)
    return pr


def run(pid: str, progress=None) -> Project:
    pr = store.load(pid)
    try:
        _run(pr, progress or (lambda msg, frac: None))
        pr.status = "ready"
    except Exception as e:
        log.exception("pipeline failed")
        pr.status, pr.error = "error", f"{e}\n{traceback.format_exc(limit=3)}"
    store.save(pr)
    return pr


def _run(pr: Project, progress) -> None:
    d = store.project_dir(pr.id)
    src = next((d / "source").iterdir())
    progress("loading pages", 0.02)
    originals = load_pages(src)
    auto = pr.mode == "auto"
    wcs: list[WorkingCopy] = []
    layouts = []
    for i, img in enumerate(originals, start=1):
        share = 0.45 if auto else 0.95
        progress(f"page {i}/{len(originals)}: quality check" + (" + layout" if auto else ""),
                 0.05 + share * (i - 1) / len(originals))
        rel = f"pages/page_{i:03d}.png"
        sha = save_png(img, d / rel)
        blur, flag, _ = page_quality(img)
        info = PageInfo(page=i, width=img.shape[1], height=img.shape[0], path=rel, sha256=sha,
                        blur_score=blur, quality_flag=flag)
        if auto:
            wc = make_working_copy(img)
            lay = analyze_page(wc, i)
            wcs.append(wc)
            layouts.append(lay)
            info.skew_deg, info.columns = wc.skew_deg, lay.columns
        pr.pages.append(info)
        store.save(pr)

    if auto:
        _auto_crop(pr, originals, wcs, layouts, progress)
        progress("verifying", 0.96)
        pr.roundtrip = {"hash_errors": hash_check(pr), **roundtrip(pr)}
    progress("ready", 1.0)


def _auto_crop(pr: Project, originals: list[np.ndarray], wcs: list[WorkingCopy], layouts, progress) -> None:
    cfg = settings()
    margin = cfg["layout"]["question_margin_px"]
    blocks = segment_document(layouts, exam_profile(pr.exam).get("subjects", []))
    passages: list[tuple[Block, object]] = []
    n_q = 0
    for k, b in enumerate(blocks):
        progress(f"analyzing crop {k + 1}/{len(blocks)}", 0.5 + 0.45 * k / max(1, len(blocks)))
        boxes = [(p.page, wcs[p.page - 1].to_original(p.box.pad(margin))) for p in b.parts]
        boxes = [(pg, bx) for pg, bx in boxes if bx.area > 0]
        if not boxes:
            continue
        if b.kind == RegionKind.question:
            n_q += 1
            q = QuestionRecord(id=f"q{n_q:04d}", exam=pr.exam, q_no=b.q_no, subject=b.subject,
                               page=boxes[0][0], box=boxes[0][1], crop_path="", crop_id="", text="",
                               original_sha256="")
            for pg, bx in boxes:
                store.attach_part(pr, originals[pg - 1], q, pg, bx)
            assemble_question(pr, q)
            pr.questions.append(q)
            continue
        for pg, bx in boxes:
            c, _ = store._save_crop(pr, originals[pg - 1], b.kind, pg, bx, f"{b.kind.value}_{len(pr.crops) + 1:04d}")
            pa = analyze_part(originals[pg - 1], pg, bx, script=pr.script)
            c.text, c.ocr_confidence, c.subject = pa.text, pa.confidence, b.subject
            c.needs_review = pa.confidence < cfg["ocr"]["confidence_review"] or c.quality_flag != "ok"
            for j, g in enumerate(pa.graphics, start=1):
                store._save_crop(pr, originals[pg - 1], g.kind, pg, g.box, f"{c.id}_{g.kind.value}{j}",
                                 parent=c.id, source_part=c.id)
            if b.kind == RegionKind.passage:
                c.passage_id = b.passage_id
                passages.append((b, c))
    _link_passages(pr, passages)


def _link_passages(pr: Project, passages) -> None:
    """Questions named in the passage's range, or (if no range) up to the next 3 after it."""
    for b, c in passages:
        if b.passage_range:
            lo, hi = b.passage_range
            targets = [q for q in pr.questions if q.q_no is not None and lo <= q.q_no <= hi]
        else:
            targets = [q for q in pr.questions
                       if (q.page, q.box.y0) > (c.page, c.box.y0) and q.subject == c.subject][:3]
        for q in targets:
            if q.passage_id is None:
                q.passage_id = b.passage_id
                assemble_question(pr, q)

"""Question type analysis.

Every type gets a score from three kinds of evidence:
  1. keywords in the text, matched fuzzily so OCR slips ("Assertlon", "Colunm I") still count
  2. the SHAPE of the options ("Both A and R ...", "P-ii, Q-i", "B, A, D, C", no options)
  3. structure: diagram/table inside the crop, numbered statements, linked passage,
     plus the exam profile's own keyword hints (configs/exams/*.yaml)
The best score wins; confidence comes from the margin over the runner-up. When the margin is
small and a trained classifier exists (training/classifier), it breaks the tie.
The reasons are returned as `signals`, so a reviewer sees WHY a type was chosen.
"""
from __future__ import annotations

import logging
import re
from collections import defaultdict
from functools import lru_cache

from rapidfuzz import fuzz

from .config import ROOT, exam_profile, settings
from .schemas import QuestionType as T

log = logging.getLogger(__name__)

# (type, keyword, weight)
KEYWORDS: list[tuple[T, str, float]] = [
    (T.assertion_reason, "assertion", 3.0), (T.assertion_reason, "reason (r)", 2.0),
    (T.assertion_reason, "assertion (a)", 1.0),
    (T.match, "column i", 3.0), (T.match, "column-i", 3.0), (T.match, "list-i", 3.0),
    (T.match, "list i", 2.0), (T.match, "match the following", 3.0), (T.match, "match column", 3.0),
    (T.match, "match list", 3.0), (T.match, "list-ii", 1.0), (T.match, "column ii", 1.0),
    (T.statement, "statement i", 2.5), (T.statement, "statement-i", 2.5), (T.statement, "statement 1", 2.0),
    (T.statement, "which of the following statements", 2.5), (T.statement, "two statements", 1.5),
    (T.statement, "correct statements", 2.0), (T.statement, "incorrect statements", 2.0),
    (T.arrange, "arrange", 3.0), (T.arrange, "correct order", 2.5), (T.arrange, "correct sequence", 2.5),
    (T.arrange, "increasing order", 2.5), (T.arrange, "decreasing order", 2.5),
    (T.arrange, "chronological order", 2.5),
    (T.multi_correct, "one or more", 3.5), (T.multi_correct, "more than one", 3.5),
    (T.multi_correct, "multiple correct", 3.5),
    (T.numerical, "integer", 2.5), (T.numerical, "numerical value", 3.0), (T.numerical, "nearest integer", 3.0),
    (T.numerical, "correct upto", 2.0), (T.numerical, "decimal places", 2.0),
    (T.other, "fill in the blank", 3.0), (T.other, "odd one out", 3.0), (T.other, "how many of the", 3.0),
    (T.other, "how many of the following", 3.0),
    (T.diagram, "figure", 1.0), (T.diagram, "diagram", 1.0), (T.diagram, "shown below", 0.8),
    (T.diagram, "given graph", 1.0), (T.diagram, "shown in the", 0.8), (T.diagram, "identify the structure", 1.5),
    (T.paragraph, "paragraph", 2.0), (T.paragraph, "passage", 2.0), (T.paragraph, "based on the above", 2.5),
    (T.paragraph, "comprehension", 2.0),
]

PROFILE_KEYS = {"paragraph": T.paragraph, "multi_correct": T.multi_correct, "numerical": T.numerical,
                "match": T.match, "assertion_reason": T.assertion_reason, "statement": T.statement,
                "arrange": T.arrange, "diagram": T.diagram}

# Option-shape patterns
PAIR = re.compile(r"\b[A-DP-Sa-dp-s]\s*[-–→>:]\s*\(?(?:i{1,3}|iv|v|vi|[1-6]|[p-tP-T])\)?\b")
SEQ = re.compile(r"^\s*\(?[A-Da-d1-4][.)]?\)?\s*([A-Ea-eP-Tp-t](?:\s*(?:,|→|->|>|<|-)\s*[A-Ea-eP-Tp-t]){2,})\s*$")
AR_OPT = re.compile(r"\b(?:both\s+)?(?:\(?A\)?\s+and\s+\(?R\)?|A\s+is\s+(?:true|false|correct|incorrect)|"
                    r"R\s+is\s+(?:true|false|correct|incorrect|the\s+correct\s+explanation))", re.I)
ST_OPT = re.compile(r"statement\s*[-–]?\s*(?:I{1,2}|1|2)\b|\bonly\s*\(?[a-e]\)?|\(?[a-e]\)?\s*,?\s*and\s*\(?[a-e]\)?\s*only",
                    re.I)
STATEMENT_LINE = re.compile(r"^\s*(?:\(?(?:i{1,3}|iv|v|vi)\)|\(?[a-e]\)|[A-E][.)])\s+\S", re.I | re.M)
BLANK = re.compile(r"_{4,}|\.{6,}")


def _norm(t: str) -> str:
    return re.sub(r"\s+", " ", t.lower())


def _kw(text: str, kw: str, threshold: int) -> bool:
    if len(text) < len(kw):
        return False
    if re.search(r"(?<!\w)" + re.escape(kw) + r"(?!\w)", text):
        return True
    # fuzzy only for longer keywords: short ones ("arrange") would fire on unrelated words
    return len(kw) >= 8 and fuzz.partial_ratio(kw, text) >= threshold


def score_types(text: str, options: list[str], statements: list[str], has_diagram: bool,
                has_table: bool, has_passage: bool, exam: str | None = None):
    thr = settings()["classify"]["fuzzy_threshold"]
    t = _norm(text)
    scores: dict[T, float] = defaultdict(float)
    signals: dict[T, list[str]] = defaultdict(list)

    def add(tp: T, w: float, why: str):
        scores[tp] += w
        signals[tp].append(why)

    hit_kw: set[str] = set()
    for tp, kw, w in KEYWORDS:
        if _kw(t, kw, thr):
            # don't double count overlapping keywords of one type (e.g. "list i" in "list-i")
            if any(kw in other or other in kw for other in hit_kw):
                continue
            hit_kw.add(kw)
            add(tp, w, f"keyword '{kw}'")

    if exam:
        try:
            for key, kws in (exam_profile(exam).get("type_hints") or {}).items():
                tp = PROFILE_KEYS.get(key)
                if tp and any(_kw(t, k.lower(), thr) for k in kws):
                    add(tp, 0.5, f"{exam} profile hint")
        except ValueError:
            pass

    opts = " ".join(options)
    if options:
        pairs = sum(len(PAIR.findall(o)) >= 2 for o in options)
        if pairs >= 2:
            add(T.match, 3.0, f"{pairs} options are pairings like P-ii, Q-i")
        seqs = sum(bool(SEQ.match(o)) for o in options)
        if seqs >= 2:
            add(T.arrange, 2.5, f"{seqs} options are orderings like B, A, D, C")
        if len(AR_OPT.findall(opts)) >= 2:
            add(T.assertion_reason, 3.0, "options talk about A and R")
        if len(ST_OPT.findall(opts)) >= 2:
            add(T.statement, 2.0, "options refer to statements")
        if len(options) >= 3:
            add(T.mcq, 1.0, f"{len(options)} options")
            if scores.get(T.numerical):           # "x is an integer" inside an MCQ
                scores[T.numerical] *= 0.3
    else:
        add(T.numerical, 2.0, "no options")
        if BLANK.search(text):
            add(T.other, 1.5, "blank to fill")

    # Count statement lines in the STEM only - option lines "(a) closed (b) adiabatic" are not statements.
    from .parse import split_options
    n_stmt = len(statements) or len(STATEMENT_LINE.findall(split_options(text)[0]))
    if n_stmt >= 2:
        add(T.statement, 1.0, f"{n_stmt} numbered statements")
    if has_table:
        add(T.match, 1.0, "table inside the crop")
    if has_diagram:
        add(T.diagram, 2.0, "diagram inside the crop")
    if has_passage:
        add(T.paragraph, 10.0, "linked to a passage")
    if BLANK.search(text) and options:
        add(T.other, 1.0, "blank in the question")
    return scores, signals


@lru_cache(maxsize=1)
def _model():
    path = ROOT / settings()["classify"]["classifier_path"]
    if not path.exists():
        return None
    try:
        import joblib
        return joblib.load(path)
    except Exception as e:
        log.warning("Could not load question-type classifier: %s", e)
        return None


def analyze_type(text: str, options: list[str], statements: list[str] | None = None,
                 has_diagram: bool = False, has_table: bool = False, has_passage: bool = False,
                 exam: str | None = None) -> tuple[T, float, list[str]]:
    """Returns (type, confidence 0-1, signals explaining the choice)."""
    scores, signals = score_types(text, options, statements or [], has_diagram, has_table,
                                  has_passage, exam)
    if not scores:
        return T.mcq, 0.5, ["no clear evidence"]
    ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    top, s1 = ranked[0]
    s2 = ranked[1][1] if len(ranked) > 1 else 0.0
    conf = 0.5 + 0.5 * (s1 - s2) / s1 if s1 > 0 else 0.5
    why = signals[top]

    if conf < 0.7:
        model = _model()
        if model is not None:
            proba = model.predict_proba([text])[0]
            i = int(proba.argmax())
            mt = T(model.classes_[i])
            if mt in (top, ranked[1][0] if len(ranked) > 1 else top):
                top, conf = mt, max(conf, float(proba[i]))
                why = signals[mt] + [f"trained classifier ({proba[i]:.2f})"]
    if top == T.multi_correct:
        why = why + ["multi-answer"]
    return top, round(min(conf, 0.99), 3), why


def classify(text: str, options, has_diagram: bool = False, has_passage: bool = False,
             exam: str | None = None) -> tuple[T, float]:
    """Backward-compatible wrapper: options may be a list or a count."""
    if isinstance(options, int):
        options = ["(x) option"] * options
    tp, conf, _ = analyze_type(text, options, None, has_diagram, False, has_passage, exam)
    return tp, conf

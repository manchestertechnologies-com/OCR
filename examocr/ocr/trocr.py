"""Offline handwriting OCR: TrOCR for handwritten text, a TrOCR math model for formulas.

TrOCR reads one short line image at a time, so the crop is first cut into pieces:
  1. binarize a throwaway copy, find text LINES from the bands where letter-sized ink is
     dense (superscripts like the "2+" of Hg²⁺ and the "4" of 10⁴ join the line they belong
     to, not a line of their own)
  2. split each line at wide gaps (the a) ... b) option columns) and cut very long lines at
     word gaps, so TrOCR never has to squash a full page-width line into 384 px
  3. read every piece with the handwriting model, from the ORIGINAL pixels; ink of a
     neighbouring line that pokes into the piece is whitened in the OCR input only
  4. pieces that look like a formula (mostly digits/symbols, or raised/lowered script ink)
     are read again with the math model; its LaTeX is written as plain Unicode text
     (H₂ + Hg²⁺ ⇌ 2Hg + 2H⁺, -6.27×10⁴ J) so the text stays readable and editable.
Fine-tune on your own handwritten samples: training/handwriting/finetune_trocr.py
"""
from __future__ import annotations

import logging
import re
import threading
from collections import Counter

import cv2
import numpy as np

from ..schemas import Box, Word
from .base import OCREngine, OCRResult
from .handseg import binarize, segment

log = logging.getLogger(__name__)


from .latex import SUB, SUP, latex_to_text  # noqa: E402,F401  (shared with the GLM-OCR reader)


# ------------------------------------------------------------------ engine
LABEL = re.compile(r"^\(?\s*([a-dA-D]|[1-9]\d{0,2})\s*([)\].])$")
LABEL_ALONE = re.compile(r"^\(?\s*([a-dA-D1-4])\s*([)\].]?)$")    # read from a label-only crop
SEQ = ("abcd", "ABCD", "1234")


def _words(text: str) -> int:
    """Real words (3+ letters) in a text-model reading - formulas have at most one."""
    return len(re.findall(r"[A-Za-z]{3,}", text))


def _latex_letter_run(latex: str) -> bool:
    """A run of 4+ plain letters in the math model's LaTeX means it was reading words."""
    plain = re.sub(r"\\[A-Za-z]+", " ", latex)
    return bool(re.search(r"[A-Za-z]{4,}", plain))


def choose_math(text: str, tconf: float, latex: str, mconf: float) -> bool:
    """Formula or words? Measured on real handwritten papers: on formulas the math model is
    sure (>= 0.75) and clearly surer than the text model; on words it is not, or it strings
    letters together ("mentionedi")."""
    return bool(latex.strip()) and mconf >= 0.75 and mconf - tconf >= 0.15 and not _latex_letter_run(latex)


def normalize_label(t: str, alone: bool = False) -> str | None:
    """'a )' -> 'a)'. alone: the crop held only the label, so a bare 'a' is 'a)' too."""
    t = t.replace(" ", "")
    m = LABEL.match(t) or (LABEL_ALONE.match(t) if alone else None)
    return f"{m.group(1)}{m.group(2) or ')'}" if m else None


def looks_like_math(text: str, scripts: int) -> bool:
    """Kept for callers/tests: a cheap guess from the text-model reading alone."""
    body = re.sub(r"^\s*(\(?[a-dA-D1-4][.)]|\([a-dA-D1-4]\))\s*", "", text)
    b = [c for c in body if not c.isspace()]
    if not b:
        return False
    sym = sum(c in "0123456789+-=×*/^_<>()[]{}.%°±" for c in b) / len(b)
    return (sym >= 0.45 or scripts >= 1) and _words(body) <= 1


_LOCK = threading.Lock()


def _total_ram_gb() -> float:
    try:
        import psutil
        return psutil.virtual_memory().total / 2**30
    except Exception:
        return 8.0


class TrOCREngine(OCREngine):
    name = "trocr"

    def __init__(self, model: str | None = None, math_model: str | None = None,
                 beams: int | None = None, quantize: bool | None = None):
        import torch
        import transformers
        from transformers import TrOCRProcessor, VisionEncoderDecoderModel

        from ..config import settings
        transformers.logging.set_verbosity_error()
        transformers.logging.disable_progress_bar()
        cfg = settings()["ocr"]
        self.torch = torch
        torch.set_num_threads(max(1, (torch.get_num_threads() or 4)))
        self.beams = int(beams if beams is not None else cfg.get("trocr_beams", 1))
        # int8 dynamic quantization makes TrOCR output garbage - off unless asked for
        quantize = cfg.get("trocr_quantize", False) if quantize is None else quantize
        model = model or cfg.get("trocr_model", "microsoft/trocr-base-handwritten")
        math_model = math_model if math_model is not None else cfg.get("trocr_math_model", "")

        # On an 8 GB laptop both models in float32 (~5 GB with PyTorch) push Windows into
        # swapping and every crop crawls. low_memory keeps the big formula model in bfloat16
        # (half the RAM). auto = on when the computer has less than 12 GB.
        low = cfg.get("low_memory", "auto")
        if low == "auto":
            low = _total_ram_gb() < 12
        self.low_memory = bool(low)

        def load(name, half=False):
            proc = TrOCRProcessor.from_pretrained(name)
            m = VisionEncoderDecoderModel.from_pretrained(
                name, dtype=torch.bfloat16 if half else torch.float32).eval()
            # transformers 5 builds models on the "meta" device; TrOCR's sinusoidal position
            # table is a plain tensor (not a weight), so it stays empty - rebuild it
            for mod in m.modules():
                w = getattr(mod, "weights", None)
                if isinstance(w, torch.Tensor) and w.is_meta and hasattr(mod, "get_embedding"):
                    # ...in the model's dtype: a float32 table in a bfloat16 model fails with
                    # "mat1 and mat2 must have the same dtype" on every crop (low_memory mode)
                    mod.weights = mod.get_embedding(w.size(0), mod.embedding_dim, mod.padding_idx).to(w.dtype)
            if quantize:
                m = torch.ao.quantization.quantize_dynamic(m, {torch.nn.Linear}, dtype=torch.qint8)
            m.generation_config.use_cache = True     # shipped off; ~2x faster decoding
            return proc, m

        log.info("loading handwriting model %s", model)
        self.proc, self.model = load(model)
        self.math = None
        if math_model:
            try:
                log.info("loading math model %s", math_model)
                self.math = load(math_model, half=self.low_memory)
            except Exception as e:   # math model is optional
                log.warning("math model %s unavailable: %s", math_model, e)

    def _generate(self, proc, model, images: list, max_new_tokens: int = 64,
                  batch_size: int = 6) -> list[tuple[str, float]]:
        """Images -> [(text, confidence)] in the same order. Images of similar width go in one
        batch: a batch decodes until its longest member is done, so a short label next to a
        full line would wait for it."""
        order = sorted(range(len(images)), key=lambda i: images[i].width / max(1, images[i].height))
        res: dict[int, tuple[str, float]] = {}
        for k in range(0, len(order), batch_size):
            ids = order[k:k + batch_size]
            for i, r in zip(ids, self._generate_batch(proc, model, [images[i] for i in ids], max_new_tokens)):
                res[i] = r
        return [res[i] for i in range(len(images))]

    def _generate_batch(self, proc, model, batch: list, max_new_tokens: int) -> list[tuple[str, float]]:
        out: list[tuple[str, float]] = []
        if batch:
            pv = proc(images=batch, return_tensors="pt").pixel_values.to(model.dtype)
            extra = {"early_stopping": True} if self.beams > 1 else {}
            with self.torch.no_grad(), _LOCK:
                g = model.generate(pv, num_beams=self.beams, max_new_tokens=max_new_tokens,
                                   output_scores=True, return_dict_in_generate=True, **extra)
                try:
                    tr = model.compute_transition_scores(g.sequences, g.scores, getattr(g, "beam_indices", None),
                                                         normalize_logits=self.beams == 1)
                except Exception:
                    tr = None
            texts = proc.batch_decode(g.sequences, skip_special_tokens=True)
            pad = next((v for v in (getattr(model.generation_config, "pad_token_id", None),
                                    getattr(getattr(model.config, "decoder", None), "pad_token_id", None))
                        if v is not None), -1)
            for b, t in enumerate(texts):
                conf = 0.5
                if tr is not None:
                    seq = g.sequences[b, 1:1 + tr.shape[1]]
                    lp = tr[b][(seq != pad)[: tr.shape[1]]]
                    lp = lp.float()
                    lp = lp[self.torch.isfinite(lp)]
                    if lp.numel():
                        conf = float(self.torch.exp(lp.mean()).clamp(0, 1))
                out.append((t.strip(), conf))
        return out

    def _crop(self, rgb: np.ndarray, bw: np.ndarray, lay, p, x0: int | None = None, x1: int | None = None):
        """OCR input for piece p (or its x-range [x0, x1)): original pixels, with ink of other
        pieces (a neighbour line's descenders) whitened. A throwaway copy - never saved."""
        from PIL import Image
        H, W = rgb.shape[:2]
        pad = int(0.25 * lay.line_h)
        lo, hi = (p.box.x0 if x0 is None else x0), (p.box.x1 if x1 is None else x1)
        b = Box(x0=lo, y0=p.box.y0, x1=hi, y1=p.box.y1).pad(pad).clip(W, H)
        piece = rgb[b.y0:b.y1, b.x0:b.x1].copy()
        own = np.isin(lay.labels[b.y0:b.y1, b.x0:b.x1], p.comps)
        other = (bw[b.y0:b.y1, b.x0:b.x1] > 0) & ~own
        piece[cv2.dilate(other.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0] = 255
        if x0 is not None:
            piece[:, :max(0, lo - b.x0)] = 255
        if x1 is not None:
            piece[:, max(0, hi - b.x0):] = 255
        return Image.fromarray(piece)

    @staticmethod
    def _lead(p, lay, gap: float) -> tuple[int, int] | None:
        """A leading label ("a)", "7]") = a narrow first cluster followed by a gap.
        Returns (end of the label, start of the rest) or None."""
        st, lh = lay.stats, lay.line_h
        iv = sorted((int(st[i, 0]), int(st[i, 0] + st[i, 2])) for i in p.comps if st[i, 3] > 0.25 * lay.char_h)
        runs: list[list[int]] = []
        for a, b in iv:
            if runs and a <= runs[-1][1] + gap * lh:
                runs[-1][1] = max(runs[-1][1], b)
            else:
                runs.append([a, b])
        if len(runs) >= 2 and runs[0][1] - runs[0][0] <= 1.4 * lh:
            return runs[0][1], runs[1][0]
        return None

    def read(self, image: np.ndarray, skip_mask: np.ndarray | None = None) -> OCRResult:
        """skip_mask: pixels (crop coordinates) to ignore - drawings already cut out as diagrams."""
        if image.size == 0:
            return OCRResult()
        bw = binarize(image)
        if skip_mask is not None:
            bw[skip_mask] = 0
        lay = segment(bw)
        pieces = lay.pieces
        if not pieces:
            return OCRResult()
        rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB) if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        crops = [self._crop(rgb, bw, lay, p) for p in pieces]

        # pieces that start a line or a column ("a) ..." "b) ...") may begin with a label
        leads: dict[int, tuple[int, int]] = {}
        first_lines = sorted({p.line for p in pieces if p.box.w > 2 * lay.line_h})[:2]
        for k, p in enumerate(pieces):
            prev = pieces[k - 1] if k and pieces[k - 1].line == p.line else None
            if prev is not None and p.box.x0 - prev.box.x1 < 2 * lay.line_h:
                continue
            ld = self._lead(p, lay, 0.3) or (self._lead(p, lay, 0.18) if p.line in first_lines else None)
            if ld:
                leads[k] = ld
        lead_ks = list(leads)
        results = self._generate(self.proc, self.model, crops)
        labels: dict[int, str | None] = {}
        for k in lead_ks:                             # the line reading may already start with it
            toks = results[k][0].split()
            labels[k] = (normalize_label("".join(toks[:2])) or normalize_label(toks[0])) if toks else None
        # otherwise read the label on its own (each image costs ~2 s on a laptop CPU: only these)
        todo = [k for k in lead_ks if not labels[k]]
        for k, (t, _) in zip(todo, self._generate(self.proc, self.model,
                                                  [self._crop(rgb, bw, lay, pieces[k], x1=leads[k][0])
                                                   for k in todo], max_new_tokens=8)):
            labels[k] = normalize_label(t, alone=True)
        _fill_sequence(labels, lead_ks)

        done: set[int] = set()
        if self.math is not None:
            # the math model is ~3x slower: only pieces that could be a formula (not sure
            # sentences, not a lone speck)
            cand = [k for k, (t, c) in enumerate(results)
                    if c < 0.9 and pieces[k].box.w >= 1.2 * lay.line_h
                    and (_words(t) <= 1 or (pieces[k].scripts and _words(t) <= 2))]
            if cand:
                mcrops = [self._crop(rgb, bw, lay, pieces[k], x0=leads[k][0] + 1) if labels.get(k) else crops[k]
                          for k in cand]
                for k, (latex, mconf) in zip(cand, self._generate(*self.math, mcrops, max_new_tokens=48, batch_size=4)):
                    t, c = results[k]
                    if choose_math(t, c, latex, mconf):
                        body = latex_to_text(latex)
                        results[k] = ((labels[k] + " " + body) if labels.get(k) else body, max(mconf, c))
                        done.add(k)
        for k, lab in labels.items():                 # the label as read on its own
            if not lab or k in done:
                continue
            t, c = results[k]
            toks = t.split()
            if len(toks) >= 2 and normalize_label(toks[0] + toks[1]):
                toks = toks[2:]
            elif toks and len(toks[0]) <= 3:
                toks = toks[1:]
            results[k] = (" ".join([lab] + toks), c)

        words = [Word(text=t, conf=c, box=p.box, line_id=p.line)
                 for p, (t, c) in zip(pieces, results) if t]
        return OCRResult(words=words)


def _fill_sequence(labels: dict[int, str | None], order: list[int]) -> None:
    """Option labels run a, b, c, d (or 1-4). When at least two of 3-4 consecutive labelled
    pieces were read as that run, the others take their place in it (a writer may mix
    "a)" and "B)"; a "9." among letters is a misread "c)")."""
    best = None
    for seq in ("abcd", "1234"):
        for s in range(len(order)):
            win = order[s:s + 4]
            if len(win) < 3:
                break
            got = [labels.get(k) for k in win]
            same = [g if g and g[0].isdigit() == seq[0].isdigit() else None for g in got]
            hits = sum(1 for i, g in enumerate(same) if g and g[0].lower() == seq[i])
            clash = any(g and g[0].lower() != seq[i] for i, g in enumerate(same))
            if hits >= 2 and not clash and (best is None or hits > best[0]):
                close = Counter(g[-1] for g in same if g).most_common(1)[0][0]
                upper = sum(g[0].isupper() for g in same if g) > hits / 2
                best = (hits, win, seq.upper() if upper else seq, close, same)
    if best:
        _, win, seq, close, same = best
        for i, k in enumerate(win):
            if not same[i]:
                labels[k] = seq[i] + close

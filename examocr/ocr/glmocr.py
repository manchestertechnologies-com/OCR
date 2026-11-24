"""Offline document OCR with GLM-OCR (zai-org/GLM-OCR, 0.9B, MIT licence).

One vision-language model reads printed text, handwriting, formulas and tables, writing
Markdown with LaTeX formulas. It runs on this computer's CPU (no internet after the first
download, ~2.6 GB) and is far more accurate on handwritten papers than line-by-line TrOCR.

How a crop is read:
  * the crop is scaled so letters are ~16 px tall (a 300-DPI scan has ~28 px letters: reading
    it at full size costs 3-4x the time for the same text) - never enlarged
  * a white margin is added (letters touching the image edge are read worse)
  * greedy decoding; the mean token probability is the confidence
  * a reading that starts repeating itself is stopped and flagged, never padded out
The output is converted to plain text (H₂O, M/3, →) by latex.markdown_to_text.
"""
from __future__ import annotations

import logging
import threading

import cv2
import numpy as np

from ..schemas import Box, Word
from .base import OCREngine, OCRResult
from .latex import markdown_lines

log = logging.getLogger(__name__)
MODEL = "zai-org/GLM-OCR"
PROMPTS = {"text": "Text Recognition:", "formula": "Formula Recognition:", "table": "Table Recognition:"}


def char_height(image: np.ndarray) -> float:
    """Median height of letter-sized ink marks (px). 0 when the image has no ink."""
    from .handseg import binarize
    bw = binarize(image)
    n, _, st, _ = cv2.connectedComponentsWithStats((bw > 0).astype(np.uint8), 8)
    if n <= 1:
        return 0.0
    h, a = st[1:, 3], st[1:, 4]
    ok = (a >= 12) & (h >= 5) & (h < 0.5 * max(8, image.shape[0]))
    return float(np.median(h[ok])) if ok.any() else 0.0


def has_ink(image: np.ndarray, min_px: int = 40) -> bool:
    from .handseg import binarize
    return image.size > 0 and int((binarize(image) > 0).sum()) >= min_px


def prepare(image: np.ndarray, char_h: float | None = None, target: float = 16.0,
            max_pixels: int = 1_400_000, margin: int = 12) -> np.ndarray:
    """RGB model input: scaled to ~target px letters (never enlarged), white margin, at
    least 56 px tall/wide (the model works on 28 px tiles)."""
    rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB) if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    ch = char_h if char_h else char_height(image)
    rgb = squeeze_blank(rgb, ch)
    s = min(1.0, target / ch) if ch > 0 else 1.0
    H, W = rgb.shape[:2]
    s = min(s, (max_pixels / max(1, H * W)) ** 0.5)
    if s < 0.98:
        rgb = cv2.resize(rgb, (max(1, round(W * s)), max(1, round(H * s))), interpolation=cv2.INTER_AREA)
    H, W = rgb.shape[:2]
    py, px = max(margin, (56 - H + 1) // 2), max(margin, (56 - W + 1) // 2)
    return cv2.copyMakeBorder(rgb, py, py, px, px, cv2.BORDER_CONSTANT, value=(255, 255, 255))


def squeeze_blank(rgb: np.ndarray, char_h: float) -> np.ndarray:
    """The model's time grows with the image area, and blank paper (margins, the white left
    where a drawing was blanked) costs as much as text. Trim the margins and shorten every
    blank stretch between lines to about one letter height. Only rows/columns with no ink at
    all are removed, so no writing is touched. OCR input only."""
    from .handseg import binarize
    ink = binarize(rgb) > 0
    rows, cols = ink.any(axis=1), ink.any(axis=0)
    if not rows.any():
        return rgb
    gap = max(4, int(round(max(char_h, 6.0))))
    r0, r1 = int(np.argmax(rows)), len(rows) - int(np.argmax(rows[::-1]))
    c0, c1 = int(np.argmax(cols)), len(cols) - int(np.argmax(cols[::-1]))
    keep = np.zeros(len(rows), bool)
    keep[r0:r1] = True
    run = 0
    for y in range(r0, r1):                      # a blank stretch keeps its first `gap` rows
        run = 0 if rows[y] else run + 1
        if run > gap:
            keep[y] = False
    return rgb[keep][:, max(0, c0 - 2):min(len(cols), c1 + 2)]


def split_bands(image: np.ndarray, char_h: float, max_lines: int = 7) -> list[np.ndarray]:
    """Cut a tall crop into horizontal bands of at most ~max_lines text lines, at the row with
    the least ink. A letter crossing a cut belongs wholly to the band holding its centre (it
    is whitened in the other one), so no letter is read in half. OCR input only."""
    from .handseg import binarize
    H = image.shape[0]
    ch = max(char_h, 6.0)
    max_h = int(max_lines * 2.3 * ch)
    if max_lines <= 0 or H <= 1.4 * max_h:
        return [image]
    bw = binarize(image) > 0
    rows = bw.sum(axis=1)
    # cut only at a fully blank row: on a tilted photo a "least ink" row runs through a
    # slanted line, and the model then invents the half line it cannot see
    blank = np.flatnonzero(rows == 0)
    cuts, y = [0], 0
    while H - y > 1.4 * max_h:
        lo, hi = y + int(0.5 * max_h), y + max_h
        cand = blank[(blank >= lo) & (blank < min(hi, H - int(0.4 * max_h)))]
        if not cand.size:                                        # none in reach: the next one
            cand = blank[(blank >= hi) & (blank < H - int(0.4 * max_h))][:1]
        if not cand.size:
            break
        y = int(cand[-1])
        cuts.append(y)
    cuts.append(H)
    if len(cuts) == 2:
        return [image]
    n, labels, st, _ = cv2.connectedComponentsWithStats(bw.astype(np.uint8), 8)
    cy = st[:, 1] + st[:, 3] / 2.0
    out = []
    for a, b in zip(cuts[:-1], cuts[1:]):
        own = (cy >= a) & (cy < b)
        own[0] = False
        y0 = int(min([a] + [st[i, 1] for i in np.flatnonzero(own)]))
        y1 = int(max([b] + [st[i, 1] + st[i, 3] for i in np.flatnonzero(own)]))
        band = image[y0:y1].copy()
        other = (labels[y0:y1] > 0) & ~own[labels[y0:y1]]
        band[other] = 255
        out.append(band)
    return out


def repeat_tail(ids: list[int], min_repeats: int = 4, min_span: int = 48, max_period: int = 24) -> int:
    """Length of a loop at the end of ids (the same 1-24 tokens 4+ times, 48+ tokens), else 0.
    Returns how many tokens to drop to keep one copy."""
    n = len(ids)
    for p in range(1, max_period + 1):
        r = 1
        while n >= (r + 1) * p and ids[n - (r + 1) * p:n - r * p] == ids[n - p:]:
            r += 1
        if r >= min_repeats and r * p >= min_span:
            return (r - 1) * p
    return 0


class GlmOcrEngine(OCREngine):
    name = "glm"

    def __init__(self, model: str | None = None, dtype: str | None = None):
        import torch
        import transformers
        from transformers import AutoModelForImageTextToText, AutoProcessor, StoppingCriteria

        from ..config import settings
        transformers.logging.set_verbosity_error()
        transformers.logging.disable_progress_bar()
        cfg = settings()["ocr"]
        self.torch = torch
        try:
            import psutil
            cores = psutil.cpu_count(logical=False) or 4
        except Exception:
            cores = 4
        torch.set_num_threads(int(cfg.get("glm_threads") or cores))   # hyper-threads only slow it down
        name = model or cfg.get("glm_model", MODEL)
        # float32: this CPU class has no fast bfloat16 - bf16 is 2.4x slower here, same text
        dt = getattr(torch, dtype or cfg.get("glm_dtype", "float32"))
        self.target = float(cfg.get("glm_letter_px", 16))
        self.band_lines = int(cfg.get("glm_band_lines", 0))   # 0 = whole crop in one pass
        log.info("loading OCR model %s (%s)", name, dt)
        self.proc = AutoProcessor.from_pretrained(name)
        self.model = AutoModelForImageTextToText.from_pretrained(name, dtype=dt, attn_implementation="sdpa").eval()
        self._lock = threading.Lock()

        class _StopLoop(StoppingCriteria):
            def __init__(self, start: int):
                self.start, self.looped = start, False

            def __call__(self, input_ids, scores, **kw):
                ids = input_ids[0, self.start:].tolist()
                if len(ids) >= 48 and len(ids) % 8 == 0 and repeat_tail(ids):
                    self.looped = True
                    return torch.ones(input_ids.shape[0], dtype=torch.bool)
                return torch.zeros(input_ids.shape[0], dtype=torch.bool)
        self._StopLoop = _StopLoop

    # ------------------------------------------------------------------ one image
    def recognize(self, image: np.ndarray, task: str = "text", char_h: float | None = None) -> tuple[str, float, bool]:
        """-> (model output as Markdown/LaTeX, confidence 0-1, looped)."""
        if image.size == 0 or not has_ink(image):
            return "", 1.0, False
        from PIL import Image
        torch = self.torch
        ch = char_h or char_height(image)
        rgb = prepare(image, ch, self.target)
        msgs = [{"role": "user", "content": [{"type": "image", "image": Image.fromarray(rgb)},
                                             {"type": "text", "text": PROMPTS[task]}]}]
        inputs = self.proc.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True,
                                               return_dict=True, return_tensors="pt")
        inputs.pop("token_type_ids", None)
        start = inputs["input_ids"].shape[1]
        # a ceiling, not a target: lines in the image x letters per line, as tokens
        letter = max(6.0, min(ch, self.target) if ch > 0 else self.target)   # letter height in rgb
        lines = max(1.0, rgb.shape[0] / (2.0 * letter))
        per_line = max(24.0, rgb.shape[1] / letter)
        max_new = int(min(2048, max(64, 1.3 * lines * per_line)))
        stop = self._StopLoop(start)
        with torch.inference_mode(), self._lock:
            out = self.model.generate(**inputs, max_new_tokens=max_new, do_sample=False,
                                      output_scores=True, return_dict_in_generate=True,
                                      stopping_criteria=[stop])
            try:
                tr = self.model.compute_transition_scores(out.sequences, out.scores, normalize_logits=True)[0]
            except Exception:
                tr = None
        ids = out.sequences[0, start:].tolist()
        looped = stop.looped or len(ids) >= max_new
        if stop.looped:
            ids = ids[:len(ids) - repeat_tail(ids)]
        conf = 0.5
        if tr is not None and len(ids):
            lp = tr[:len(ids)].float()
            lp = lp[torch.isfinite(lp)]
            if lp.numel():
                conf = float(torch.exp(lp.mean()).clamp(0, 1))
        text = self.proc.decode(ids, skip_special_tokens=True).strip()
        return text, conf, looped

    def read_text(self, image: np.ndarray, char_h: float | None = None) -> tuple[str, float, bool]:
        """-> (plain text with the image's line breaks, confidence, looped).
        A tall crop is read in bands of ~7 lines: on a whole question the model skips a line
        now and then (measured: "C) 4 3 2 1" of a 2-column answer grid); on bands it doesn't."""
        plain, _, conf, looped = self.read_text_both(image, char_h)
        return plain, conf, looped

    def read_text_both(self, image: np.ndarray, char_h: float | None = None) -> tuple[str, str, float, bool]:
        """-> (plain text, the same lines with formulas as LaTeX $...$, confidence, looped)."""
        ch = char_h or char_height(image)
        plains, latexes, confs, looped = [], [], [], False
        for band in split_bands(image, ch, self.band_lines):
            md, c, lp = self.recognize(band, "text", ch)
            lines = markdown_lines(md)
            t = "\n".join(p for p, _ in lines)
            if t.strip():
                plains.append(t)
                latexes.append("\n".join(x for _, x in lines))
                confs.append((c, len(t)))
            looped |= lp
        n = sum(k for _, k in confs)
        conf = sum(c * k for c, k in confs) / n if n else 1.0
        return "\n".join(plains), "\n".join(latexes), conf, looped

    # ------------------------------------------------------------------ OCREngine interface
    def read(self, image: np.ndarray) -> OCRResult:
        """One Word per text line. The model gives no positions, so each line gets an even
        slice of the image height (enough for reading order; not used for layout)."""
        text, conf, _ = self.read_text(image)
        lines = [ln for ln in text.splitlines() if ln.strip()]
        H, W = image.shape[:2]
        step = H / max(1, len(lines))
        return OCRResult(words=[Word(text=ln.strip(), conf=conf, line_id=i,
                                     box=Box(x0=0, y0=int(i * step), x1=W, y1=max(int(i * step) + 1, int((i + 1) * step))))
                                for i, ln in enumerate(lines)])

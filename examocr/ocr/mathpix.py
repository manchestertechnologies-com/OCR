"""Mathpix OCR (online, paid, needs an API key): reads every box, ~1-3 s per question.

One request per box the user drew: the crop image -> Mathpix v3/text ->
  text        Mathpix Markdown, formulas as $...$ / $$...$$ (rendered by MathJax here)
  line_data   every line / diagram with its outline (cnt) -> diagram crops
  <smiles>    chemical structures as SMILES (include_smiles) -> Ketcher suggestions
Only the crop is sent, never the page. The key is never logged or sent to the browser.

Key, first found wins:
  1. environment variables MATHPIX_APP_ID / MATHPIX_APP_KEY
  2. Windows user variables set with `setx` (read from the registry, no restart needed)
  3. configs/secrets.yaml  ->  mathpix: {app_id: ..., app_key: ...}   (git-ignored)
"""
from __future__ import annotations

import base64
import logging
import os
import re
import time
from dataclasses import dataclass, field

import cv2
import numpy as np

from ..schemas import Box

log = logging.getLogger(__name__)
API = "https://api.mathpix.com/v3/text"
MAX_B64 = 4_500_000                   # Mathpix's JSON body limit is 5 MB
FIGURE_TYPES = {"diagram", "chart", "chemistry", "chemical_diagram", "figure", "image"}


class MathpixError(Exception):
    """Mathpix could not read the box (no key, network, quota, bad image ...)."""


@dataclass
class MathpixFigure:
    box: Box                          # crop coordinates
    kind: str                         # Mathpix line type: diagram / chart / chemistry ...
    smiles: str | None = None
    text_before: int = 0              # letters of text Mathpix read before this drawing (reading order)


@dataclass
class MathpixResult:
    text: str                         # Mathpix Markdown ($...$ formulas, ![](..) / <smiles> where figures are)
    confidence: float
    figures: list[MathpixFigure] = field(default_factory=list)
    handwritten: bool = False
    seconds: float = 0.0
    request_id: str | None = None
    text_letters: int = 0             # letters of all the text lines (to place figures by text_before)


def letters(s: str | None) -> int:
    """How much text: letters and digits only, so delimiters and spacing don't matter."""
    return len(re.sub(r"[\W_]", "", s or ""))


# ------------------------------------------------------------------ key
def _registry_user_env(name: str) -> str | None:
    """`setx NAME value` writes HKCU\\Environment; a process started before that does not
    see it in os.environ - read it directly (Windows only)."""
    try:
        import winreg
        with winreg.OpenKey(winreg.HKEY_CURRENT_USER, "Environment") as k:
            v, _ = winreg.QueryValueEx(k, name)
            return str(v).strip() or None
    except Exception:
        return None


def credentials() -> tuple[str, str] | None:
    app_id = os.environ.get("MATHPIX_APP_ID") or _registry_user_env("MATHPIX_APP_ID")
    app_key = os.environ.get("MATHPIX_APP_KEY") or _registry_user_env("MATHPIX_APP_KEY")
    if not (app_id and app_key):
        try:
            import yaml

            from ..config import CONFIG_DIR
            p = CONFIG_DIR / "secrets.yaml"
            if p.exists():
                sec = (yaml.safe_load(p.read_text(encoding="utf-8")) or {}).get("mathpix", {}) or {}
                app_id, app_key = app_id or sec.get("app_id"), app_key or sec.get("app_key")
        except Exception:
            pass
    if app_id and app_key:
        return str(app_id).strip(), str(app_key).strip()
    return None


# ------------------------------------------------------------------ request
def encode(img: np.ndarray) -> tuple[str, float]:
    """Image -> (data URI, scale applied). PNG (lossless); a very large crop is scaled down
    (and if still too big, sent as high-quality JPEG) to stay under Mathpix's size limit."""
    scale = 1.0
    if img.ndim == 3:                  # grey: text and formulas need no colour; ~3x smaller upload
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = img.shape[:2]
    if max(h, w) > 3200:
        scale = 3200 / max(h, w)
        img = cv2.resize(img, (round(w * scale), round(h * scale)), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".png", img, [cv2.IMWRITE_PNG_COMPRESSION, 3])
    data, mime = buf.tobytes(), "png"
    quality = 95
    while ok and len(data) * 4 / 3 > MAX_B64:          # rare (a huge, noisy photo): shrink until it fits
        ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, quality])
        data, mime = buf.tobytes(), "jpeg"
        if len(data) * 4 / 3 <= MAX_B64:
            break
        if quality > 80:
            quality -= 5
        else:
            scale *= 0.8
            img = cv2.resize(img, (max(1, round(img.shape[1] * 0.8)), max(1, round(img.shape[0] * 0.8))),
                             interpolation=cv2.INTER_AREA)
    if not ok:
        raise MathpixError("could not encode the image")
    return f"data:image/{mime};base64," + base64.b64encode(data).decode(), scale


def request_body(src: str, smiles: bool = True) -> dict:
    return {
        "src": src,
        "formats": ["text"],
        "math_inline_delimiters": ["$", "$"],
        "math_display_delimiters": ["$$", "$$"],
        "rm_spaces": True,
        "include_line_data": True,
        "include_smiles": smiles,
        "idiomatic_eqn_arrays": True,
    }


def read_image(img: np.ndarray, timeout_s: float = 15.0, smiles: bool = True, _post=None) -> MathpixResult:
    """One crop -> MathpixResult. Raises MathpixError with a short, user-readable reason.
    _post: test hook (url, headers, body, timeout) -> (status, json)."""
    cred = credentials()
    if cred is None:
        raise MathpixError("no Mathpix key set (MATHPIX_APP_ID / MATHPIX_APP_KEY)")
    src, scale = encode(img)
    headers = {"app_id": cred[0], "app_key": cred[1], "Content-type": "application/json"}
    body = request_body(src, smiles)
    t0 = time.perf_counter()
    status, data = (_post or _http_post)(API, headers, body, timeout_s)
    secs = time.perf_counter() - t0
    if status == 401 or status == 403:
        raise MathpixError("the Mathpix key was refused - check MATHPIX_APP_ID / MATHPIX_APP_KEY")
    if status == 429:
        raise MathpixError("Mathpix rate limit reached - try again in a moment")
    if status >= 400 or not isinstance(data, dict):
        raise MathpixError(f"Mathpix error (HTTP {status})")
    if data.get("error"):
        info = data.get("error_info") or {}
        raise MathpixError(f"Mathpix: {info.get('message') or data['error']}")
    return parse_response(data, scale, img.shape[1], img.shape[0], secs)


def _http_post(url: str, headers: dict, body: dict, timeout_s: float) -> tuple[int, dict | None]:
    import httpx
    try:
        r = httpx.post(url, headers=headers, json=body, timeout=timeout_s)
    except httpx.TimeoutException:
        raise MathpixError(f"Mathpix did not answer within {timeout_s:.0f} s") from None
    except httpx.HTTPError as e:
        raise MathpixError(f"cannot reach Mathpix (no internet?): {type(e).__name__}") from None
    try:
        return r.status_code, r.json()
    except ValueError:
        return r.status_code, None


# ------------------------------------------------------------------ response
def parse_response(data: dict, scale: float, width: int, height: int, seconds: float = 0.0) -> MathpixResult:
    """Mathpix JSON -> text + figures (crop coordinates, in reading order) + handwriting flag.
    line_data lists text lines and drawings in reading order: each figure remembers how much
    text came before it, which says where it sits in the text."""
    figures: list[MathpixFigure] = []
    hand = printed = seen = 0
    for ln in data.get("line_data") or []:
        if ln.get("is_handwritten"):
            hand += 1
        elif ln.get("is_printed"):
            printed += 1
        kind = str(ln.get("type") or "").lower()
        if kind not in FIGURE_TYPES:     # drawings come back "included": false (no text for them):
            if ln.get("included") is not False:      # they are still the figures of the box
                seen += letters(ln.get("text"))
            continue
        pts = ln.get("cnt") or []
        if len(pts) < 2:
            continue
        xs = [p[0] / scale for p in pts]
        ys = [p[1] / scale for p in pts]
        b = Box(x0=int(max(0, min(xs))), y0=int(max(0, min(ys))),
                x1=int(min(width, max(xs) + 1)), y1=int(min(height, max(ys) + 1)))
        if b.w >= 6 and b.h >= 6:
            figures.append(MathpixFigure(box=b, kind=kind, text_before=seen))
    conf = data.get("confidence")
    if conf is None:
        conf = data.get("confidence_rate", 0.0)
    return MathpixResult(text=data.get("text") or "", confidence=float(conf or 0.0), figures=figures,
                         handwritten=hand > printed, seconds=round(seconds, 2),
                         request_id=data.get("request_id"), text_letters=seen)


IMAGE_LINK = re.compile(r"!\[[^\]]*\]\([^)]*\)")
SMILES_TAG = re.compile(r"<smiles[^>]*>(.*?)</smiles>", re.S | re.I)
FIG_SLOT = re.compile(r"(!\[[^\]]*\]\([^)]*\)|<smiles[^>]*>.*?</smiles>)", re.S | re.I)


def figure_slots(text: str) -> tuple[str, list[str | None]]:
    """Mathpix marks where each figure sits with an image link (![](url)) or, for a chemical
    structure, <smiles>..</smiles>. -> (text with \\x02k\\x02 placeholders, smiles per slot)."""
    smiles: list[str | None] = []

    def put(m):
        s = SMILES_TAG.match(m.group(0))
        smiles.append(s.group(1).strip() if s else None)
        return f"\x02{len(smiles) - 1}\x02"
    return FIG_SLOT.sub(put, text), smiles

"""Mathpix spending: images sent and their cost, per calendar month, with a monthly limit.

Stored in data/mathpix_usage.json  ->  {"2026-11": {"images": 120, "usd": 0.24}, ...}
When the month's spending reaches mathpix.monthly_limit_usd, Fast papers are read on this
computer instead (never an error, never a surprise bill).
"""
from __future__ import annotations

import json
import os
import threading
import uuid
from datetime import datetime

from .config import settings

_LOCK = threading.Lock()


def _cfg() -> dict:
    return settings().get("mathpix", {}) or {}


def price_per_image() -> float:
    return float(_cfg().get("price_per_image_usd", 0.002))


def limit_usd() -> float:
    return float(_cfg().get("monthly_limit_usd", 0) or 0)        # 0 = no limit


def _path():
    from . import store                  # the same data folder as the projects
    return store.data_dir() / "mathpix_usage.json"


def _load() -> dict:
    try:
        return json.loads(_path().read_text(encoding="utf-8"))
    except Exception:
        return {}


def _month() -> str:
    return datetime.now().strftime("%Y-%m")


def can_spend() -> bool:
    lim = limit_usd()
    if lim <= 0:
        return True
    with _LOCK:
        used = _load().get(_month(), {}).get("usd", 0.0)
    return used + price_per_image() <= lim + 1e-9


def record(images: int = 1) -> None:
    with _LOCK:
        data = _load()
        m = data.setdefault(_month(), {"images": 0, "usd": 0.0})
        m["images"] += images
        m["usd"] = round(m["usd"] + images * price_per_image(), 4)
        tmp = _path().with_suffix(f".{uuid.uuid4().hex[:6]}.tmp")
        tmp.write_text(json.dumps(data, indent=2), encoding="utf-8")
        os.replace(tmp, _path())


def status() -> dict:
    """For the page: is Mathpix set up, and this month's images / cost / limit."""
    from .ocr.mathpix import credentials
    with _LOCK:
        m = _load().get(_month(), {"images": 0, "usd": 0.0})
    inr = float(_cfg().get("inr_per_usd", 85))
    lim = limit_usd()
    return {"configured": credentials() is not None, "month": _month(),
            "images": m["images"], "usd": round(m["usd"], 4), "inr": round(m["usd"] * inr, 2),
            "limit_usd": lim, "limit_inr": round(lim * inr, 2),
            "price_inr": round(price_per_image() * inr, 3),
            "limit_reached": lim > 0 and m["usd"] + price_per_image() > lim + 1e-9}

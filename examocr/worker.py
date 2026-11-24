"""Background reading: drawing a box answers in milliseconds; the OCR runs here.

One worker thread, jobs in the order the boxes were drawn: the OCR model fits in this
laptop's RAM only once, so reading two boxes at the same time would not be faster.
A job reads WITHOUT holding the project lock (the reviewer keeps cropping and editing) and
takes the lock only to store its result - see store.complete_part / store.complete_crop.
"""
from __future__ import annotations

import logging
import queue
import threading
from collections import Counter
from typing import Callable

log = logging.getLogger(__name__)
_JOBS: "queue.Queue[tuple[str, str, Callable[[], None]]]" = queue.Queue()
_PENDING: Counter = Counter()          # project id -> jobs waiting or running
_QUEUED: set[str] = set()              # job keys, so a restart never queues one twice
_LOCK = threading.Lock()
_THREAD: threading.Thread | None = None


def submit(pid: str, key: str, fn: Callable[[], None]) -> None:
    """Queue fn (a reading job for project pid). key identifies the job (e.g. the crop id)."""
    global _THREAD
    with _LOCK:
        if key in _QUEUED:
            return
        _QUEUED.add(key)
        _PENDING[pid] += 1
        if _THREAD is None or not _THREAD.is_alive():
            _THREAD = threading.Thread(target=_run, daemon=True, name="reader")
            _THREAD.start()
    _JOBS.put((pid, key, fn))


def pending(pid: str) -> int:
    with _LOCK:
        return _PENDING.get(pid, 0)


def _run() -> None:
    while True:
        pid, key, fn = _JOBS.get()
        try:
            fn()
        except Exception:                 # one bad box never stops the queue
            log.exception("reading job %s failed", key)
        finally:
            with _LOCK:
                _QUEUED.discard(key)
                _PENDING[pid] -= 1
                if _PENDING[pid] <= 0:
                    del _PENDING[pid]
            _JOBS.task_done()


def wait_idle(timeout: float | None = None) -> bool:
    """Tests / CLI: block until every queued job is done."""
    import time
    end = None if timeout is None else time.time() + timeout
    while True:
        with _LOCK:
            if not _QUEUED:
                return True
        if end is not None and time.time() > end:
            return False
        time.sleep(0.05)

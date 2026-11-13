"""Tiny disk cache for search results (Day 25). Off unless SEARCH_CACHE=true, so tests and
normal runs never read stale data by accident. Same query twice = no API call."""
import hashlib
import json
import os
import threading
import time
from pathlib import Path

CACHE_PATH = Path("data/cache/search_cache.json")
TTL_SECONDS = 24 * 3600

_lock = threading.Lock()
_stats = {"hits": 0, "misses": 0}


def enabled() -> bool:
    return os.getenv("SEARCH_CACHE", "false").lower() == "true"


def _key(kind: str, *parts) -> str:
    raw = json.dumps([kind, *[str(p).strip().lower() for p in parts]])
    return hashlib.sha1(raw.encode()).hexdigest()


def _load() -> dict:
    try:
        return json.loads(CACHE_PATH.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def get(kind: str, *parts, ttl: int = TTL_SECONDS):
    if not enabled():
        return None
    with _lock:
        entry = _load().get(_key(kind, *parts))
        if entry and time.time() - entry["t"] <= ttl:
            _stats["hits"] += 1
            return entry["v"]
        _stats["misses"] += 1
        return None


def put(kind: str, *parts, value) -> None:
    if not enabled():
        return
    with _lock:
        data = _load()
        data[_key(kind, *parts)] = {"t": time.time(), "v": value}
        CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        tmp = CACHE_PATH.with_suffix(".tmp")
        tmp.write_text(json.dumps(data), encoding="utf-8")
        os.replace(tmp, CACHE_PATH)


def stats() -> dict:
    return dict(_stats)

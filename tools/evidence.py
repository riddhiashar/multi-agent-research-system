from typing import Dict, List
from urllib.parse import urlparse

MIN_INDEPENDENT_SOURCES = 2

# Fine to read, but not counted as independent corroboration.
LOW_QUALITY_DOMAINS = {
    "youtube.com", "youtu.be", "facebook.com", "tiktok.com", "pinterest.com",
    "instagram.com", "x.com", "twitter.com", "quora.com",
}

_TWO_PART_SUFFIXES = {"co.uk", "com.au", "co.in", "com.cn", "co.jp", "com.br", "co.kr"}


def _registrable(host: str) -> str:
    host = host.lower().split(":")[0]
    if host.startswith("www."):
        host = host[4:]
    parts = host.split(".")
    if len(parts) >= 3 and ".".join(parts[-2:]) in _TWO_PART_SUFFIXES:
        return ".".join(parts[-3:])
    if len(parts) >= 2:
        return ".".join(parts[-2:])
    return host


def source_domain(source: Dict) -> str:
    url = source.get("url", "") or ""
    host = urlparse(url).hostname or ""
    # Gemini grounding links redirect through a Google domain; the real site is in the title.
    if host.endswith("vertexaisearch.cloud.google.com"):
        title = (source.get("title") or "").strip().lower()
        if "." in title and " " not in title:
            return _registrable(title)
    return _registrable(host) if host else ""


def independent_domains(sources: List[Dict]) -> set:
    domains = {source_domain(s) for s in sources}
    domains.discard("")
    return domains - LOW_QUALITY_DOMAINS


def independent_source_count(sources: List[Dict]) -> int:
    return len(independent_domains(sources))

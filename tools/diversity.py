"""Source diversity helpers (Day 24): prefer DIFFERENT domains over MORE results."""
from collections import Counter
from typing import Dict, List

from tools.evidence import source_domain, LOW_QUALITY_DOMAINS

MAX_SOURCES_PER_QUERY = 4
MIN_DIVERSE_DOMAINS = 3     


def _domain(r: Dict) -> str:
    return source_domain({"url": r.get("url", ""), "title": r.get("title", "")})


def select_diverse(results: List[Dict], max_total: int = MAX_SOURCES_PER_QUERY,
                   max_per_domain: int = 1) -> List[Dict]:
    """results: ranked best-first, each {"title","url","content"}. Keeps at most
    `max_per_domain` per registrable domain and skips low-quality domains. If that
    leaves nothing, falls back to the top results so the agent still has text."""
    kept, per = [], Counter()
    for r in results:
        d = _domain(r)
        if not d or d in LOW_QUALITY_DOMAINS or per[d] >= max_per_domain:
            continue
        per[d] += 1
        kept.append(r)
        if len(kept) >= max_total:
            break
    return kept or results[:max_total]


def domain_count(results: List[Dict]) -> int:
    return len({d for d in map(_domain, results) if d and d not in LOW_QUALITY_DOMAINS})


def add_new_domains(primary: List[Dict], extra: List[Dict],
                    max_total: int = MAX_SOURCES_PER_QUERY) -> List[Dict]:
    """Top up `primary` with results from `extra` ONLY if their domain is new."""
    out = list(primary)
    seen = {_domain(r) for r in out}
    for r in extra:
        d = _domain(r)
        if len(out) >= max_total:
            break
        if d and d not in seen and d not in LOW_QUALITY_DOMAINS:
            seen.add(d)
            out.append(r)
    return out

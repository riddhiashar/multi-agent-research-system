"""Merge equivalent claims across sub-questions BEFORE fact-checking (Day 24).

Why: the same fact often appears under several sub-questions, each time backed by a
single snippet. Checked separately, each copy has 1 source. Merged, the copies pool
their sources, so real corroboration becomes visible. No LLM call is used (Groq's
output-token budget is the bottleneck), only deterministic text rules.
"""
import re
from typing import Dict, List

STOPWORDS = {
    "the", "a", "an", "of", "in", "on", "at", "to", "for", "and", "or", "is", "are", "was",
    "were", "be", "been", "by", "with", "as", "that", "this", "it", "its", "from", "has",
    "have", "had", "which", "their", "also", "than", "then", "into", "over", "about",
}
NEGATIONS = {"not", "no", "never", "without", "cannot", "none", "neither", "nor"}
MIN_CONTENT_TOKENS = 4      # shorter claims are never merged (too little to compare safely)
JACCARD_THRESHOLD = 0.6
CONTAINMENT_THRESHOLD = 0.85


def _stem(w: str) -> str:
    return w[:-1] if len(w) > 3 and w.endswith("s") and not w.endswith("ss") else w


def _profile(claim: str) -> Dict:
    text = claim.lower().replace(",", "")
    words = re.findall(r"\d+(?:\.\d+)?|[a-z]+", text)
    numbers = frozenset(w for w in words if w[0].isdigit())
    negated = any(w in NEGATIONS for w in words)
    tokens = frozenset(_stem(w) for w in words
                       if not w[0].isdigit() and w not in STOPWORDS and w not in NEGATIONS)
    return {"numbers": numbers, "negated": negated, "tokens": tokens}


def claims_equivalent(a: str, b: str) -> bool:
    pa, pb = _profile(a), _profile(b)
    if pa["numbers"] != pb["numbers"] or pa["negated"] != pb["negated"]:
        return False   # "95%" vs "70%", or a claim and its negation, are NOT the same claim
    ta, tb = pa["tokens"], pb["tokens"]
    if min(len(ta), len(tb)) < MIN_CONTENT_TOKENS:
        return False
    inter = len(ta & tb)
    return (inter / len(ta | tb) >= JACCARD_THRESHOLD
            or inter / min(len(ta), len(tb)) >= CONTAINMENT_THRESHOLD)


def _pool_sources(members: List[Dict]) -> List[Dict]:
    seen, pooled = set(), []
    for m in members:
        for s in m.get("sources", []) or []:
            key = s.get("url") or s.get("title")
            if key and key not in seen:
                seen.add(key)
                pooled.append({**s, "id": len(pooled) + 1})
    return pooled


def merge_duplicate_claims(facts: List[Dict]) -> List[Dict]:
    """Greedy clustering. Singletons pass through untouched; merged facts carry
    `origins` (one entry per sub_question) and `merged_count` so verdicts can be
    expanded back afterwards."""
    clusters: List[List[Dict]] = []
    for fact in facts:
        for cluster in clusters:
            if claims_equivalent(cluster[0]["claim"], fact["claim"]):
                cluster.append(fact)
                break
        else:
            clusters.append([fact])

    merged = []
    for members in clusters:
        if len(members) == 1:
            merged.append(members[0])
            continue
        origins, seen_sq = [], set()
        for m in members:
            sq = m.get("sub_question")
            if sq not in seen_sq:
                seen_sq.add(sq)
                origins.append({"sub_question": sq, "claim": m["claim"]})
        best = max(members, key=lambda m: len(m["claim"]))
        merged.append({**members[0], "claim": best["claim"],
                       "sources": _pool_sources(members),
                       "origins": origins, "merged_count": len(members)})
    return merged


def expand_verdicts(results: List[Dict]) -> List[Dict]:
    """After fact-checking: give every originating sub-question its own copy of the
    shared verdict, so summarization still sees each planned topic's claims."""
    expanded = []
    for r in results:
        origins = r.get("origins")
        if not origins:
            expanded.append(r)
            continue
        base = {k: v for k, v in r.items() if k != "origins"}
        for o in origins:
            copy = {**base, "claim": o["claim"]}
            if o["sub_question"] is not None:
                copy["sub_question"] = o["sub_question"]
            expanded.append(copy)
    return expanded

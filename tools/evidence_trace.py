"""Evidence traceability (Day 27). Pure functions: no LLM, no network, no Streamlit.

For every claim this answers, from real run data: which sources back it, which exact sentence in each,
whether the source came from the original search or from targeted verification, and WHY the claim got
its verdict (confirmed / disputed / unverified-because-too-few-domains / inconclusive / failed)."""
import re
from typing import Dict, List, Optional, Tuple

from tools import claim_verifier
from tools.claim_merge import _profile
from tools.evidence import MIN_INDEPENDENT_SOURCES, independent_domains, independent_source_count, source_domain
from tools.evidence_check import snippet_records
from tools.verification import _sentences

FAILED_MARKERS = ("call failed", "not returned by model")   # strings written by the fact-checker's fail-safes
CLOSEST_MIN_OVERLAP = 0.3
SHARED_ORIGIN_OVERLAP = 0.8

CATEGORY_LABEL = {
    "confirmed": "Confirmed",
    "disputed": "Disputed",
    "insufficient_sources": "Unverified: too few independent sources",
    "citations_not_supporting": "Unverified: cited sources do not state the claim",
    "inconclusive": "Unverified: evidence inconclusive",
    "failed": "Unverified: research or verification failed",
}
CATEGORY_ORDER = ["confirmed", "disputed", "insufficient_sources", "citations_not_supporting", "inconclusive", "failed"]


def snippet_pool(search_results: List[Dict]) -> Dict[str, List[Dict]]:
    """url -> [{"title","text"}] for every snippet the search step stored."""
    pool: Dict[str, List[Dict]] = {}
    for r in snippet_records(search_results):
        pool.setdefault(r["url"], []).append({"title": r["title"], "text": r["text"]})
    return pool


def trace_source(claim: str, source: Dict, pool: Dict[str, List[Dict]], entities=()) -> Dict:
    """One source with its domain, origin and the sentence checked against the claim.
    match: 'exact' = the sentence states the claim (figures, units, subject, period all agree)
           'contradicts' = the sentence says something incompatible
           'closest' = related but not conclusive (shown for context only)
           'none' = nothing relevant in the snippet; 'unavailable' = snippet text was not saved (old runs)."""
    url = source.get("url", "") or ""
    origin = "verification" if source.get("via") == "verification" else "search"
    evidence, match, reasons = source.get("evidence"), "none", list(source.get("reasons") or [])
    rel = source.get("relation")
    if evidence and rel in (None, "supports"):
        match = "exact"
    elif evidence and rel:
        match = {"contradicts": "contradicts", "uncertain": "closest"}.get(rel, "closest")
    if not evidence:
        recs = pool.get(url, [])
        if not recs:
            match = "unavailable"
        else:
            best = None
            for rec in recs:
                a = claim_verifier.assess_text(claim, rec["text"], rec["title"], entities)
                cand = a["supports"] or a["contradicts"] or a["closest"]
                if cand and (best is None or (cand["relation"] == "supports") > (best["relation"] == "supports")):
                    best = cand
            if best:
                evidence, reasons = best["sentence"].strip(), best["reasons"]
                match = {"supports": "exact", "contradicts": "contradicts"}.get(best["relation"], "closest")
    domain = source_domain(source)
    return {"title": source.get("title") or url or "source", "url": url, "domain": domain, "origin": origin,
            "evidence": (evidence or "")[:400], "match": match, "reasons": reasons,
            "method": source.get("method", "rules"), "independent": domain in independent_domains([source])}


def possible_shared_origin(traced: List[Dict]) -> bool:
    """True when two sources' evidence sentences are near-identical: the domains differ but the wording
    suggests one upstream statement repeated (syndication or citation). A warning, not a verdict change."""
    sents = [(t["domain"], {w for w in _profile(t["evidence"])["tokens"]}) for t in traced
             if t["match"] == "exact" and t["evidence"]]
    for i in range(len(sents)):
        for j in range(i + 1, len(sents)):
            (da, a), (db, b) = sents[i], sents[j]
            if da != db and a and b and len(a & b) / len(a | b) >= SHARED_ORIGIN_OVERLAP:
                return True
    return False


def verdict_category(fact: Dict) -> str:
    status = (fact.get("status") or "UNVERIFIED").upper()
    if status == "CONFIRMED":
        return "confirmed"
    if status == "DISPUTED":
        return "disputed"
    reasoning = (fact.get("reasoning") or "").lower()
    if any(m in reasoning for m in FAILED_MARKERS):
        return "failed"
    n = fact.get("independent_sources")
    if n is None:
        n = independent_source_count(fact.get("sources") or [])
    if n < MIN_INDEPENDENT_SOURCES:
        return "failed" if fact.get("verification") == "search_failed" else "insufficient_sources"
    sup = fact.get("supported_domains")
    checked = [c for c in fact.get("citation_checks") or [] if c.get("relation") != "unavailable"]
    if sup is not None and sup < MIN_INDEPENDENT_SOURCES and checked:
        return "citations_not_supporting"
    return "inconclusive"


def verdict_explanation(fact: Dict) -> str:
    cat = verdict_category(fact)
    n = fact.get("independent_sources")
    if n is None:
        n = independent_source_count(fact.get("sources") or [])
    reasoning = (fact.get("reasoning") or "").strip()
    if cat == "confirmed":
        return f"{n} independent domains back this claim and nothing in its batch contradicted it."
    if cat == "disputed":
        return f"The fact-checker found another claim in the same batch that contradicts this one. {reasoning}".strip()
    if cat == "failed":
        if fact.get("verification") == "search_failed":
            return "The verification search for a second source errored, so this claim was not properly checked."
        return "The fact-check step failed or returned nothing for this claim, so it defaulted to unverified."
    if cat == "citations_not_supporting":
        sup = fact.get("supported_domains", 0)
        return (f"{n} domains are cited, but only {sup} have a sentence that states the claim with the same figures, "
                f"subject and period, so the citations do not support it. {reasoning}").strip()
    if cat == "inconclusive":
        return (f"It has {n} independent domains, but the fact-checker did not confirm it. {reasoning}").strip()
    # insufficient
    how = {"no_match": "A targeted search was run but no second independent source stated the same figures.",
           "added": "A targeted search added a source, but still fewer than two independent domains."}.get(
        fact.get("verification"), "No targeted search was run for it (the per-run verification budget covers only some claims).")
    return f"Only {n} independent domain(s) found; {MIN_INDEPENDENT_SOURCES} are required. {how}"


def trace_claim(fact: Dict, pool: Dict[str, List[Dict]], entities=()) -> Dict:
    traced = [trace_source(fact.get("claim", ""), s, pool, entities) for s in fact.get("sources") or []]
    cat = verdict_category(fact)
    return {
        "claim": fact.get("claim", ""), "status": fact.get("status", "UNVERIFIED"),
        "topic": fact.get("sub_question", ""), "category": cat, "category_label": CATEGORY_LABEL[cat],
        "explanation": verdict_explanation(fact),
        "independent_domains": fact.get("independent_sources", independent_source_count(fact.get("sources") or [])),
        "original_sources": sum(1 for t in traced if t["origin"] == "search"),
        "verification_sources": sum(1 for t in traced if t["origin"] == "verification"),
        "verification": fact.get("verification", ""), "verified_by_search": bool(fact.get("verified_by_search")),
        "downgraded": bool(fact.get("downgraded")), "merged_count": fact.get("merged_count", 1),
        "shared_origin_warning": possible_shared_origin(traced), "sources": traced,
        "supported_domains": fact.get("supported_domains"),
        "conflicts": list(fact.get("conflicting_evidence") or []),
    }


def trace_all(results: List[Dict], search_results: List[Dict], query: str = "") -> List[Dict]:
    pool, entities = snippet_pool(search_results), claim_verifier.query_entities(query)
    return [trace_claim(r, pool, entities) for r in results or []]


def category_counts(results: List[Dict]) -> Dict[str, int]:
    counts = {c: 0 for c in CATEGORY_ORDER}
    for r in results or []:
        counts[verdict_category(r)] += 1
    return counts


def _initial_domains(fact: Dict) -> int:
    return len(independent_domains([s for s in fact.get("sources") or [] if s.get("via") != "verification"]))


def claim_funnel(extracted: int, results: List[Dict], usage_log: List[Dict]) -> Dict:
    """Numbers that explain 'extracted vs checked' and what verification did. Everything is computed from
    run data. 'Distinct' collapses the per-sub-topic copies of one merged claim."""
    distinct = {}
    for r in results or []:
        distinct.setdefault(r.get("claim", ""), r)
    fc = next((u for u in (usage_log or []) if u.get("agent") == "fact_checking_agent" and "facts_checked" in u), {})
    return {
        "extracted": extracted, "distinct": len(distinct), "rows": len(results or []),
        "merged_away": max(0, extracted - len(results or [])) if extracted else 0,
        "initially_under_supported": sum(1 for r in distinct.values() if _initial_domains(r) < MIN_INDEPENDENT_SOURCES),
        "verification_tried": fc.get("verify_tried", 0), "verification_upgraded": fc.get("verify_upgraded", 0),
        "verification_sources_added": fc.get("verify_sources_added", 0),
        "verification_failed": fc.get("verify_failed", 0),
        "verification_recorded": "verify_tried" in fc,
    }

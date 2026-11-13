"""Run-level evidence checks on the claims, before the LLM fact-check (Day 27). Pure, no network.

1. Citation check: does the sentence in each CITED source actually state the claim (figures + units, subject,
   polarity, direction, period)? Gives each claim `supported_domains` next to `independent_sources`
   (cited domains). A cited source that does not state the claim is a wrong citation, not evidence.
2. Conflict scan: do sentences in ANY source retrieved during the run contradict the claim? This catches
   contradictions across sub-topics, which the LLM never sees because it checks claims in batches."""
import re
from typing import Dict, Iterable, List

from tools import claim_verifier
from tools.evidence import independent_domains, source_domain

MAX_CONFLICTS = 3


def snippet_records(search_results: List[Dict]) -> List[Dict]:
    """[{"url","title","text"}] rebuilt from the tagged '[S1] ... [S2] ...' answers the search step stored."""
    out = []
    for r in search_results or []:
        if not isinstance(r, dict):
            continue
        parts = re.split(r"\[S(\d+)\]", r.get("answer") or "")
        texts = {int(parts[i]): parts[i + 1].strip() for i in range(1, len(parts) - 1, 2)}
        for s in r.get("sources") or []:
            if s.get("id") in texts and s.get("url"):
                out.append({"url": s["url"], "title": s.get("title", ""), "text": texts[s["id"]]})
    return out


def check_citations(fact: Dict, records: List[Dict], entities: Iterable[str] = ()) -> Dict:
    """Adds relation/evidence to every cited source plus supported_domains and citation_checks to the fact."""
    by_url: Dict[str, List[Dict]] = {}
    for r in records:
        by_url.setdefault(r["url"], []).append(r)
    sources, checks, supported = [], [], []
    for src in fact.get("sources") or []:
        src = dict(src)
        if src.get("via") == "verification" and src.get("evidence"):
            rel = src.get("relation", "supports")
        elif src.get("url") in by_url:
            best = None
            for rec in by_url[src["url"]]:
                a = claim_verifier.assess_text(fact["claim"], rec["text"], rec["title"], entities)
                cand = a["supports"] or a["contradicts"] or a["closest"]
                if cand and (best is None or {"supports": 3, "contradicts": 2}.get(cand["relation"], 1) >
                             {"supports": 3, "contradicts": 2}.get(best["relation"], 1)):
                    best = cand
            rel = best["relation"] if best else "unrelated"
            if best:
                src.update({"evidence": best["sentence"][:300], "relation": rel, "reasons": best["reasons"]})
        else:
            rel = "unavailable"
        src.setdefault("relation", rel)
        sources.append(src)
        checks.append({"url": src.get("url", ""), "domain": source_domain(src), "relation": rel})
        if rel == "supports":
            supported.append(src)
    return {**fact, "sources": sources, "citation_checks": checks,
            "supported_domains": len(independent_domains(supported))}


def find_conflicts(fact: Dict, records: List[Dict], entities: Iterable[str] = ()) -> Dict:
    cited = {s.get("url") for s in fact.get("sources") or []}
    conflicts = list(fact.get("conflicting_evidence") or [])
    seen = {c["domain"] for c in conflicts}
    for rec in records:
        if len(conflicts) >= MAX_CONFLICTS:
            break
        domain = source_domain(rec)
        if domain in seen or not domain:
            continue
        hit = claim_verifier.assess_text(fact["claim"], rec["text"], rec["title"], entities)["contradicts"]
        if hit:
            seen.add(domain)
            conflicts.append({"title": rec["title"], "url": rec["url"], "domain": domain, "evidence": hit["sentence"][:300],
                              "reason": "; ".join(hit["reasons"]), "method": "rules",
                              "cited_by_claim": rec["url"] in cited})
    return {**fact, "conflicting_evidence": conflicts} if conflicts else fact


def attach_citation_checks(facts: List[Dict], records: List[Dict], entities: Iterable[str] = ()) -> List[Dict]:
    return [check_citations(f, records, entities) for f in facts]


def attach_conflicts(facts: List[Dict], records: List[Dict], entities: Iterable[str] = ()) -> List[Dict]:
    return [find_conflicts(f, records, entities) for f in facts]


def citation_stats(facts: List[Dict]) -> Dict:
    checks = [c for f in facts for c in f.get("citation_checks") or []]
    judged = [c for c in checks if c["relation"] != "unavailable"]
    return {"cited": len(checks), "checkable": len(judged),
            "supporting": sum(1 for c in judged if c["relation"] == "supports"),
            "contradicting": sum(1 for c in judged if c["relation"] == "contradicts"),
            "claims_with_conflicts": sum(1 for f in facts if f.get("conflicting_evidence"))}

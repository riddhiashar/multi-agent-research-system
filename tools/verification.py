"""Targeted corroboration (Day 25, upgraded Day 27).

For claims backed by fewer than the required number of independent domains, search for the claim itself,
excluding the domain(s) already cited. A new source is accepted ONLY if one of its sentences states the claim:
the same figures WITH units, the right subject, the same polarity and direction, a compatible time period
(tools/claim_verifier.py). Optionally an LLM judge may rescue clear paraphrases (tools/semantic_judge.py), but
only for pairs whose deterministic guards passed. Sentences that CONTRADICT the claim are recorded as
conflicting evidence instead of being thrown away."""
import os
from typing import Callable, Dict, List, Optional, Tuple

from tools import claim_verifier
from tools.claim_merge import _profile
from tools.evidence import independent_domains, independent_source_count, MIN_INDEPENDENT_SOURCES, source_domain

MAX_VERIFY_CLAIMS = 6            # per run, keeps search volume inside free-tier limits
MAX_CONFLICTS = 3


def enabled() -> bool:
    return os.getenv("VERIFY_BY_SEARCH", "false").lower() == "true"


def _sentences(text: str) -> List[str]:
    return claim_verifier.split_sentences(text)


def matching_sentence(claim: str, text: str, title: str = "", entities=()) -> Optional[str]:
    """The best sentence of `text` that states the claim, or None."""
    hit = claim_verifier.assess_text(claim, text, title, entities)["supports"]
    return hit["sentence"] if hit else None


def supports(claim: str, text: str) -> bool:
    return matching_sentence(claim, text) is not None


def verify_claim(fact: Dict, search_fn: Callable, entities=(), judge_fn: Optional[Callable] = None) -> Tuple[Dict, int, bool]:
    """Returns (fact, sources_added, search_failed). fact["verification"] records what happened:
    "added" | "no_match" | "search_failed". Contradicting sentences go to fact["conflicting_evidence"]."""
    have = independent_domains(fact.get("sources", []))
    try:
        candidates = search_fn(fact["claim"], exclude_domains=sorted(have))
    except Exception:
        return {**fact, "verification": "search_failed"}, 0, True

    sources, added = list(fact.get("sources", [])), 0
    conflicts = list(fact.get("conflicting_evidence") or [])
    pending = []   # judgeable-uncertain sentences, for one optional LLM call
    for c in candidates or []:
        domain = source_domain(c)
        new_domain = independent_domains([c]) - have
        analysis = claim_verifier.assess_text(fact["claim"], c.get("content", ""), c.get("title", ""), entities)
        if analysis["contradicts"] and len(conflicts) < MAX_CONFLICTS and domain not in {x["domain"] for x in conflicts}:
            a = analysis["contradicts"]
            conflicts.append({"title": c.get("title", ""), "url": c.get("url", ""), "domain": domain,
                              "evidence": a["sentence"][:300], "reason": "; ".join(a["reasons"]), "method": "rules"})
        if not new_domain:
            continue
        if analysis["supports"]:
            a = analysis["supports"]
            have |= new_domain
            sources.append({"id": len(sources) + 1, "title": c.get("title", ""), "url": c.get("url", ""),
                            "via": "verification", "evidence": a["sentence"].strip()[:300], "relation": "supports",
                            "method": "rules", "reasons": a["reasons"]})
            added += 1
        else:
            pending += [(c, a) for a in analysis["uncertain"][:1]]

    if judge_fn and pending:
        verdicts = judge_fn(fact["claim"], [a["sentence"] for _, a in pending])
        for (c, a), v in zip(pending, verdicts or []):
            new_domain = independent_domains([c]) - have
            if v["relation"] == "supports" and new_domain:      # guards already passed (judgeable)
                have |= new_domain
                sources.append({"id": len(sources) + 1, "title": c.get("title", ""), "url": c.get("url", ""),
                                "via": "verification", "evidence": a["sentence"].strip()[:300], "relation": "supports",
                                "method": "llm", "reasons": [f"LLM judge: {v['why']}"]})
                added += 1
            elif v["relation"] == "contradicts" and len(conflicts) < MAX_CONFLICTS:
                conflicts.append({"title": c.get("title", ""), "url": c.get("url", ""), "domain": source_domain(c),
                                  "evidence": a["sentence"][:300], "reason": f"LLM judge: {v['why']}", "method": "llm"})

    out = {**fact}
    if conflicts:
        out["conflicting_evidence"] = conflicts
    if not added:
        return {**out, "verification": "no_match"}, 0, False
    return {**out, "sources": sources, "verified_by_search": added, "verification": "added"}, added, False


def _priority_order(facts: List[Dict], needy: List[int]) -> List[int]:
    """Round-robin across sub-questions so one topic can't use up the whole verification budget;
    inside each topic, claims with figures go first."""
    groups: Dict = {}
    for i in needy:
        groups.setdefault(facts[i].get("sub_question"), []).append(i)
    queues = []
    for idxs in groups.values():
        idxs.sort(key=lambda i: (not _profile(facts[i]["claim"])["numbers"], i))
        queues.append(idxs)
    order = []
    while any(queues):
        for q in queues:
            if q:
                order.append(q.pop(0))
    return order


def verify_claims(facts: List[Dict], search_fn: Callable, max_claims: int = MAX_VERIFY_CLAIMS,
                  entities=(), judge_fn: Optional[Callable] = None) -> Tuple[List[Dict], Dict]:
    """Verify the most checkable under-sourced claims (those with figures first), sequentially."""
    needy = [i for i, f in enumerate(facts)
             if independent_source_count(f.get("sources", [])) < MIN_INDEPENDENT_SOURCES]
    needy = _priority_order(facts, needy)

    out = list(facts)
    stats = {"tried": 0, "upgraded": 0, "sources_added": 0, "failed": 0, "conflicts": 0, "judged": 0}
    for i in needy[:max_claims]:
        stats["tried"] += 1
        new_fact, added, failed = verify_claim(out[i], search_fn, entities, judge_fn)
        stats["failed"] += int(failed)
        stats["sources_added"] += added
        stats["conflicts"] += len(new_fact.get("conflicting_evidence") or []) - len(out[i].get("conflicting_evidence") or [])
        stats["judged"] += sum(1 for s in new_fact.get("sources", []) if s.get("method") == "llm") - \
                           sum(1 for s in out[i].get("sources", []) if s.get("method") == "llm")
        if independent_source_count(new_fact.get("sources", [])) >= MIN_INDEPENDENT_SOURCES:
            stats["upgraded"] += 1
        out[i] = new_fact
    return out, stats

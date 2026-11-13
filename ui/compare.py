"""Compare two saved runs (Day 27): what is new, what changed verdict, what gained or lost evidence."""
from typing import Dict, List

from tools.claim_merge import claims_equivalent
from tools.evidence_trace import verdict_category
from ui.runstats import run_stats


def _distinct(results: List[Dict]) -> List[Dict]:
    seen, out = set(), []
    for r in results or []:
        key = " ".join((r.get("claim") or "").lower().split())
        if key and key not in seen:
            seen.add(key)
            out.append(r)
    return out


def _find(claim: str, pool: List[Dict]):
    key = " ".join(claim.lower().split())
    for r in pool:
        if " ".join((r.get("claim") or "").lower().split()) == key:
            return r
    for r in pool:
        if claims_equivalent(claim, r.get("claim", "")):
            return r
    return None


def _n(r: Dict) -> int:
    return r.get("independent_sources", 0)


def compare_runs(a: Dict, b: Dict) -> Dict:
    """a = older run, b = newer run (any order works; labels say 'run A'/'run B')."""
    ra, rb = _distinct(a.get("fact_check_results")), _distinct(b.get("fact_check_results"))
    new, changed, gained, lost, same = [], [], [], [], 0
    matched_a = set()
    for r in rb:
        m = _find(r["claim"], ra)
        if m is None:
            new.append(r)
            continue
        matched_a.add(id(m))
        if m.get("status") != r.get("status"):
            changed.append({"claim": r["claim"], "from": m.get("status"), "to": r.get("status"),
                            "from_category": verdict_category(m), "to_category": verdict_category(r)})
        elif _n(r) > _n(m):
            gained.append({"claim": r["claim"], "from": _n(m), "to": _n(r)})
        elif _n(r) < _n(m):
            lost.append({"claim": r["claim"], "from": _n(m), "to": _n(r)})
        else:
            same += 1
    dropped = [r for r in ra if id(r) not in matched_a]
    urls = lambda s: {x.get("url") for r in s.get("fact_check_results") or [] for x in r.get("sources") or [] if x.get("url")}
    ua, ub = urls(a), urls(b)
    sa, sb = run_stats(a), run_stats(b)
    conf = lambda rows: sum(1 for r in rows if r.get("status") == "CONFIRMED")
    table = [("Distinct claims", len(ra), len(rb)), ("Confirmed claims", conf(ra), conf(rb)),
             ("Sources cited", len(ua), len(ub)), ("LLM calls", sa["llm_calls"], sb["llm_calls"]),
             ("Run time (s)", round(sa["wall_seconds"] or 0), round(sb["wall_seconds"] or 0))]
    return {"same_question": (a.get("query", "").strip().lower() == b.get("query", "").strip().lower()),
            "new_claims": new, "dropped_claims": dropped, "status_changed": changed,
            "gained_evidence": gained, "lost_evidence": lost, "unchanged": same,
            "new_sources": sorted(ub - ua), "dropped_sources": sorted(ua - ub),
            "table": [{"metric": m, "run_a": x, "run_b": y, "change": y - x} for m, x, y in table]}

"""Run-level numbers for the Analytics tab and the health banner (Day 27). Reads a saved snapshot, reuses the
backend metrics already in it, and never invents a figure. Pure and unit-tested."""
from typing import Dict, List

from tools.evidence_trace import FAILED_MARKERS, trace_all, verdict_category
from ui import views


def _usage(snap: Dict) -> Dict:
    return next((u for u in snap.get("usage_log") or []
                 if u.get("agent") == "fact_checking_agent" and "facts_checked" in u), {})


def _failed_summaries(snap: Dict) -> List[str]:
    return [s.get("sub_question", "") for s in snap.get("summaries") or [] if s.get("status") == "failed"]


def _failed_fact_rows(snap: Dict) -> int:
    return sum(1 for r in snap.get("fact_check_results") or [] if verdict_category(r) == "failed"
               and any(m in (r.get("reasoning") or "").lower() for m in FAILED_MARKERS))


def run_health(snap: Dict) -> Dict:
    """complete / partial / failed, with the reasons. A run with failed stages is never shown as a success."""
    issues = []
    if not snap.get("final_report"):
        return {"state": "failed", "issues": ["No report was produced."]}
    fs = _failed_summaries(snap)
    if fs:
        issues.append(f"{len(fs)} sub-topic summary(ies) failed.")
    fr = _failed_fact_rows(snap)
    if fr:
        issues.append(f"{fr} claim(s) could not be fact-checked and defaulted to unverified.")
    vf = _usage(snap).get("verify_failed", 0)
    stuck = sum(1 for r in snap.get("fact_check_results") or [] if r.get("verification") == "search_failed")
    if vf or stuck:
        issues.append(f"{stuck or vf} verification search(es) errored.")
    se = snap.get("state_errors") or []
    if se:
        issues.append(f"{len(se)} agent error(s) were recorded.")
    if not snap.get("fact_check_results"):
        issues.append("No claims were extracted.")
    return {"state": "partial" if issues else "complete", "issues": issues}


def run_stats(snap: Dict) -> Dict:
    m = snap.get("metrics") or {}
    nodes = m.get("nodes") or {}
    cache = snap.get("cache") or {}
    usage = _usage(snap)
    results = snap.get("fact_check_results") or []
    searched = {s.get("url") for r in snap.get("search_results") or [] if isinstance(r, dict)
                for s in r.get("sources") or [] if s.get("url")}
    cited = {s.get("url") for r in results for s in r.get("sources") or [] if s.get("url")}
    return {
        "llm_calls": sum(n.get("calls", 0) for n in nodes.values()),
        "failed_llm_calls": sum(n.get("rate_limited", 0) + n.get("other_errors", 0) for n in nodes.values()),
        "rate_limited_calls": m.get("rate_limit_errors", 0),
        "input_tokens": sum(n.get("input", 0) for n in nodes.values()),
        "output_tokens": m.get("total_output_tokens", 0),
        "peak_output_60s": m.get("peak_output_tokens_60s", 0),
        "search_requests": len(snap.get("search_results") or []),
        "verification_searches": usage.get("verify_tried", 0),
        "cache_hits": cache.get("hits", 0), "cache_misses": cache.get("misses", 0),
        "sources_retrieved": len(searched | cited), "sources_cited": len(cited),
        "unrecovered_failures": len(_failed_summaries(snap)) + _failed_fact_rows(snap)
                                + len(snap.get("state_errors") or []) + usage.get("verify_failed", 0),
        "wall_seconds": snap.get("wall_seconds") or m.get("total_seconds", 0),
    }


def quality_metrics(snap: Dict) -> Dict:
    """Automatic, label-free quality signals. citation_coverage only says a source is attached;
    evidence_sentence_coverage says a stored sentence actually states the claim (a proxy for citation
    correctness, not proof of it). Real correctness needs a hand-labelled dataset."""
    results = snap.get("fact_check_results") or []
    n = len(results)
    traced = trace_all(results, snap.get("search_results") or [], snap.get("query", ""))
    pct = lambda k: round(k / n, 3) if n else 0.0
    topics = [s for s in snap.get("summaries") or []]
    checkable = [x for t in traced for x in t["sources"] if x["match"] != "unavailable"]
    return {
        "claims": n,
        "citation_coverage": pct(sum(1 for t in traced if t["sources"])),
        "evidence_sentence_coverage": pct(sum(1 for t in traced if any(s["match"] == "exact" for s in t["sources"]))),
        "multi_domain_rate": pct(sum(1 for t in traced if t["independent_domains"] >= 2)),
        "unsupported_rate": pct(sum(1 for t in traced if t["independent_domains"] < 2)),
        "citation_support_rate": round(sum(1 for x in checkable if x["match"] == "exact") / len(checkable), 3) if checkable else 0.0,
        "contradicting_citations": sum(1 for x in checkable if x["match"] == "contradicts"),
        "claims_with_conflicts": sum(1 for t in traced if t["conflicts"]),
        "topics_planned": len(topics),
        "topics_summarised": sum(1 for s in topics if s.get("status") == "ok"),
        "topics_with_confirmed": sum(1 for r in views.status_by_topic(results) if r["CONFIRMED"] > 0),
    }

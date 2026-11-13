"""Retry what failed instead of re-running the whole research (Day 27).

Works on a saved run snapshot with injected functions, so it is unit-testable without any API:
  verify_fn(row)         -> (row, sources_added, search_failed)     e.g. tools.verification.verify_claim
  fact_check_fn(rows)    -> rows with status/reasoning/gate applied  e.g. agents.fact_checking_agent.fact_check_facts
  summarize_fn(q, rows)  -> {"sub_question","summary","status",...} e.g. agents.summarization_agent.summarize_sub_topic
  report_fn(query, summaries, results) -> {"report": str}            e.g. agents.report_generator_agent.generate_report
Order: re-run failed verification searches, re-check claims whose fact-check failed, re-summarise failed or
affected sub-topics, then rebuild the report. Successful work is never redone."""
from copy import deepcopy
from datetime import datetime
from typing import Callable, Dict, List, Optional

from tools.evidence_trace import FAILED_MARKERS

VERDICT_KEYS = ("status", "reasoning", "independent_sources", "downgraded", "sources", "verified_by_search", "verification")


def _fc_failed(row: Dict) -> bool:
    return row.get("status") == "UNVERIFIED" and any(m in (row.get("reasoning") or "").lower() for m in FAILED_MARKERS)


def find_failures(snap: Dict) -> Dict:
    results = snap.get("fact_check_results") or []
    return {
        "summaries": [s.get("sub_question", "") for s in snap.get("summaries") or [] if s.get("status") == "failed"],
        "fact_check": sorted({r.get("claim", "") for r in results if _fc_failed(r)}),
        "verification": sorted({r.get("claim", "") for r in results if r.get("verification") == "search_failed"}),
    }


def count_failures(snap: Dict) -> int:
    return sum(len(v) for v in find_failures(snap).values())


def retry_failures(snap: Dict, *, summarize_fn: Callable, fact_check_fn: Callable, report_fn: Callable,
                   verify_fn: Optional[Callable] = None) -> Dict:
    """Returns {"snapshot": new_snapshot, "log": [str], "fixed": int, "remaining": int}."""
    snap = deepcopy(snap)
    before = find_failures(snap)
    log: List[str] = []
    results = snap.get("fact_check_results") or []
    by_claim: Dict[str, List[int]] = {}
    for i, r in enumerate(results):
        by_claim.setdefault(r.get("claim", ""), []).append(i)

    recheck = set(before["fact_check"])
    if verify_fn:
        for claim in before["verification"]:
            rep = results[by_claim[claim][0]]
            new_row, added, failed = verify_fn(rep)
            if failed:
                log.append(f"Verification search failed again: {claim[:70]}")
                continue
            for i in by_claim[claim]:
                results[i] = {**results[i], "sources": new_row["sources"], "verification": new_row["verification"],
                              "verified_by_search": new_row.get("verified_by_search", results[i].get("verified_by_search"))}
            recheck.add(claim)
            log.append(f"Verification search re-run ({added} new source(s)): {claim[:70]}")

    touched_topics = set()
    if recheck:
        reps = [results[by_claim[c][0]] for c in sorted(recheck)]
        checked = fact_check_fn(reps)
        for claim, rep in zip(sorted(recheck), checked):
            for i in by_claim[claim]:
                results[i] = {**results[i], **{k: rep[k] for k in VERDICT_KEYS if k in rep}}
                touched_topics.add(results[i].get("sub_question"))
        log.append(f"Re-checked {len(recheck)} claim(s).")

    summaries = snap.get("summaries") or []
    redo = {q for q in before["summaries"]} | touched_topics
    changed_summary = False
    for i, s in enumerate(summaries):
        q = s.get("sub_question")
        if q in redo:
            summaries[i] = summarize_fn(q, [r for r in results if r.get("sub_question") == q])
            changed_summary = True
            log.append(f"Re-summarised: {q[:70]} ({summaries[i].get('status')})")

    if recheck or changed_summary:
        out = report_fn(snap.get("query", ""), summaries, results)
        snap["final_report"] = out["report"]
        log.append("Report rebuilt.")
    else:
        log.append("Nothing needed retrying.")

    snap["fact_check_results"], snap["summaries"] = results, summaries
    after = find_failures(snap)
    fixed = sum(len(before[k]) - len(after[k]) for k in before)
    snap.setdefault("retries", []).append({"at": datetime.now().isoformat(timespec="seconds"), "log": log,
                                           "fixed": fixed, "remaining": sum(len(v) for v in after.values())})
    return {"snapshot": snap, "log": log, "fixed": fixed, "remaining": sum(len(v) for v in after.values())}

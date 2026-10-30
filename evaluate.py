"""Evaluation harness: runs a set of test queries through the full pipeline
and scores each report against concrete, automatically-checkable criteria —
not subjective quality, but structural guarantees the system should uphold."""
from typing import Dict, List


def eval_covers_all_planned_topics(result: Dict) -> Dict:
    """Every sub-question in the plan should have a corresponding summary.
    Directly tests the Day 21 finding."""
    planned = set(result.get("plan", []))
    covered = {s["sub_question"] for s in result.get("summaries", [])}
    missing = planned - covered
    return {
        "name": "covers_all_planned_topics",
        "passed": len(missing) == 0,
        "detail": f"{len(covered)}/{len(planned)} topics covered" + (f", missing: {missing}" if missing else ""),
    }


def eval_report_cites_sources(result: Dict) -> Dict:
    """The final report must contain the Confidence Appendix with real claim
    counts — not an empty/degenerate report."""
    report = result.get("final_report") or ""
    passed = "Confidence Appendix" in report and "Total claims reviewed" in report
    return {"name": "report_has_confidence_appendix", "passed": passed, "detail": f"report length: {len(report)} chars"}


def eval_disputes_are_surfaced(result: Dict) -> Dict:
    """If any fact was DISPUTED, the final report text must actually mention
    it — disputes shouldn't exist in fact_check_results but vanish from
    the narrative the user actually reads."""
    disputed = [f for f in result.get("fact_check_results", []) if f.get("status") == "DISPUTED"]
    if not disputed:
        return {"name": "disputes_surfaced_if_present", "passed": True, "detail": "no disputed claims this run (n/a)"}
    report = (result.get("final_report") or "").lower()
    mentions_dispute = any(w in report for w in ["dispute", "disagree", "conflict", "contradic"])
    return {
        "name": "disputes_surfaced_if_present",
        "passed": mentions_dispute,
        "detail": f"{len(disputed)} disputed claim(s) in state; dispute language in report: {mentions_dispute}",
    }


def eval_no_unhandled_errors(result: Dict) -> Dict:
    """Errors should be absorbed by fail-safes, not crash the pipeline —
    but this checks that if errors DID occur, the pipeline still completed."""
    has_errors = len(result.get("errors", [])) > 0
    completed = result.get("final_report") is not None
    return {
        "name": "completes_despite_errors",
        "passed": completed,
        "detail": f"errors: {len(result.get('errors', []))}, completed: {completed}",
    }


def eval_terminates_within_iteration_budget(result: Dict, max_expected: int = 7) -> Dict:
    """Sanity check against runaway loops — should finish well within the
    MAX_ITERATIONS safety cap, not right up against it."""
    iterations = result.get("iteration_count", 0)
    return {
        "name": "terminates_within_budget",
        "passed": iterations <= max_expected,
        "detail": f"used {iterations} iterations (budget: {max_expected})",
    }


ALL_CHECKS = [
    eval_covers_all_planned_topics,
    eval_report_cites_sources,
    eval_disputes_are_surfaced,
    eval_no_unhandled_errors,
    eval_terminates_within_iteration_budget,
]


def evaluate_run(result: Dict) -> List[Dict]:
    return [check(result) for check in ALL_CHECKS]


def print_eval_report(query: str, checks: List[Dict]):
    passed = sum(1 for c in checks if c["passed"])
    print(f"\n{'='*70}\nQUERY: {query}\nScore: {passed}/{len(checks)}\n{'='*70}")
    for c in checks:
        mark = "✅" if c["passed"] else "❌"
        print(f"{mark} {c['name']}: {c['detail']}")

from collections import defaultdict
from typing import Dict, List

from tools.llm_clients import call_llm
from tools.logger import log_agent_event

AGENT_NAME = "summarization_agent"


def _group_by_sub_question(fact_check_results: List[Dict]) -> Dict[str, List[Dict]]:
    grouped = defaultdict(list)
    for fact in fact_check_results:
        grouped[fact.get("sub_question", "unknown")].append(fact)
    return dict(grouped)


def _build_summary_prompt(sub_question: str, facts: List[Dict]) -> str:
    confirmed = [f for f in facts if f["status"] == "CONFIRMED"]
    disputed = [f for f in facts if f["status"] == "DISPUTED"]
    unverified = [f for f in facts if f["status"] == "UNVERIFIED"]

    def fmt(fs):
        return "\n".join(f'- "{f["claim"]}" (reasoning: {f.get("reasoning", "")})' for f in fs) or "(none)"

    return f"""Write a short, honest research brief (3-6 sentences) answering: "{sub_question}"

Rules:
- State CONFIRMED facts plainly.
- For DISPUTED facts, explicitly say they are disputed and describe BOTH sides.
- For UNVERIFIED facts, hedge explicitly (e.g. "according to a single source...").
- Do not invent facts. Do not hide disputes.

CONFIRMED facts:
{fmt(confirmed)}

DISPUTED facts:
{fmt(disputed)}

UNVERIFIED facts:
{fmt(unverified)}

Write only the brief itself.
"""


def summarize_sub_topic(sub_question: str, facts: List[Dict]) -> Dict:
    if not facts:
        return {
            "sub_question": sub_question,
            "summary": "No verified information was found for this sub-topic despite an active search attempt.",
            "status": "no_facts",
            "fact_counts": {"CONFIRMED": 0, "DISPUTED": 0, "UNVERIFIED": 0},
        }

    counts = {"CONFIRMED": 0, "DISPUTED": 0, "UNVERIFIED": 0}
    for f in facts:
        counts[f["status"]] = counts.get(f["status"], 0) + 1

    try:
        summary_text = call_llm(_build_summary_prompt(sub_question, facts)).strip()
        status = "ok"
    except Exception as e:
        summary_text = f"Summary could not be generated due to an error ({type(e).__name__}). Raw fact counts: {counts}."
        status = "failed"
        log_agent_event(AGENT_NAME, "summary_failed", sub_question=sub_question, error=f"{type(e).__name__}: {e}")

    log_agent_event(AGENT_NAME, "summary_done", sub_question=sub_question, status=status, **counts)
    return {"sub_question": sub_question, "summary": summary_text, "status": status, "fact_counts": counts}


def summarization_node(state: Dict) -> Dict:
    """FIXED (Day 22, per Day 21's finding): iterates over the ORIGINAL plan,
    not just the sub-questions present in fact_check_results. Any planned
    sub-question with zero results now gets an explicit 'no information
    found' summary instead of silently vanishing from the final report."""
    fact_check_results = state.get("fact_check_results", [])
    plan = state.get("plan", [])

    if not fact_check_results and not plan:
        return {
            "errors": [{"agent": AGENT_NAME, "error": "no fact_check_results or plan to summarize"}],
            "model_usage_log": [{"agent": AGENT_NAME, "status": "skipped_no_input"}],
        }

    grouped = _group_by_sub_question(fact_check_results)

    # ensure every planned sub-question gets a summary, even if it has zero facts
    all_topics = list(dict.fromkeys(list(grouped.keys()) + plan))  # preserves order, dedupes

    summaries = [summarize_sub_topic(q, grouped.get(q, [])) for q in all_topics]

    return {
        "summaries": summaries,
        "model_usage_log": [{"agent": AGENT_NAME, "provider": "groq", "sub_topics": len(summaries)}],
    }

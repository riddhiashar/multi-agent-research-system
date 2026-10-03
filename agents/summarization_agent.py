from collections import defaultdict
from typing import Dict, List

from tools.llm_clients import call_llm
from tools.logger import log_agent_event

AGENT_NAME = "summarization_agent"


def _group_by_sub_question(fact_check_results: List[Dict]) -> Dict[str, List[Dict]]:
    grouped = defaultdict(list)
    for fact in fact_check_results:
        key = fact.get("sub_question", "unknown")
        grouped[key].append(fact)
    return dict(grouped)


def _build_summary_prompt(sub_question: str, facts: List[Dict]) -> str:
    confirmed = [f for f in facts if f["status"] == "CONFIRMED"]
    disputed = [f for f in facts if f["status"] == "DISPUTED"]
    unverified = [f for f in facts if f["status"] == "UNVERIFIED"]

    def fmt(fs):
        return "\n".join(f'- "{f["claim"]}" (reasoning: {f.get("reasoning", "")})' for f in fs) or "(none)"

    return f"""Write a short, honest research brief (3-6 sentences) answering: "{sub_question}"

You MUST follow these rules exactly:
- State CONFIRMED facts plainly, as established.
- For DISPUTED facts, explicitly say they are disputed and briefly describe BOTH sides —
  never silently pick one side or omit the disagreement.
- For UNVERIFIED facts, hedge them explicitly (e.g. "according to a single source...").
- Do not invent facts not listed below. Do not soften or hide disputes to make the
  summary sound cleaner.

CONFIRMED facts:
{fmt(confirmed)}

DISPUTED facts:
{fmt(disputed)}

UNVERIFIED facts:
{fmt(unverified)}

Write only the brief itself, no preamble.
"""


def summarize_sub_topic(sub_question: str, facts: List[Dict]) -> Dict:
    """Summarizes one sub-topic's fact-checked claims. Never raises: on failure,
    returns a summary that explicitly states it could not be generated, rather
    than silently omitting the sub-topic or fabricating a summary."""
    if not facts:
        return {
            "sub_question": sub_question,
            "summary": "No verified information available for this sub-topic.",
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
        summary_text = (f"Summary could not be generated due to an error "
                        f"({type(e).__name__}). Raw fact counts: {counts}.")
        status = "failed"
        log_agent_event(AGENT_NAME, "summary_failed", sub_question=sub_question,
                        error=f"{type(e).__name__}: {e}")

    log_agent_event(AGENT_NAME, "summary_done", sub_question=sub_question, status=status,
                    **counts)

    return {
        "sub_question": sub_question,
        "summary": summary_text,
        "status": status,
        "fact_counts": counts,
    }


def summarization_node(state: Dict) -> Dict:
    """LangGraph node: summarizes every sub-topic's fact-checked claims."""
    fact_check_results = state.get("fact_check_results", [])
    if not fact_check_results:
        return {"errors": [{"agent": AGENT_NAME, "error": "no fact_check_results to summarize"}]}

    grouped = _group_by_sub_question(fact_check_results)
    summaries = [summarize_sub_topic(q, facts) for q, facts in grouped.items()]

    return {
        "summaries": summaries,
        "model_usage_log": [{"agent": AGENT_NAME, "provider": "groq",
                            "sub_topics": len(summaries)}],
    }

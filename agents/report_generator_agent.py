from typing import Dict, List
from collections import Counter

from tools.llm_clients import call_llm
from tools.logger import log_agent_event

AGENT_NAME = "report_generator_agent"


def _build_narrative_prompt(query: str, summaries: List[Dict]) -> str:
    sections = "\n\n".join(f"## {s['sub_question']}\n{s['summary']}" for s in summaries)
    return f"""Write a clear, well-organized research report answering: "{query}"

Combine the sub-topic briefs below into flowing prose with headings. Do not invent facts.
Preserve hedging language exactly as given — do not make claims sound more certain.

Sub-topic briefs:
{sections}

Markdown, short intro, one section per sub-topic, one-paragraph conclusion.
Do not include a confidence/sources appendix — that is added separately.
"""


def _generate_narrative(query: str, summaries: List[Dict]) -> Dict:
    try:
        text = call_llm(_build_narrative_prompt(query, summaries)).strip()
        return {"text": text, "method": "llm"}
    except Exception as e:
        log_agent_event(AGENT_NAME, "narrative_failed", error=f"{type(e).__name__}: {e}")
        fallback = f"# Research Report: {query}\n\n" + "\n\n".join(
            f"## {s['sub_question']}\n{s['summary']}" for s in summaries)
        return {"text": fallback, "method": "fallback_template"}


def _build_confidence_appendix(fact_check_results: List[Dict]) -> str:
    if not fact_check_results:
        return "\n\n---\n## Confidence Appendix\nNo fact-checked claims were available.\n"

    counts = Counter(f["status"] for f in fact_check_results)
    total = len(fact_check_results)
    lines = ["\n\n---\n## Confidence Appendix\n"]
    lines.append(f"**Total claims reviewed:** {total}  ")
    lines.append(f"**Confirmed:** {counts.get('CONFIRMED', 0)} | "
                f"**Disputed:** {counts.get('DISPUTED', 0)} | "
                f"**Unverified:** {counts.get('UNVERIFIED', 0)}\n")

    downgraded = [f for f in fact_check_results if f.get("downgraded")]
    if downgraded:
        lines.append(f"**Held back by the evidence gate (fewer than 2 independent sources):** {len(downgraded)}\n")

    disputed = [f for f in fact_check_results if f["status"] == "DISPUTED"]
    if disputed:
        lines.append("### ⚠️ Disputed claims\n")
        for f in disputed:
            lines.append(f"- \"{f['claim']}\" — *{f.get('reasoning', '')}*")

    unverified = [f for f in fact_check_results if f["status"] == "UNVERIFIED"]
    if unverified:
        lines.append("\n### ❓ Unverified claims (single source, no corroboration)\n")
        for f in unverified:
            lines.append(f"- \"{f['claim']}\"")

    return "\n".join(lines)


def generate_report(query: str, summaries: List[Dict], fact_check_results: List[Dict]) -> Dict:
    if not summaries:
        return {
            "report": f"# Research Report: {query}\n\nNo verified information was found for this query.",
            "method": "no_data",
        }

    narrative = _generate_narrative(query, summaries)
    appendix = _build_confidence_appendix(fact_check_results)
    full_report = narrative["text"] + appendix

    log_agent_event(AGENT_NAME, "report_done", method=narrative["method"],
                    sub_topics=len(summaries), total_claims=len(fact_check_results))
    return {"report": full_report, "method": narrative["method"]}


def report_generator_node(state: Dict) -> Dict:
    summaries = state.get("summaries", [])
    fact_check_results = state.get("fact_check_results", [])
    query = state.get("query", "Unknown query")

    if not summaries:
        return {
            "errors": [{"agent": AGENT_NAME, "error": "no summaries to build report from"}],
            "model_usage_log": [{"agent": AGENT_NAME, "status": "skipped_no_input"}],
        }

    result = generate_report(query, summaries, fact_check_results)
    return {
        "final_report": result["report"],
        "model_usage_log": [{"agent": AGENT_NAME, "provider": "groq", "method": result["method"]}],
    }

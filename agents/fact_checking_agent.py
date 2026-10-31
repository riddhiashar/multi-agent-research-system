import json
import re
from typing import Dict, List

from tools.llm_clients import call_llm
from tools.logger import log_agent_event
from tools.evidence import independent_source_count, MIN_INDEPENDENT_SOURCES

AGENT_NAME = "fact_checking_agent"
MAX_FACTS_PER_BATCH = 15


def _build_fact_check_prompt(facts: List[Dict]) -> str:
    numbered = "\n".join(
        f'{i}. "{f["claim"]}" (independent sources: {independent_source_count(f.get("sources", []))})'
        for i, f in enumerate(facts)
    )
    return f"""You are fact-checking a set of claims gathered during research. For EACH
numbered claim below, classify it as exactly one of:

- CONFIRMED: at least {MIN_INDEPENDENT_SOURCES} independent sources (see the count) and nothing in this list contradicts it
- DISPUTED: contradicts another claim in this list
- UNVERIFIED: fewer than {MIN_INDEPENDENT_SOURCES} independent sources, and no contradiction

Claims:
{numbered}

Respond with ONLY a JSON array: [{{"index": 0, "status": "CONFIRMED", "reasoning": "one short sentence"}}, ...]
"""


def _parse_fact_check_response(raw_text: str, expected_count: int) -> List[Dict]:
    cleaned = re.sub(r"^```(json)?", "", raw_text.strip()).strip()
    cleaned = re.sub(r"```$", "", cleaned).strip()
    try:
        parsed = json.loads(cleaned)
        if not isinstance(parsed, list):
            return []
        results = {}
        for item in parsed:
            idx = item.get("index")
            if isinstance(idx, int) and 0 <= idx < expected_count:
                status = item.get("status", "UNVERIFIED").upper()
                if status not in ("CONFIRMED", "DISPUTED", "UNVERIFIED"):
                    status = "UNVERIFIED"
                results[idx] = {"status": status, "reasoning": item.get("reasoning", "")}
        return [results.get(i, {"status": "UNVERIFIED", "reasoning": "not returned by model"})
                for i in range(expected_count)]
    except json.JSONDecodeError:
        return []


def apply_evidence_gate(fact: Dict, verdict: Dict) -> Dict:
    """Deterministic check applied AFTER the LLM verdict: a claim cannot stay CONFIRMED
    unless it has enough independent source domains, whatever the model said."""
    count = independent_source_count(fact.get("sources", []))
    status, reasoning = verdict["status"], verdict["reasoning"]
    downgraded = False
    if status == "CONFIRMED" and count < MIN_INDEPENDENT_SOURCES:
        status = "UNVERIFIED"
        reasoning = f"[downgraded: {count} independent source(s), need {MIN_INDEPENDENT_SOURCES}] {reasoning}"
        downgraded = True
    return {**fact, "status": status, "reasoning": reasoning,
            "independent_sources": count, "downgraded": downgraded}


def fact_check_batch(facts: List[Dict]) -> List[Dict]:
    if not facts:
        return []
    try:
        raw_output = call_llm(_build_fact_check_prompt(facts))
        verdicts = _parse_fact_check_response(raw_output, len(facts))
        if not verdicts:
            raise ValueError("fact-check response could not be parsed")
    except Exception as e:
        log_agent_event(AGENT_NAME, "batch_failed", error=f"{type(e).__name__}: {e}", batch_size=len(facts))
        verdicts = [{"status": "UNVERIFIED", "reasoning": "fact-check call failed; defaulted to unverified"}
                    for _ in facts]

    checked = [apply_evidence_gate(f, v) for f, v in zip(facts, verdicts)]

    counts: Dict[str, int] = {}
    for c in checked:
        counts[c["status"]] = counts.get(c["status"], 0) + 1
    log_agent_event(AGENT_NAME, "batch_done", downgraded=sum(1 for c in checked if c["downgraded"]), **counts)
    return checked


def fact_check_facts(facts: List[Dict]) -> List[Dict]:
    all_checked = []
    for i in range(0, len(facts), MAX_FACTS_PER_BATCH):
        all_checked.extend(fact_check_batch(facts[i:i + MAX_FACTS_PER_BATCH]))
    return all_checked


def fact_checking_node(state: Dict) -> Dict:
    extracted_facts = state.get("extracted_facts", [])
    if not extracted_facts:
        return {
            "errors": [{"agent": AGENT_NAME, "error": "no extracted_facts to check"}],
            "model_usage_log": [{"agent": AGENT_NAME, "status": "skipped_no_input"}],
        }

    checked = fact_check_facts(extracted_facts)
    return {
        "fact_check_results": checked,
        "model_usage_log": [{"agent": AGENT_NAME, "provider": "groq", "facts_checked": len(checked)}],
    }

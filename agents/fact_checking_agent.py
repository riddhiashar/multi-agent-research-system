import json
import re
from typing import Dict, List

from tools.llm_clients import call_llm
from tools.logger import log_agent_event

AGENT_NAME = "fact_checking_agent"
MAX_FACTS_PER_BATCH = 15


def _build_fact_check_prompt(facts: List[Dict]) -> str:
    numbered = "\n".join(f'{i}. "{f["claim"]}" (sources: {len(f.get("sources", []))})'
                         for i, f in enumerate(facts))
    return f"""You are fact-checking a set of claims gathered during research. For EACH
numbered claim below, classify it as exactly one of:

- CONFIRMED: corroborated by multiple independent sources, or clearly consistent with other claims
- DISPUTED: contradicts another claim in this list
- UNVERIFIED: plausible but only backed by a single source, with no corroboration or contradiction

Claims:
{numbered}

Respond with ONLY a JSON array: [{{"index": 0, "status": "CONFIRMED", "reasoning": "..."}}, ...]
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

    checked = [{**fact, "status": v["status"], "reasoning": v["reasoning"]} for fact, v in zip(facts, verdicts)]
    counts = {}
    for v in verdicts:
        counts[v["status"]] = counts.get(v["status"], 0) + 1
    log_agent_event(AGENT_NAME, "batch_done", **counts)
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

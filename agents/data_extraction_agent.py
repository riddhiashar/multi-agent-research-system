import json
import re
from typing import Dict, List

from tools.llm_clients import call_llm
from tools.logger import log_agent_event

AGENT_NAME = "data_extraction_agent"

# Day 23: ask the model which snippet(s) support each claim. This costs a few extra output
# tokens per fact. If Groq's per-minute budget gets tight, set it to False: claims then
# inherit all of the search result's sources and are flagged attribution="shared".
ATTRIBUTE_CLAIMS = True


def _build_extraction_prompt(sub_question: str, answer: str, attribute: bool = True) -> str:
    if attribute:
        output_rules = (
            "- The text is split into snippets tagged [S1], [S2], and so on. For each fact, list the\n"
            "  numbers of the snippets that actually state it. Do not guess: if only one snippet\n"
            "  states a fact, list only that one.\n"
            '- Respond with ONLY a JSON array of objects, nothing else: [{"claim": "...", "sources": [1, 3]}]\n'
            "- If there are no clear factual claims, return []."
        )
    else:
        output_rules = (
            "- Respond with ONLY a JSON array of strings, nothing else.\n"
            "- If there are no clear factual claims, return []."
        )
    return f"""Extract discrete, atomic factual claims from the text below, related to the
question: "{sub_question}"

Rules:
- Each fact must be a single, self-contained, checkable statement.
- Do NOT include opinions, vague generalities, or marketing language.
{output_rules}

Text:
\"\"\"{answer}\"\"\"
"""


def _parse_json_facts(raw_text: str) -> List[Dict]:
    """Returns [{"claim": str, "source_ids": [int, ...]}]. Accepts the object format and
    the older plain-string format."""
    cleaned = re.sub(r"^```(json)?", "", raw_text.strip()).strip()
    cleaned = re.sub(r"```$", "", cleaned).strip()
    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError:
        return []
    if not isinstance(parsed, list):
        return []

    facts = []
    for item in parsed:
        if isinstance(item, str):
            claim, ids = item.strip(), []
        elif isinstance(item, dict):
            claim = str(item.get("claim", "")).strip()
            raw_ids = item.get("sources", [])
            ids = [i for i in raw_ids if isinstance(i, int)] if isinstance(raw_ids, list) else []
        else:
            continue
        if claim:
            facts.append({"claim": claim, "source_ids": ids})
    return facts


def _resolve_sources(source_ids: List[int], all_sources: List[Dict]) -> List[Dict]:
    by_id = {s.get("id"): s for s in all_sources if s.get("id") is not None}
    return [by_id[i] for i in dict.fromkeys(source_ids) if i in by_id]


def extract_facts_from_result(search_result: Dict) -> List[Dict]:
    sub_question = search_result.get("sub_question", "")
    answer = search_result.get("answer", "")
    sources = search_result.get("sources", [])

    if not answer.strip():
        return []

    attribute = ATTRIBUTE_CLAIMS and "[S1]" in answer

    try:
        raw_output = call_llm(_build_extraction_prompt(sub_question, answer, attribute))
        raw_facts = _parse_json_facts(raw_output)
    except Exception as e:
        log_agent_event(AGENT_NAME, "extraction_failed", sub_question=sub_question,
                        error=f"{type(e).__name__}: {e}")
        return []

    facts = []
    for item in raw_facts:
        if not attribute:
            claim_sources, attribution = sources, "shared"
        else:
            cited = _resolve_sources(item["source_ids"], sources)
            if cited:
                claim_sources, attribution = cited, "per_claim"
            else:
                # Tagged text but no valid citation: fail toward UNVERIFIED, not toward trusted.
                claim_sources, attribution = [], "unattributed"
        facts.append({"claim": item["claim"], "sub_question": sub_question,
                      "sources": claim_sources, "attribution": attribution})

    log_agent_event(AGENT_NAME, "extraction_done", sub_question=sub_question, facts_found=len(facts))
    return facts


def data_extraction_node(state: Dict) -> Dict:
    search_results = state.get("search_results", [])
    if not search_results:
        return {
            "errors": [{"agent": AGENT_NAME, "error": "no search_results to extract from"}],
            "model_usage_log": [{"agent": AGENT_NAME, "status": "skipped_no_input"}],
        }

    all_facts: List[Dict] = []
    errors: List[Dict] = []
    for result in search_results:
        try:
            all_facts.extend(extract_facts_from_result(result))
        except Exception as e:
            errors.append({"agent": AGENT_NAME, "sub_question": result.get("sub_question", "unknown"),
                           "error": f"{type(e).__name__}: {e}"})

    return {
        "extracted_facts": all_facts,
        "errors": errors,
        "model_usage_log": [{"agent": AGENT_NAME, "provider": "groq", "calls": len(search_results)}],
    }

import json
import re
from typing import Dict, List

from tools.llm_clients import call_llm
from tools.logger import log_agent_event

AGENT_NAME = "data_extraction_agent"


def _build_extraction_prompt(sub_question: str, answer: str) -> str:
    return f"""Extract discrete, atomic factual claims from the text below, related to the
question: "{sub_question}"

Rules:
- Each fact must be a single, self-contained, checkable statement.
- Do NOT include opinions, vague generalities, or marketing language.
- If the text contains no clear factual claims, return an empty list.
- Respond with ONLY a JSON array of strings, nothing else.

Text:
\"\"\"{answer}\"\"\"

Example valid response: ["Fact one stated plainly.", "Fact two stated plainly."]
Example for irrelevant/no facts: []
"""


def _parse_json_facts(raw_text: str) -> List[str]:
    cleaned = re.sub(r"^```(json)?", "", raw_text.strip()).strip()
    cleaned = re.sub(r"```$", "", cleaned).strip()
    try:
        parsed = json.loads(cleaned)
        if isinstance(parsed, list):
            return [str(f).strip() for f in parsed if str(f).strip()]
        return []
    except json.JSONDecodeError:
        return []


def extract_facts_from_result(search_result: Dict) -> List[Dict]:
    sub_question = search_result.get("sub_question", "")
    answer = search_result.get("answer", "")
    sources = search_result.get("sources", [])

    if not answer.strip():
        return []

    try:
        raw_output = call_llm(_build_extraction_prompt(sub_question, answer))
        raw_facts = _parse_json_facts(raw_output)
    except Exception as e:
        log_agent_event(AGENT_NAME, "extraction_failed", sub_question=sub_question,
                        error=f"{type(e).__name__}: {e}")
        return []

    facts = [{"claim": t, "sub_question": sub_question, "sources": sources} for t in raw_facts]
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

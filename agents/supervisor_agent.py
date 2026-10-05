from typing import Dict, List

from tools.llm_clients import call_llm
from tools.logger import log_agent_event

AGENT_NAME = "supervisor_agent"
MAX_ITERATIONS = 8

# Hard data-dependency order: each stage needs the previous stage's output to function.
# This is NOT an LLM decision — it's fixed by what data exists at each point, so routing
# it deterministically is more reliable, cheaper, and fully testable without mocking.
PIPELINE = [
    ("web_search_agent", "web_search"),
    ("data_extraction_agent", "data_extraction"),
    ("fact_checking_agent", "fact_checking"),
    ("summarization_agent", "summarization"),
    ("report_generator_agent", "report_generator"),
]


def _build_plan_prompt(query: str) -> str:
    return f"""Break this research question into 2-5 specific, researchable sub-questions:
"{query}"

Respond with each sub-question on its own line, nothing else. No numbering, no preamble.
"""


def generate_plan(query: str) -> List[str]:
    """The one genuinely ambiguous decision the Supervisor makes with an LLM —
    everything else below is deterministic rule-based routing."""
    try:
        raw = call_llm(_build_plan_prompt(query))
        sub_questions = [line.strip("- ").strip() for line in raw.split("\n") if line.strip()]
        return sub_questions[:5] if sub_questions else [query]
    except Exception as e:
        log_agent_event(AGENT_NAME, "plan_generation_failed", error=f"{type(e).__name__}: {e}")
        return [query]  # fail-safe: treat the raw query as the only sub-question


def _agents_completed(state: Dict) -> set:
    """Reads model_usage_log to determine which agents have already been attempted
    (success OR a logged 'ran but found nothing' / 'skipped, no input' marker)."""
    return {entry.get("agent") for entry in state.get("model_usage_log", []) if entry.get("agent")}


def supervisor_node(state: Dict) -> Dict:
    """Deterministic router. Only calls an LLM once, to generate the initial plan."""
    iteration = state.get("iteration_count", 0) + 1

    if iteration > MAX_ITERATIONS:
        log_agent_event(AGENT_NAME, "max_iterations_exceeded", iteration=iteration)
        return {"iteration_count": iteration, "next_agent": "failed"}

    updates: Dict = {"iteration_count": iteration}

    if not state.get("plan"):
        plan = generate_plan(state["query"])
        updates["plan"] = plan
        updates["next_agent"] = "web_search"
        log_agent_event(AGENT_NAME, "plan_generated", sub_questions=len(plan))
        return updates

    completed = _agents_completed(state)
    for agent_name, node_name in PIPELINE:
        if agent_name not in completed:
            updates["next_agent"] = node_name
            log_agent_event(AGENT_NAME, "routing", decision=node_name, iteration=iteration)
            return updates

    updates["next_agent"] = "done"
    log_agent_event(AGENT_NAME, "pipeline_complete", iteration=iteration)
    return updates


def route_from_supervisor(state: Dict) -> str:
    """Conditional-edge function: just reads what supervisor_node already decided."""
    return state.get("next_agent", "failed")

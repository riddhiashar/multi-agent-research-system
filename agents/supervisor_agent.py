from typing import Dict, List

from langgraph.types import Send
from tools.llm_clients import call_llm
from tools.logger import log_agent_event
from agents.web_search_agent import MAX_QUERIES

AGENT_NAME = "supervisor_agent"
MAX_ITERATIONS = 8

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
    try:
        raw = call_llm(_build_plan_prompt(query))
        sub_questions = [line.strip("- ").strip() for line in raw.split("\n") if line.strip()]
        return sub_questions[:5] if sub_questions else [query]
    except Exception as e:
        log_agent_event(AGENT_NAME, "plan_generation_failed", error=f"{type(e).__name__}: {e}")
        return [query]


def _agents_completed(state: Dict) -> set:
    return {entry.get("agent") for entry in state.get("model_usage_log", []) if entry.get("agent")}


def supervisor_node(state: Dict) -> Dict:
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


def route_from_supervisor(state: Dict):
    """Returns either a plain node-name string (unchanged stages) or a list of
    Send objects (NEW, Day 17) to fan out web_search into parallel branches —
    one per sub-question, capped at MAX_QUERIES, deduped."""
    next_agent = state.get("next_agent", "failed")

    if next_agent == "web_search":
        seen, queries = set(), []
        for q in state.get("plan", []):
            q = q.strip()
            if q and q.lower() not in seen:
                seen.add(q.lower())
                queries.append(q)
        queries = queries[:MAX_QUERIES]

        if not queries:
            return "data_extraction"  # nothing to search; let downstream handle empty input

        return [Send("web_search_single", {"query": q}) for q in queries]

    return next_agent

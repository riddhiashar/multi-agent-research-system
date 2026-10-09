import time
from typing import Dict, List

from tools.llm_clients import research_with_fallback
from tools.logger import log_agent_event

AGENT_NAME = "web_search_agent"
MAX_QUERIES = 5


def _dedupe_sources(sources: List[Dict]) -> List[Dict]:
    seen, unique = set(), []
    for s in sources:
        url = s.get("url")
        if url and url not in seen:
            seen.add(url)
            unique.append(s)
    return unique


def _is_rate_limit_error(e: Exception) -> bool:
    """Detects rate-limit errors across providers by message content, since
    exception TYPES differ between Groq/Gemini/Tavily SDKs but the signal
    (429, 'rate limit', 'RESOURCE_EXHAUSTED') is consistent enough to check
    as text. Found necessary after Day 15's benchmark showed parallel calls
    trigger this more than sequential ones."""
    msg = str(e).lower()
    return "429" in msg or "rate limit" in msg or "resource_exhausted" in msg


def search_one(sub_question: str, max_retries: int = 2) -> Dict:
    """Research a single sub-question. Never raises. Retries specifically on
    rate-limit errors with backoff (new, Day 17) before falling back to a
    failed status — this is the prerequisite Day 15 flagged before shipping
    real parallelism."""
    start = time.time()
    last_error = None

    for attempt in range(max_retries + 1):
        try:
            raw = research_with_fallback(sub_question)
            sources = _dedupe_sources(raw.get("sources", []))
            answer = (raw.get("answer") or "").strip()
            status = "ok" if (answer and sources) else "empty"
            result = {
                "sub_question": sub_question, "answer": answer, "sources": sources,
                "provider": raw.get("provider", "unknown"), "status": status,
                "latency_s": round(time.time() - start, 2),
            }
            log_agent_event(AGENT_NAME, "search_done", status=result["status"],
                            provider=result["provider"], sources=len(result["sources"]),
                            retries=attempt)
            return result
        except Exception as e:
            last_error = e
            if _is_rate_limit_error(e) and attempt < max_retries:
                wait = 2 ** attempt
                log_agent_event(AGENT_NAME, "rate_limited_retrying", sub_question=sub_question,
                                attempt=attempt, wait_s=wait)
                time.sleep(wait)
                continue
            break  # non-rate-limit error, or retries exhausted: fail now

    result = {
        "sub_question": sub_question, "answer": "", "sources": [], "provider": "none",
        "status": "failed", "error": f"{type(last_error).__name__}: {last_error}",
        "latency_s": round(time.time() - start, 2),
    }
    log_agent_event(AGENT_NAME, "search_done", status="failed",
                    provider="none", sources=0)
    return result


def web_search_node(state: Dict) -> Dict:
    """Sequential path — kept for direct testing and as a non-graph entry
    point. The real graph (Day 17) uses search_single_node via Send instead."""
    seen, queries = set(), []
    for q in state.get("plan", []):
        q = q.strip()
        if q and q.lower() not in seen:
            seen.add(q.lower())
            queries.append(q)
    queries = queries[:MAX_QUERIES]

    if not queries:
        return {
            "errors": [{"agent": AGENT_NAME, "error": "empty plan: nothing to search"}],
            "model_usage_log": [{"agent": AGENT_NAME, "status": "skipped_no_input"}],
        }

    results = [search_one(q) for q in queries]
    good = [r for r in results if r["status"] != "failed"]
    failed = [r for r in results if r["status"] == "failed"]

    usage: Dict[str, int] = {}
    for r in good:
        usage[r["provider"]] = usage.get(r["provider"], 0) + 1

    return {
        "search_results": good,
        "errors": [{"agent": AGENT_NAME, "sub_question": r["sub_question"], "error": r["error"]}
                   for r in failed],
        "model_usage_log": [{"agent": AGENT_NAME, "provider": p, "queries": n}
                            for p, n in usage.items()] or [{"agent": AGENT_NAME, "status": "ran_no_results"}],
    }


def search_single_node(state: Dict) -> Dict:
    """Send-target node: handles ONE query, invoked N times in parallel by
    LangGraph's Send mechanism. Each parallel invocation gets an isolated
    state slice ({"query": ...}), and results merge back into the parent
    state via the Annotated[list, add] reducers — verified safe on Day 16."""
    query = state["query"]
    result = search_one(query)

    if result["status"] == "failed":
        return {
            "errors": [{"agent": AGENT_NAME, "sub_question": query, "error": result["error"]}],
            "model_usage_log": [{"agent": AGENT_NAME, "provider": "none", "query": query, "status": "failed"}],
        }

    return {
        "search_results": [result],
        "model_usage_log": [{"agent": AGENT_NAME, "provider": result["provider"], "query": query}],
    }

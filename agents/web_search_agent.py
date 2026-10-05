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


def search_one(sub_question: str) -> Dict:
    start = time.time()
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
    except Exception as e:
        result = {
            "sub_question": sub_question, "answer": "", "sources": [], "provider": "none",
            "status": "failed", "error": f"{type(e).__name__}: {e}",
            "latency_s": round(time.time() - start, 2),
        }
    log_agent_event(AGENT_NAME, "search_done", status=result["status"],
                    provider=result["provider"], sources=len(result["sources"]))
    return result


def web_search_node(state: Dict) -> Dict:
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

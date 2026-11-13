"""Builds the saved-run snapshot (Day 27). Used by the app and by benchmark.py so both store the same schema.
Schema 2 adds search_results (snippets, for evidence tracing), usage_log, state_errors and plan.
Older snapshots without these keys still open: every reader uses .get with a default."""
from datetime import datetime
from typing import Dict, Iterable, Optional


def build_snapshot(query: str, final: Dict, *, metrics: Dict, evidence: Dict, llm_errors: Iterable = (),
                   stages: Optional[list] = None, events: Optional[list] = None, options: Optional[Dict] = None,
                   cache: Optional[Dict] = None, wall_seconds: float = 0.0) -> Dict:
    return {
        "query": query, "created_at": datetime.now().isoformat(timespec="seconds"),
        "final_report": final.get("final_report") or "",
        "fact_check_results": final.get("fact_check_results") or [],
        "summaries": final.get("summaries") or [],
        "search_results": [r for r in final.get("search_results") or [] if isinstance(r, dict)],
        "usage_log": final.get("model_usage_log") or [],
        "state_errors": final.get("errors") or [],
        "plan": final.get("plan") or [],
        "metrics": metrics, "evidence": evidence,
        "llm_errors": [list(e) for e in llm_errors],
        "stages": stages or [], "events": events or [], "options": options or {},
        "cache": cache or {}, "wall_seconds": round(wall_seconds, 1),
    }

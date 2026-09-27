from typing import TypedDict, Annotated, List, Dict, Optional
from operator import add


class ResearchState(TypedDict):
    """Shared state that flows through every node in the research graph."""
    query: str
    plan: List[str]
    needs_more_research: bool

    # fields multiple parallel agents may write to — append, don't overwrite
    search_results: Annotated[List[Dict], add]
    extracted_facts: Annotated[List[Dict], add]
    fact_check_results: Annotated[List[Dict], add]
    summaries: Annotated[List[Dict], add]

    final_report: Optional[str]

    # observability — which model handled which agent (LangSmith covers detailed tracing;
    # this stays for report-time attribution, e.g. citing which sources fed the final report)
    model_usage_log: Annotated[List[Dict], add]

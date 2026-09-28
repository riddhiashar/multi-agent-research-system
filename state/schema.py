from typing import TypedDict, Annotated, List, Dict, Optional
from operator import add


class ResearchState(TypedDict):
    """Shared state that flows through every node in the research graph. (v2, Day 7)"""
    query: str
    plan: List[str]


    next_agent: str
    iteration_count: int          
    needs_more_research: bool


    search_results: Annotated[List[Dict], add]
    extracted_facts: Annotated[List[Dict], add]
    fact_check_results: Annotated[List[Dict], add]
    summaries: Annotated[List[Dict], add]


    errors: Annotated[List[Dict], add]

    final_report: Optional[str]

    model_usage_log: Annotated[List[Dict], add]

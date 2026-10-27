from langgraph.graph import StateGraph, START, END
from state.schema import ResearchState
from agents.supervisor_agent import supervisor_node, route_from_supervisor
from agents.web_search_agent import search_single_node
from agents.data_extraction_agent import data_extraction_node
from agents.fact_checking_agent import fact_checking_node
from agents.summarization_agent import summarization_node
from agents.report_generator_agent import report_generator_node


def build_graph(checkpointer=None, interrupt_before=None):
    """checkpointer + interrupt_before are optional (Day 19) — when provided,
    the graph pauses before the named node(s) so a human can review state
    before execution continues. Omitted, the graph runs fully autonomously
    as it has since Day 13."""
    graph = StateGraph(ResearchState)

    graph.add_node("supervisor", supervisor_node)
    graph.add_node("web_search_single", search_single_node)
    graph.add_node("data_extraction", data_extraction_node)
    graph.add_node("fact_checking", fact_checking_node)
    graph.add_node("summarization", summarization_node)
    graph.add_node("report_generator", report_generator_node)

    graph.add_edge(START, "supervisor")
    graph.add_conditional_edges("supervisor", route_from_supervisor, {
        "web_search_single": "web_search_single",
        "data_extraction": "data_extraction",
        "fact_checking": "fact_checking",
        "summarization": "summarization",
        "report_generator": "report_generator",
        "done": END,
        "failed": END,
    })
    for node in ["web_search_single", "data_extraction", "fact_checking", "summarization", "report_generator"]:
        graph.add_edge(node, "supervisor")

    compile_kwargs = {}
    if checkpointer is not None:
        compile_kwargs["checkpointer"] = checkpointer
    if interrupt_before is not None:
        compile_kwargs["interrupt_before"] = interrupt_before

    return graph.compile(**compile_kwargs)

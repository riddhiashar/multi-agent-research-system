from langgraph.graph import StateGraph, START, END
from state.schema import ResearchState
from agents.supervisor_agent import supervisor_node, route_from_supervisor
from agents.web_search_agent import web_search_node
from agents.data_extraction_agent import data_extraction_node
from agents.fact_checking_agent import fact_checking_node
from agents.summarization_agent import summarization_node
from agents.report_generator_agent import report_generator_node


def build_graph():
    graph = StateGraph(ResearchState)

    graph.add_node("supervisor", supervisor_node)
    graph.add_node("web_search", web_search_node)
    graph.add_node("data_extraction", data_extraction_node)
    graph.add_node("fact_checking", fact_checking_node)
    graph.add_node("summarization", summarization_node)
    graph.add_node("report_generator", report_generator_node)

    graph.add_edge(START, "supervisor")
    graph.add_conditional_edges("supervisor", route_from_supervisor, {
        "web_search": "web_search",
        "data_extraction": "data_extraction",
        "fact_checking": "fact_checking",
        "summarization": "summarization",
        "report_generator": "report_generator",
        "done": END,
        "failed": END,
    })
    for node in ["web_search", "data_extraction", "fact_checking", "summarization", "report_generator"]:
        graph.add_edge(node, "supervisor")

    return graph.compile()

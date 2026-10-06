"""Integration tests for the full graph. Unlike the per-agent unit tests, these
mock entire NODE functions (not just call_llm) to verify the graph's wiring,
routing, and state-passing are correct — independent of any agent's internal logic."""
import agents.supervisor_agent as sup
from graph.build_graph import build_graph


def _mock_node(name, **extra_state):
    def node(state):
        return {"model_usage_log": [{"agent": name}], **extra_state}
    return node


def test_full_pipeline_runs_in_correct_order(monkeypatch):
    call_order = []

    def tracking_node(name, **extra_state):
        def node(state):
            call_order.append(name)
            return {"model_usage_log": [{"agent": name}], **extra_state}
        return node

    monkeypatch.setattr(sup, "generate_plan", lambda q: ["sub-q 1"])

    import graph.build_graph as gb
    monkeypatch.setattr(gb, "web_search_node", tracking_node("web_search_agent", search_results=[{"x": 1}]))
    monkeypatch.setattr(gb, "data_extraction_node", tracking_node("data_extraction_agent", extracted_facts=[{"x": 1}]))
    monkeypatch.setattr(gb, "fact_checking_node", tracking_node("fact_checking_agent", fact_check_results=[{"x": 1}]))
    monkeypatch.setattr(gb, "summarization_node", tracking_node("summarization_agent", summaries=[{"x": 1}]))
    monkeypatch.setattr(gb, "report_generator_node", tracking_node("report_generator_agent", final_report="done"))

    app = gb.build_graph()
    initial_state = {
        "query": "test query", "plan": [], "next_agent": "", "iteration_count": 0,
        "needs_more_research": False, "search_results": [], "extracted_facts": [],
        "fact_check_results": [], "summaries": [], "errors": [], "final_report": None,
        "model_usage_log": [],
    }
    result = app.invoke(initial_state)

    assert call_order == [
        "web_search_agent", "data_extraction_agent", "fact_checking_agent",
        "summarization_agent", "report_generator_agent",
    ]
    assert result["final_report"] == "done"


def test_graph_terminates_on_empty_plan_fallback(monkeypatch):
    """If plan generation itself fails, generate_plan's fail-safe returns the raw
    query as a 1-item plan — the graph should still proceed, not hang."""
    monkeypatch.setattr(sup, "call_llm", lambda p: (_ for _ in ()).throw(RuntimeError("down")))

    import graph.build_graph as gb
    monkeypatch.setattr(gb, "web_search_node", _mock_node("web_search_agent", search_results=[]))
    monkeypatch.setattr(gb, "data_extraction_node", _mock_node("data_extraction_agent", extracted_facts=[]))
    monkeypatch.setattr(gb, "fact_checking_node", _mock_node("fact_checking_agent", fact_check_results=[]))
    monkeypatch.setattr(gb, "summarization_node", _mock_node("summarization_agent", summaries=[]))
    monkeypatch.setattr(gb, "report_generator_node", _mock_node("report_generator_agent", final_report="ok"))

    app = gb.build_graph()
    initial_state = {
        "query": "test query", "plan": [], "next_agent": "", "iteration_count": 0,
        "needs_more_research": False, "search_results": [], "extracted_facts": [],
        "fact_check_results": [], "summaries": [], "errors": [], "final_report": None,
        "model_usage_log": [],
    }
    result = app.invoke(initial_state)
    assert result["plan"] == ["test query"]
    assert result["final_report"] == "ok"


def test_graph_respects_max_iterations(monkeypatch):
    """If a node never logs to model_usage_log (simulating a broken agent that
    doesn't report completion), the Supervisor should still terminate via the
    loop guard rather than looping forever."""
    import graph.build_graph as gb
    broken_node = lambda state: {}  # never logs, so supervisor thinks nothing ran

    monkeypatch.setattr(sup, "generate_plan", lambda q: ["sub-q 1"])
    monkeypatch.setattr(gb, "web_search_node", broken_node)
    monkeypatch.setattr(gb, "data_extraction_node", broken_node)
    monkeypatch.setattr(gb, "fact_checking_node", broken_node)
    monkeypatch.setattr(gb, "summarization_node", broken_node)
    monkeypatch.setattr(gb, "report_generator_node", broken_node)

    app = gb.build_graph()
    initial_state = {
        "query": "test query", "plan": [], "next_agent": "", "iteration_count": 0,
        "needs_more_research": False, "search_results": [], "extracted_facts": [],
        "fact_check_results": [], "summaries": [], "errors": [], "final_report": None,
        "model_usage_log": [],
    }
    result = app.invoke(initial_state)
    assert result["iteration_count"] <= sup.MAX_ITERATIONS + 1
    assert result["final_report"] is None  # correctly never reached report generation

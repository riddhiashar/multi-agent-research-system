from langgraph.checkpoint.memory import MemorySaver
import agents.supervisor_agent as sup


def test_graph_pauses_before_report_generator(monkeypatch):
    import graph.build_graph as gb

    monkeypatch.setattr(sup, "generate_plan", lambda q: ["sub-q 1"])
    monkeypatch.setattr(gb, "search_single_node", lambda state: {
        "search_results": [{"sub_question": state["query"], "status": "ok"}],
        "model_usage_log": [{"agent": "web_search_agent"}]})
    monkeypatch.setattr(gb, "data_extraction_node", lambda state: {
        "extracted_facts": [{"claim": "x"}], "model_usage_log": [{"agent": "data_extraction_agent"}]})
    monkeypatch.setattr(gb, "fact_checking_node", lambda state: {
        "fact_check_results": [{"claim": "x", "status": "CONFIRMED"}],
        "model_usage_log": [{"agent": "fact_checking_agent"}]})
    monkeypatch.setattr(gb, "summarization_node", lambda state: {
        "summaries": [{"sub_question": "q1", "summary": "a summary"}],
        "model_usage_log": [{"agent": "summarization_agent"}]})
    monkeypatch.setattr(gb, "report_generator_node", lambda state: {
        "final_report": "the final report", "model_usage_log": [{"agent": "report_generator_agent"}]})

    checkpointer = MemorySaver()
    app = gb.build_graph(checkpointer=checkpointer, interrupt_before=["report_generator"])
    config = {"configurable": {"thread_id": "test-thread-1"}}

    initial_state = {
        "query": "test query", "plan": [], "next_agent": "", "iteration_count": 0,
        "needs_more_research": False, "search_results": [], "extracted_facts": [],
        "fact_check_results": [], "summaries": [], "errors": [], "final_report": None,
        "model_usage_log": [],
    }
    result = app.invoke(initial_state, config=config)

    state = app.get_state(config)
    assert state.next == ("report_generator",)
    assert result.get("final_report") is None  # confirms it paused BEFORE report generation
    assert len(state.values["summaries"]) == 1  # but summaries ARE ready to review

    final = app.invoke(None, config=config)
    assert final["final_report"] == "the final report"

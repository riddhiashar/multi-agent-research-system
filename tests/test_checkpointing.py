import os
import sqlite3
import agents.supervisor_agent as sup


def test_state_persists_across_separate_connections(tmp_path, monkeypatch):
    import graph.build_graph as gb
    from langgraph.checkpoint.sqlite import SqliteSaver

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
        "final_report": "the report", "model_usage_log": [{"agent": "report_generator_agent"}]})

    db_path = str(tmp_path / "test_checkpoints.db")
    config = {"configurable": {"thread_id": "test-thread"}}
    initial_state = {
        "query": "test", "plan": [], "next_agent": "", "iteration_count": 0,
        "needs_more_research": False, "search_results": [], "extracted_facts": [],
        "fact_check_results": [], "summaries": [], "errors": [], "final_report": None,
        "model_usage_log": [],
    }

    conn1 = sqlite3.connect(db_path, check_same_thread=False)
    cp1 = SqliteSaver(conn1)
    cp1.setup()
    app1 = gb.build_graph(checkpointer=cp1, interrupt_before=["report_generator"])
    app1.invoke(initial_state, config=config)
    conn1.close()

    conn2 = sqlite3.connect(db_path, check_same_thread=False)
    cp2 = SqliteSaver(conn2)
    app2 = gb.build_graph(checkpointer=cp2, interrupt_before=["report_generator"])

    recovered = app2.get_state(config)
    assert recovered.next == ("report_generator",)
    assert len(recovered.values["summaries"]) == 1

    final = app2.invoke(None, config=config)
    assert final["final_report"] == "the report"

import agents.summarization_agent as sa


def test_summarize_empty_facts_no_llm_call(monkeypatch):
    called = {"value": False}
    def fake_call(p):
        called["value"] = True
        return "should not be called"
    monkeypatch.setattr(sa, "call_llm", fake_call)
    result = sa.summarize_sub_topic("q", [])
    assert result["status"] == "no_facts"
    assert called["value"] is False


def test_summarize_llm_failure_does_not_crash(monkeypatch):
    def boom(p):
        raise RuntimeError("down")
    monkeypatch.setattr(sa, "call_llm", boom)
    facts = [{"claim": "A", "status": "CONFIRMED", "reasoning": "", "sources": []}]
    result = sa.summarize_sub_topic("q", facts)
    assert result["status"] == "failed"
    assert "error" in result["summary"].lower() or "RuntimeError" in result["summary"]


def test_summarize_counts_statuses_correctly(monkeypatch):
    monkeypatch.setattr(sa, "call_llm", lambda p: "a summary")
    facts = [
        {"claim": "A", "status": "CONFIRMED", "reasoning": "", "sources": []},
        {"claim": "B", "status": "CONFIRMED", "reasoning": "", "sources": []},
        {"claim": "C", "status": "DISPUTED", "reasoning": "", "sources": []},
        {"claim": "D", "status": "UNVERIFIED", "reasoning": "", "sources": []},
    ]
    result = sa.summarize_sub_topic("q", facts)
    assert result["fact_counts"] == {"CONFIRMED": 2, "DISPUTED": 1, "UNVERIFIED": 1}


def test_group_by_sub_question():
    facts = [
        {"claim": "A", "sub_question": "q1", "status": "CONFIRMED"},
        {"claim": "B", "sub_question": "q1", "status": "CONFIRMED"},
        {"claim": "C", "sub_question": "q2", "status": "DISPUTED"},
    ]
    grouped = sa._group_by_sub_question(facts)
    assert len(grouped) == 2
    assert len(grouped["q1"]) == 2
    assert len(grouped["q2"]) == 1


def test_node_aggregates_multiple_sub_topics(monkeypatch):
    monkeypatch.setattr(sa, "call_llm", lambda p: "a summary")
    state = {"fact_check_results": [
        {"claim": "A", "sub_question": "q1", "status": "CONFIRMED", "reasoning": "", "sources": []},
        {"claim": "B", "sub_question": "q2", "status": "CONFIRMED", "reasoning": "", "sources": []},
    ]}
    out = sa.summarization_node(state)
    assert len(out["summaries"]) == 2


def test_node_empty_fact_check_results():
    out = sa.summarization_node({"fact_check_results": []})
    assert out["errors"] and "summaries" not in out

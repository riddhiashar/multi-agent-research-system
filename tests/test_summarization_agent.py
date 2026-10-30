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


def test_node_aggregates_multiple_sub_topics(monkeypatch):
    monkeypatch.setattr(sa, "call_llm", lambda p: "a summary")
    state = {
        "plan": ["q1", "q2"],
        "fact_check_results": [
            {"claim": "A", "sub_question": "q1", "status": "CONFIRMED", "reasoning": "", "sources": []},
            {"claim": "B", "sub_question": "q2", "status": "CONFIRMED", "reasoning": "", "sources": []},
        ],
    }
    out = sa.summarization_node(state)
    assert len(out["summaries"]) == 2


def test_node_empty_fact_check_results_and_empty_plan():
    out = sa.summarization_node({"fact_check_results": [], "plan": []})
    assert out["errors"] and "summaries" not in out


def test_node_regression_day21_silent_topic_drop(monkeypatch):
    """Regression test for the Day 21 finding: a planned sub-question with
    zero fact_check_results must still appear in summaries, explicitly
    marked as having no information, not silently dropped."""
    monkeypatch.setattr(sa, "call_llm", lambda p: "a real summary")
    state = {
        "plan": ["topic A", "topic B", "topic with zero facts"],
        "fact_check_results": [
            {"claim": "fact 1", "sub_question": "topic A", "status": "CONFIRMED", "reasoning": "", "sources": []},
            {"claim": "fact 2", "sub_question": "topic B", "status": "CONFIRMED", "reasoning": "", "sources": []},
        ],
    }
    out = sa.summarization_node(state)
    assert len(out["summaries"]) == 3
    topics_covered = {s["sub_question"] for s in out["summaries"]}
    assert "topic with zero facts" in topics_covered
    empty_summary = next(s for s in out["summaries"] if s["sub_question"] == "topic with zero facts")
    assert empty_summary["status"] == "no_facts"

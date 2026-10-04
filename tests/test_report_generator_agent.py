import agents.report_generator_agent as rga


def test_generate_report_no_summaries():
    result = rga.generate_report("q", [], [])
    assert result["method"] == "no_data"
    assert "No verified information" in result["report"]


def test_generate_report_llm_success(monkeypatch):
    monkeypatch.setattr(rga, "call_llm", lambda p: "A polished narrative.")
    summaries = [{"sub_question": "q1", "summary": "brief 1"}]
    facts = [{"claim": "A", "status": "CONFIRMED", "reasoning": "", "sources": []}]
    result = rga.generate_report("q", summaries, facts)
    assert result["method"] == "llm"
    assert "A polished narrative." in result["report"]
    assert "Confidence Appendix" in result["report"]


def test_generate_report_llm_failure_falls_back(monkeypatch):
    def boom(p):
        raise RuntimeError("down")
    monkeypatch.setattr(rga, "call_llm", boom)
    summaries = [{"sub_question": "q1", "summary": "brief 1"}]
    result = rga.generate_report("q", summaries, [])
    assert result["method"] == "fallback_template"
    assert "brief 1" in result["report"]


def test_appendix_lists_disputed_and_unverified():
    facts = [
        {"claim": "Disputed one", "status": "DISPUTED", "reasoning": "x", "sources": []},
        {"claim": "Unverified one", "status": "UNVERIFIED", "reasoning": "y", "sources": []},
        {"claim": "Confirmed one", "status": "CONFIRMED", "reasoning": "z", "sources": []},
    ]
    appendix = rga._build_confidence_appendix(facts)
    assert "Disputed one" in appendix
    assert "Unverified one" in appendix
    assert "**Confirmed:** 1" in appendix  # matches the actual bold-markdown format produced


def test_appendix_empty_facts():
    appendix = rga._build_confidence_appendix([])
    assert "No fact-checked claims" in appendix


def test_node_empty_summaries():
    out = rga.report_generator_node({"query": "q", "summaries": [], "fact_check_results": []})
    assert out["errors"] and "final_report" not in out


def test_node_success(monkeypatch):
    monkeypatch.setattr(rga, "call_llm", lambda p: "narrative text")
    state = {
        "query": "q",
        "summaries": [{"sub_question": "q1", "summary": "brief"}],
        "fact_check_results": [{"claim": "A", "status": "CONFIRMED", "reasoning": "", "sources": []}],
    }
    out = rga.report_generator_node(state)
    assert "final_report" in out
    assert "narrative text" in out["final_report"]

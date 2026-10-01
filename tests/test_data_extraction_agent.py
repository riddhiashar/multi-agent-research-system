import agents.data_extraction_agent as dea


def test_extract_facts_valid_json(monkeypatch):
    monkeypatch.setattr(dea, "call_llm", lambda p: '["Fact A.", "Fact B."]')
    facts = dea.extract_facts_from_result({"sub_question": "q", "answer": "some text", "sources": []})
    assert len(facts) == 2
    assert facts[0]["claim"] == "Fact A."


def test_extract_facts_handles_markdown_fence(monkeypatch):
    monkeypatch.setattr(dea, "call_llm", lambda p: '```json\n["Fact A."]\n```')
    facts = dea.extract_facts_from_result({"sub_question": "q", "answer": "some text", "sources": []})
    assert len(facts) == 1


def test_extract_facts_malformed_json_returns_empty(monkeypatch):
    monkeypatch.setattr(dea, "call_llm", lambda p: "this is not json at all")
    facts = dea.extract_facts_from_result({"sub_question": "q", "answer": "some text", "sources": []})
    assert facts == []


def test_extract_facts_empty_answer_skips_llm_call(monkeypatch):
    called = {"value": False}
    def fake_call(p):
        called["value"] = True
        return "[]"
    monkeypatch.setattr(dea, "call_llm", fake_call)
    facts = dea.extract_facts_from_result({"sub_question": "q", "answer": "", "sources": []})
    assert facts == [] and called["value"] is False


def test_extract_facts_llm_exception_returns_empty(monkeypatch):
    def boom(p):
        raise RuntimeError("API down")
    monkeypatch.setattr(dea, "call_llm", boom)
    facts = dea.extract_facts_from_result({"sub_question": "q", "answer": "text", "sources": []})
    assert facts == []


def test_node_aggregates_across_results(monkeypatch):
    monkeypatch.setattr(dea, "call_llm", lambda p: '["Fact X."]')
    state = {"search_results": [
        {"sub_question": "q1", "answer": "text1", "sources": []},
        {"sub_question": "q2", "answer": "text2", "sources": []},
    ]}
    out = dea.data_extraction_node(state)
    assert len(out["extracted_facts"]) == 2


def test_node_empty_search_results(monkeypatch):
    out = dea.data_extraction_node({"search_results": []})
    assert out["errors"] and "extracted_facts" not in out

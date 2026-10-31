import agents.data_extraction_agent as dea

SOURCES = [
    {"id": 1, "title": "A", "url": "http://a.com/1"},
    {"id": 2, "title": "B", "url": "http://b.com/2"},
]
MARKED = "[S1] first snippet text. [S2] second snippet text."


def _extract(monkeypatch, llm_output, answer=MARKED):
    monkeypatch.setattr(dea, "call_llm", lambda p: llm_output)
    return dea.extract_facts_from_result({"sub_question": "q", "answer": answer, "sources": SOURCES})


def test_per_claim_attribution_keeps_only_cited_sources(monkeypatch):
    facts = _extract(monkeypatch, '[{"claim": "X", "sources": [2]}]')
    assert facts[0]["sources"] == [SOURCES[1]]
    assert facts[0]["attribution"] == "per_claim"


def test_unknown_source_id_is_unattributed(monkeypatch):
    facts = _extract(monkeypatch, '[{"claim": "X", "sources": [9]}]')
    assert facts[0]["sources"] == []
    assert facts[0]["attribution"] == "unattributed"


def test_plain_string_output_on_tagged_text_is_unattributed(monkeypatch):
    facts = _extract(monkeypatch, '["X"]')
    assert facts[0]["sources"] == []
    assert facts[0]["attribution"] == "unattributed"


def test_untagged_text_shares_all_sources(monkeypatch):
    facts = _extract(monkeypatch, '["X"]', answer="plain synthesized answer with no tags")
    assert facts[0]["sources"] == SOURCES
    assert facts[0]["attribution"] == "shared"


def test_attribution_flag_off_shares_all_sources(monkeypatch):
    monkeypatch.setattr(dea, "ATTRIBUTE_CLAIMS", False)
    facts = _extract(monkeypatch, '["X"]')
    assert facts[0]["sources"] == SOURCES
    assert facts[0]["attribution"] == "shared"

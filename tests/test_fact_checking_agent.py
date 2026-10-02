import json
import agents.fact_checking_agent as fca


def test_batch_parses_valid_response(monkeypatch):
    monkeypatch.setattr(fca, "call_llm", lambda p: json.dumps([
        {"index": 0, "status": "CONFIRMED", "reasoning": "two sources agree"},
        {"index": 1, "status": "DISPUTED", "reasoning": "contradicts claim 0"},
    ]))
    facts = [{"claim": "A", "sources": []}, {"claim": "B", "sources": []}]
    checked = fca.fact_check_batch(facts)
    assert checked[0]["status"] == "CONFIRMED"
    assert checked[1]["status"] == "DISPUTED"


def test_batch_defaults_to_unverified_on_llm_failure(monkeypatch):
    def boom(p):
        raise RuntimeError("down")
    monkeypatch.setattr(fca, "call_llm", boom)
    facts = [{"claim": "A", "sources": []}]
    checked = fca.fact_check_batch(facts)
    assert checked[0]["status"] == "UNVERIFIED"


def test_batch_defaults_to_unverified_on_malformed_json(monkeypatch):
    monkeypatch.setattr(fca, "call_llm", lambda p: "not json")
    facts = [{"claim": "A", "sources": []}]
    checked = fca.fact_check_batch(facts)
    assert checked[0]["status"] == "UNVERIFIED"


def test_batch_fills_missing_index_as_unverified(monkeypatch):
    monkeypatch.setattr(fca, "call_llm", lambda p: json.dumps([
        {"index": 0, "status": "CONFIRMED", "reasoning": "ok"}
    ]))
    facts = [{"claim": "A", "sources": []}, {"claim": "B", "sources": []}]
    checked = fca.fact_check_batch(facts)
    assert checked[0]["status"] == "CONFIRMED"
    assert checked[1]["status"] == "UNVERIFIED"


def test_empty_facts_returns_empty(monkeypatch):
    assert fca.fact_check_facts([]) == []


def test_chunking_splits_large_batches(monkeypatch):
    calls = {"count": 0}
    def fake_call(p):
        calls["count"] += 1
        return json.dumps([{"index": i, "status": "UNVERIFIED", "reasoning": "x"}
                           for i in range(fca.MAX_FACTS_PER_BATCH)])
    monkeypatch.setattr(fca, "call_llm", fake_call)
    facts = [{"claim": f"fact {i}", "sources": []} for i in range(fca.MAX_FACTS_PER_BATCH + 5)]
    checked = fca.fact_check_facts(facts)
    assert len(checked) == fca.MAX_FACTS_PER_BATCH + 5
    assert calls["count"] == 2


def test_node_empty_extracted_facts(monkeypatch):
    out = fca.fact_checking_node({"extracted_facts": []})
    assert out["errors"] and "fact_check_results" not in out

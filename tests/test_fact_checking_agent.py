import json
import agents.fact_checking_agent as fca

TWO_INDEPENDENT = [{"url": "http://a.com/1"}, {"url": "http://b.com/2"}]
ONE_SOURCE = [{"url": "http://a.com/1"}]
SAME_DOMAIN = [{"url": "http://a.com/1"}, {"url": "http://a.com/2"}]


def _verdict_llm(status):
    return lambda p: json.dumps([{"index": 0, "status": status, "reasoning": "model said so"}])


def test_batch_parses_valid_response(monkeypatch):
    monkeypatch.setattr(fca, "call_llm", lambda p: json.dumps([
        {"index": 0, "status": "CONFIRMED", "reasoning": "two sources agree"},
        {"index": 1, "status": "DISPUTED", "reasoning": "contradicts claim 0"},
    ]))
    facts = [{"claim": "A", "sources": TWO_INDEPENDENT}, {"claim": "B", "sources": TWO_INDEPENDENT}]
    checked = fca.fact_check_batch(facts)
    assert checked[0]["status"] == "CONFIRMED"
    assert checked[1]["status"] == "DISPUTED"


def test_batch_defaults_to_unverified_on_llm_failure(monkeypatch):
    def boom(p):
        raise RuntimeError("down")
    monkeypatch.setattr(fca, "call_llm", boom)
    checked = fca.fact_check_batch([{"claim": "A", "sources": TWO_INDEPENDENT}])
    assert checked[0]["status"] == "UNVERIFIED"


def test_batch_defaults_to_unverified_on_malformed_json(monkeypatch):
    monkeypatch.setattr(fca, "call_llm", lambda p: "not json")
    checked = fca.fact_check_batch([{"claim": "A", "sources": TWO_INDEPENDENT}])
    assert checked[0]["status"] == "UNVERIFIED"


def test_batch_fills_missing_index_as_unverified(monkeypatch):
    monkeypatch.setattr(fca, "call_llm", lambda p: json.dumps([
        {"index": 0, "status": "CONFIRMED", "reasoning": "ok"}
    ]))
    facts = [{"claim": "A", "sources": TWO_INDEPENDENT}, {"claim": "B", "sources": TWO_INDEPENDENT}]
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


def test_gate_downgrades_confirmed_with_one_independent_source(monkeypatch):
    monkeypatch.setattr(fca, "call_llm", _verdict_llm("CONFIRMED"))
    checked = fca.fact_check_batch([{"claim": "A", "sources": ONE_SOURCE}])
    assert checked[0]["status"] == "UNVERIFIED"
    assert checked[0]["downgraded"] is True
    assert checked[0]["independent_sources"] == 1


def test_gate_keeps_confirmed_with_two_independent_sources(monkeypatch):
    monkeypatch.setattr(fca, "call_llm", _verdict_llm("CONFIRMED"))
    checked = fca.fact_check_batch([{"claim": "A", "sources": TWO_INDEPENDENT}])
    assert checked[0]["status"] == "CONFIRMED"
    assert checked[0]["downgraded"] is False
    assert checked[0]["independent_sources"] == 2


def test_gate_does_not_touch_disputed(monkeypatch):
    monkeypatch.setattr(fca, "call_llm", _verdict_llm("DISPUTED"))
    checked = fca.fact_check_batch([{"claim": "A", "sources": ONE_SOURCE}])
    assert checked[0]["status"] == "DISPUTED"
    assert checked[0]["downgraded"] is False


def test_gate_same_domain_sources_are_not_independent(monkeypatch):
    monkeypatch.setattr(fca, "call_llm", _verdict_llm("CONFIRMED"))
    checked = fca.fact_check_batch([{"claim": "A", "sources": SAME_DOMAIN}])
    assert checked[0]["status"] == "UNVERIFIED"

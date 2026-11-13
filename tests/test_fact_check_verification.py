import json
import agents.fact_checking_agent as fc

CLAIM = "Hydrometallurgical processing achieves a 95% recovery rate for lithium."


def _facts():
    return [{"claim": CLAIM, "sub_question": "Q1",
             "sources": [{"id": 1, "title": "a", "url": "https://a.org/x"}]}]


def test_verification_upgrades_claim_to_confirmed(monkeypatch):
    monkeypatch.setenv("VERIFY_BY_SEARCH", "true")
    monkeypatch.setattr(fc, "search_candidates", lambda q, exclude_domains=(): [
        {"title": "b", "url": "https://b.com/z", "content": CLAIM}])
    monkeypatch.setattr(fc, "call_llm", lambda p: json.dumps(
        [{"index": 0, "status": "CONFIRMED", "reasoning": "two domains"}]))
    out = fc.fact_checking_node({"extracted_facts": _facts()})
    r = out["fact_check_results"][0]
    assert r["status"] == "CONFIRMED" and r["independent_sources"] == 2 and r["verified_by_search"] == 1
    assert out["model_usage_log"][0]["verify_upgraded"] == 1


def test_verification_off_by_default_never_searches(monkeypatch):
    monkeypatch.delenv("VERIFY_BY_SEARCH", raising=False)
    monkeypatch.setattr(fc, "search_candidates", lambda *a, **k: 1 / 0)
    monkeypatch.setattr(fc, "call_llm", lambda p: json.dumps(
        [{"index": 0, "status": "UNVERIFIED", "reasoning": "one source"}]))
    r = fc.fact_checking_node({"extracted_facts": _facts()})["fact_check_results"][0]
    assert r["status"] == "UNVERIFIED"

import json
import agents.fact_checking_agent as fc

A = "Redwood Materials recovers 95% of cobalt and nickel from used lithium-ion batteries"
A2 = "Redwood Materials recovers 95% of the cobalt and nickel in used lithium-ion batteries"


def _facts():
    return [
        {"claim": A, "sub_question": "Q1", "sources": [{"id": 1, "title": "x", "url": "https://a.com/1"}]},
        {"claim": A2, "sub_question": "Q2", "sources": [{"id": 1, "title": "y", "url": "https://b.org/1"}]},
    ]


def test_merged_claim_is_checked_once_and_confirmed_with_pooled_sources(monkeypatch):
    prompts = []

    def fake(prompt):
        prompts.append(prompt)
        return json.dumps([{"index": 0, "status": "CONFIRMED", "reasoning": "two domains"}])

    monkeypatch.setattr(fc, "call_llm", fake)
    out = fc.fact_checking_node({"extracted_facts": _facts()})["fact_check_results"]
    assert len(prompts) == 1 and "independent sources: 2" in prompts[0]
    assert sorted(r["sub_question"] for r in out) == ["Q1", "Q2"]
    assert all(r["status"] == "CONFIRMED" and r["independent_sources"] == 2 for r in out)


def test_gate_still_downgrades_merged_claim_from_one_domain(monkeypatch):
    facts = _facts()
    facts[1]["sources"][0]["url"] = "https://a.com/2"      # same domain as the first
    monkeypatch.setattr(fc, "call_llm", lambda p: json.dumps(
        [{"index": 0, "status": "CONFIRMED", "reasoning": "x"}]))
    out = fc.fact_checking_node({"extracted_facts": facts})["fact_check_results"]
    assert all(r["status"] == "UNVERIFIED" and r["downgraded"] for r in out)

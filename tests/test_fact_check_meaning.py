import json

import agents.fact_checking_agent as fc

SEARCH = [{"answer": "[S1] Tesla recycles 92% of battery materials at Nevada. [S2] CATL recycles 92% of battery materials. "
                     "[S3] Tesla recycles 70% of battery materials according to one report.",
           "sources": [{"id": 1, "title": "A", "url": "https://a.org/1"}, {"id": 2, "title": "B", "url": "https://b.com/2"},
                       {"id": 3, "title": "C", "url": "https://c.net/3"}]}]
FACT = {"claim": "Tesla recycles 92% of battery materials.", "sub_question": "Q1",
        "sources": [{"id": 1, "title": "A", "url": "https://a.org/1"}, {"id": 2, "title": "B", "url": "https://b.com/2"}]}
SAYS_CONFIRMED = lambda p: json.dumps([{"index": 0, "status": "CONFIRMED", "reasoning": "two domains"}])


def run(monkeypatch, **env):
    for k in ("EVIDENCE_GATE", "DISPUTE_ON_CONFLICT", "VERIFY_BY_SEARCH"):
        monkeypatch.delenv(k, raising=False)
    for k, v in env.items():
        monkeypatch.setenv(k, v)
    monkeypatch.setattr(fc, "call_llm", SAYS_CONFIRMED)
    return fc.fact_checking_node({"query": "Compare Tesla and CATL recycling", "extracted_facts": [dict(FACT)],
                                  "search_results": SEARCH})


def test_default_mode_keeps_domain_rule_but_records_that_one_citation_is_wrong(monkeypatch):
    out = run(monkeypatch)
    r = out["fact_check_results"][0]
    assert r["status"] == "CONFIRMED" and r["independent_sources"] == 2 and r["supported_domains"] == 1
    assert out["model_usage_log"][0]["citations_supporting"] == 1 and out["model_usage_log"][0]["citations_checkable"] == 2


def test_strict_mode_downgrades_when_a_cited_domain_does_not_state_the_claim(monkeypatch):
    r = run(monkeypatch, EVIDENCE_GATE="strict")["fact_check_results"][0]
    assert r["status"] == "UNVERIFIED" and r["downgraded"] and "only 1 of 2 cited domains" in r["reasoning"]


def test_conflict_is_recorded_and_only_disputes_when_enabled(monkeypatch):
    r = run(monkeypatch)["fact_check_results"][0]
    assert r["conflicting_evidence"][0]["domain"] == "c.net" and r["status"] == "CONFIRMED"
    d = run(monkeypatch, DISPUTE_ON_CONFLICT="true")["fact_check_results"][0]
    assert d["status"] == "DISPUTED" and "c.net" in d["reasoning"]


def test_prompt_shows_the_model_the_evidence_and_the_conflict(monkeypatch):
    seen = {}
    monkeypatch.setattr(fc, "call_llm", lambda p: seen.setdefault("p", p) and SAYS_CONFIRMED(p))
    fc.fact_checking_node({"query": "Tesla and CATL", "extracted_facts": [dict(FACT)], "search_results": SEARCH})
    assert "supports (a.org)" in seen["p"] and "CONFLICT (c.net)" in seen["p"] and "figures, units" in seen["p"]


def test_conflict_lines_are_informational_unless_dispute_mode_is_on(monkeypatch):
    seen = {}
    for env, want in (("false", "do NOT mark a claim DISPUTED because of a CONFLICT line"), ("true", "about the SAME fact")):
        monkeypatch.setenv("DISPUTE_ON_CONFLICT", env)
        monkeypatch.setattr(fc, "call_llm", lambda p: seen.setdefault("p", p) and SAYS_CONFIRMED(p))
        seen.clear()
        fc.fact_checking_node({"query": "Tesla and CATL", "extracted_facts": [dict(FACT)], "search_results": SEARCH})
        assert want in seen["p"]

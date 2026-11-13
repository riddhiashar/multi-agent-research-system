from tools.evidence_check import (attach_citation_checks, attach_conflicts, check_citations, citation_stats, find_conflicts,
                                  snippet_records)

SEARCH = [{"answer": "[S1] Tesla recycles 92% of battery materials at Nevada. [S2] CATL recycles 92% of battery materials. "
                     "[S3] Tesla recycles 70% of battery materials according to one report.",
           "sources": [{"id": 1, "title": "A", "url": "https://a.org/1"}, {"id": 2, "title": "B", "url": "https://b.com/2"},
                       {"id": 3, "title": "C", "url": "https://c.net/3"}]}]
FACT = {"claim": "Tesla recycles 92% of battery materials.",
        "sources": [{"id": 1, "title": "A", "url": "https://a.org/1"}, {"id": 2, "title": "B", "url": "https://b.com/2"}]}
ENTS = {"tesla", "catl"}


def test_records_come_from_tagged_snippets():
    recs = snippet_records(SEARCH)
    assert len(recs) == 3 and recs[0]["url"] == "https://a.org/1" and recs[0]["text"].startswith("Tesla")


def test_two_cited_domains_but_only_one_states_the_claim():
    out = check_citations(FACT, snippet_records(SEARCH), ENTS)
    rels = {c["domain"]: c["relation"] for c in out["citation_checks"]}
    assert rels == {"a.org": "supports", "b.com": "unrelated"}      # b.com is about CATL, not Tesla
    assert out["supported_domains"] == 1
    assert out["sources"][0]["evidence"].startswith("Tesla recycles 92%")


def test_missing_snippet_is_unavailable_not_failed():
    f = {"claim": FACT["claim"], "sources": [{"url": "https://zzz.org/x"}]}
    out = check_citations(f, snippet_records(SEARCH), ENTS)
    assert out["citation_checks"][0]["relation"] == "unavailable" and out["supported_domains"] == 0


def test_verification_sources_count_as_supported():
    f = {"claim": FACT["claim"], "sources": [{"url": "https://v.org/x", "via": "verification", "evidence": "Tesla recycles 92%.",
                                              "relation": "supports"}]}
    assert check_citations(f, [], ENTS)["supported_domains"] == 1


def test_conflict_scan_finds_contradiction_in_an_uncited_source():
    out = find_conflicts(FACT, snippet_records(SEARCH), ENTS)
    c = out["conflicting_evidence"]
    assert len(c) == 1 and c[0]["domain"] == "c.net" and "70%" in c[0]["evidence"] and not c[0]["cited_by_claim"]


def test_attach_and_stats():
    facts = attach_conflicts(attach_citation_checks([FACT], snippet_records(SEARCH), ENTS), snippet_records(SEARCH), ENTS)
    s = citation_stats(facts)
    assert s == {"cited": 2, "checkable": 2, "supporting": 1, "contradicting": 0, "claims_with_conflicts": 1}

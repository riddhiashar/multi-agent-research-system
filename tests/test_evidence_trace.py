import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from snap_fixture import CLAIM_A, CLAIM_B, RESULTS, SEARCH, make_snap
from tools.evidence_trace import (category_counts, claim_funnel, possible_shared_origin, snippet_pool, trace_all,
                                  trace_claim, verdict_category, verdict_explanation)


def test_snippet_pool_maps_urls_to_their_tagged_text():
    pool = snippet_pool(SEARCH)
    assert "95% recovery" in pool["https://www.alpha.org/a"][0]["text"]
    assert "Unrelated" not in pool["https://www.alpha.org/a"][0]["text"]
    assert "https://beta.com/b" in pool


def test_original_source_gets_exact_sentence_and_verification_source_keeps_its_own():
    t = trace_claim(RESULTS[0], snippet_pool(SEARCH))
    orig, ver = t["sources"]
    assert orig["origin"] == "search" and orig["match"] == "exact" and "95%" in orig["evidence"]
    assert orig["domain"] == "alpha.org"
    assert ver["origin"] == "verification" and ver["evidence"].startswith("Plants reach")
    assert (t["original_sources"], t["verification_sources"]) == (1, 1)


def test_missing_snippet_is_marked_unavailable_not_invented():
    t = trace_claim(RESULTS[3], snippet_pool(SEARCH))
    assert all(s["match"] == "unavailable" and s["evidence"] == "" for s in t["sources"])


def test_closest_sentence_is_labelled_closest():
    fact = {"claim": "Pyrometallurgy recovers lithium efficiently overall.", "sources": [{"url": "https://gamma.io/c"}]}
    s = trace_claim(fact, snippet_pool(SEARCH))["sources"][0]
    assert s["match"] in ("closest", "none")   # never "exact" for a claim the snippet does not state


def test_verdict_categories_separate_the_three_kinds_of_unverified():
    cats = [verdict_category(r) for r in RESULTS]
    assert cats == ["confirmed", "insufficient_sources", "failed", "inconclusive"]
    assert category_counts(RESULTS)["failed"] == 1


def test_verification_search_failure_is_a_failure_not_insufficient():
    fact = {"claim": CLAIM_B, "status": "UNVERIFIED", "independent_sources": 1, "verification": "search_failed", "reasoning": "x"}
    assert verdict_category(fact) == "failed"
    assert "errored" in verdict_explanation(fact)


def test_explanations_state_what_was_tried():
    assert "no second independent source" in verdict_explanation(RESULTS[1])
    assert "budget" in verdict_explanation({"claim": "c", "status": "UNVERIFIED", "independent_sources": 1, "sources": []})


def test_shared_origin_flags_near_identical_sentences_on_different_domains():
    same = "The IEA says recycling will meet 20 percent of battery demand by 2030."
    traced = [{"domain": "a.com", "evidence": same, "match": "exact"},
              {"domain": "b.org", "evidence": same + " ", "match": "exact"}]
    assert possible_shared_origin(traced)
    traced[1]["evidence"] = "Totally different wording about lithium prices rising."
    assert not possible_shared_origin(traced)


def test_funnel_numbers_come_from_run_data():
    f = claim_funnel(5, RESULTS, make_snap()["usage_log"])
    assert f["extracted"] == 5 and f["rows"] == 4 and f["distinct"] == 4 and f["merged_away"] == 1
    assert f["verification_tried"] == 2 and f["verification_upgraded"] == 1 and f["verification_recorded"]
    assert f["initially_under_supported"] == 3   # A started with 1 original domain; B, C have 1; D has 2


def test_trace_all_handles_runs_without_saved_snippets():
    assert len(trace_all(RESULTS, [])) == 4


def test_cited_but_not_supporting_gets_its_own_category():
    fact = {"claim": "Tesla recycles 92% of battery materials.", "status": "UNVERIFIED", "independent_sources": 2,
            "supported_domains": 1, "reasoning": "[downgraded: only 1 of 2 cited domains have a sentence that states the claim]",
            "citation_checks": [{"url": "u1", "domain": "a.org", "relation": "supports"},
                                {"url": "u2", "domain": "b.com", "relation": "unrelated"}], "sources": []}
    assert verdict_category(fact) == "citations_not_supporting"
    assert "only 1 have a sentence that states the claim" in verdict_explanation(fact)
    fact["citation_checks"] = [{"url": "u1", "domain": "a.org", "relation": "unavailable"}]
    assert verdict_category(fact) == "inconclusive"      # nothing could be checked, so don't blame the citations


def test_trace_marks_a_contradicting_snippet():
    search = [{"answer": "[S1] Hydrometallurgical recycling reaches 70% recovery of nickel and cobalt.",
               "sources": [{"id": 1, "title": "Alpha", "url": "https://www.alpha.org/a"}]}]
    fact = {"claim": CLAIM_A, "status": "UNVERIFIED", "sources": [{"title": "Alpha", "url": "https://www.alpha.org/a"}]}
    s = trace_claim(fact, snippet_pool(search))["sources"][0]
    assert s["match"] == "contradicts" and "figure differs" in s["reasons"][0]

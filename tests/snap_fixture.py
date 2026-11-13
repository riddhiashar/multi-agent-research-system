"""A small but realistic run snapshot shared by the Day 27 tests."""
import copy

CLAIM_A = "Recovery rates for nickel and cobalt reach 95% in hydrometallurgical recycling."
CLAIM_B = "Pyrometallurgy recovers lithium poorly."
CLAIM_C = "Redwood Materials supplies cathode material to Panasonic."
CLAIM_D = "CATL operates battery recycling in China."

SEARCH = [
    {"sub_question": "Q1?", "provider": "tavily",
     "answer": "[S1] Hydrometallurgical recycling lets plants reach 95% recovery of nickel and cobalt. [S2] Unrelated text about weather.",
     "sources": [{"id": 1, "title": "Alpha", "url": "https://www.alpha.org/a"},
                 {"id": 2, "title": "Beta", "url": "https://beta.com/b"}]},
    {"sub_question": "Q2?", "provider": "duckduckgo",
     "answer": "[S1] Pyrometallurgy has poor lithium recovery. [S2] Redwood supplies cathode material to Panasonic.",
     "sources": [{"id": 1, "title": "Gamma", "url": "https://gamma.io/c"},
                 {"id": 2, "title": "Delta", "url": "https://delta.net/d"}]},
]

RESULTS = [
    {"claim": CLAIM_A, "status": "CONFIRMED", "sub_question": "Q1?", "independent_sources": 2, "verified_by_search": 1,
     "verification": "added", "merged_count": 1, "reasoning": "two domains",
     "sources": [{"id": 1, "title": "Alpha", "url": "https://www.alpha.org/a"},
                 {"id": 2, "title": "Zeta", "url": "https://zeta.com/z", "via": "verification",
                  "evidence": "Plants reach 95% recovery of nickel and cobalt."}]},
    {"claim": CLAIM_B, "status": "UNVERIFIED", "sub_question": "Q2?", "independent_sources": 1, "verification": "no_match",
     "reasoning": "single source", "sources": [{"id": 1, "title": "Gamma", "url": "https://gamma.io/c"}]},
    {"claim": CLAIM_C, "status": "UNVERIFIED", "sub_question": "Q2?", "independent_sources": 1,
     "reasoning": "fact-check call failed; defaulted to unverified",
     "sources": [{"id": 2, "title": "Delta", "url": "https://delta.net/d"}]},
    {"claim": CLAIM_D, "status": "UNVERIFIED", "sub_question": "Q1?", "independent_sources": 2,
     "reasoning": "model was unsure", "sources": [{"title": "E", "url": "https://e.org/e"}, {"title": "F", "url": "https://f.org/f"}]},
]

SNAP = {
    "query": "Compare recycling approaches", "created_at": "2026-11-03T18:00:00", "final_report": "# Report\n\nBody **bold** text.",
    "fact_check_results": RESULTS,
    "summaries": [{"sub_question": "Q1?", "summary": "Q1 summary.", "status": "ok"},
                  {"sub_question": "Q2?", "summary": "Summary could not be generated (RateLimit)", "status": "failed"}],
    "search_results": SEARCH,
    "usage_log": [{"agent": "fact_checking_agent", "facts_checked": 4, "facts_in": 5, "verify_tried": 2,
                   "verify_upgraded": 1, "verify_sources_added": 1, "verify_failed": 0}],
    "state_errors": [],
    "metrics": {"total_seconds": 100.0, "total_output_tokens": 500, "peak_output_tokens_60s": 400, "rate_limit_errors": 1,
                "nodes": {"data_extraction": {"calls": 2, "input": 1000, "output": 300, "rate_limited": 1, "other_errors": 0},
                          "summarization": {"calls": 2, "input": 800, "output": 200, "rate_limited": 0, "other_errors": 0}}},
    "evidence": {"extracted_claims": 5, "claims_verified_by_search": 1},
    "cache": {"hits": 3, "misses": 2}, "wall_seconds": 101.0,
}


def make_snap():
    return copy.deepcopy(SNAP)

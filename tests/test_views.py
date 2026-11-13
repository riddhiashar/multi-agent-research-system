from ui.views import (claim_rows, confirmation_rate, filter_rows, node_rows, short_topic,
                      status_by_topic, status_counts, topic_options)

RESULTS = [
    {"claim": "Hydrometallurgy recovers 95% of lithium", "status": "CONFIRMED", "sub_question": "How do processes work in recycling?",
     "independent_sources": 2, "sources": [{"url": "https://a.org"}, {"url": "https://b.com", "via": "verification"}],
     "verified_by_search": 1},
    {"claim": "Redwood is in Nevada", "status": "UNVERIFIED", "sub_question": "How do processes work in recycling?",
     "independent_sources": 1, "sources": [{"url": "https://a.org"}]},
    {"claim": "China hosts 85% of capacity", "status": "UNVERIFIED", "sub_question": "What do regulations say?",
     "independent_sources": 1, "sources": []},
    {"claim": "Rules conflict", "status": "DISPUTED", "sub_question": "What do regulations say?", "merged_count": 2},
]


def test_counts_and_rate():
    assert status_counts(RESULTS) == {"CONFIRMED": 1, "DISPUTED": 1, "UNVERIFIED": 2}
    assert confirmation_rate(RESULTS) == 0.25 and confirmation_rate([]) == 0.0


def test_status_by_topic_has_one_row_per_subquestion():
    rows = status_by_topic(RESULTS)
    assert [r["topic"][:3] for r in rows] == ["Q1:", "Q2:"]
    assert rows[0]["CONFIRMED"] == 1 and rows[0]["UNVERIFIED"] == 1
    assert rows[1]["DISPUTED"] == 1 and rows[1]["UNVERIFIED"] == 1


def test_filters_combine():
    rows = claim_rows(RESULTS)
    assert len(filter_rows(rows)) == 4
    assert [r["claim"] for r in filter_rows(rows, statuses=["UNVERIFIED"])] == ["Redwood is in Nevada", "China hosts 85% of capacity"]
    assert len(filter_rows(rows, verified_only=True)) == 1
    assert len(filter_rows(rows, topic=topic_options(RESULTS)[1], statuses=["DISPUTED"])) == 1
    assert [r["claim"] for r in filter_rows(rows, text="NEVADA")] == ["Redwood is in Nevada"]


def test_claim_rows_are_defensive_about_missing_fields():
    row = claim_rows([{"claim": "x"}])[0]
    assert row["status"] == "UNVERIFIED" and row["sources"] == [] and row["merged"] == 1 and not row["verified"]


def test_short_topic_and_node_rows():
    assert short_topic("a " * 40, 20).endswith("…") and short_topic("short") == "short"
    rows = node_rows({"nodes": {"data_extraction": {"node_seconds": 58.14, "output": 1937, "calls": 4, "rate_limited": 0},
                                "idle": {}}})
    assert rows == [{"node": "data_extraction", "seconds": 58.1, "output_tokens": 1937, "llm_calls": 4, "rate_limited": 0}]


def test_source_html_escapes_web_content_and_blocks_bad_links():
    from ui.views import source_html
    out = source_html({"url": "javascript:alert(1)", "title": "<img src=x onerror=alert(1)>", "evidence": "a <b>quote</b>"})
    assert 'href="#"' in out and "<img" not in out and "&lt;img" in out and "<b>" not in out
    ok = source_html({"url": "https://a.org/x?q=1&r=2", "title": "A", "via": "verification", "evidence": "the quote"})
    assert 'href="https://a.org/x?q=1&amp;r=2"' in ok and "(verification)" in ok and "the quote" in ok


def test_validate_question():
    from ui.views import validate_question
    assert validate_question("  compare   battery\x00 recycling ")[0] == "compare battery recycling"
    assert validate_question("hi")[1] and validate_question("x" * 501)[1]
    assert validate_question("a perfectly fine research question")[1] is None


def test_source_card_escapes_web_text_and_labels_origin():
    from ui.views import source_card_html
    html = source_card_html({"title": "<script>x</script>", "url": "javascript:alert(1)", "domain": "evil.com",
                             "origin": "verification", "evidence": "a <b>quote</b>", "match": "exact"})
    assert "<script>" not in html and "<b>quote" not in html and 'href="#"' in html
    assert "targeted verification" in html and "states the claim" in html

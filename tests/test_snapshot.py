from ui.snapshot import build_snapshot


def test_snapshot_schema_and_defaults():
    s = build_snapshot("q", {"final_report": "r", "search_results": [{"answer": "a"}, "junk"], "model_usage_log": [{"a": 1}]},
                       metrics={"nodes": {}}, evidence={}, llm_errors=[("n", "other", "boom")], wall_seconds=12.34)
    assert s["query"] == "q" and s["search_results"] == [{"answer": "a"}] and s["usage_log"] == [{"a": 1}]
    assert s["llm_errors"] == [["n", "other", "boom"]] and s["wall_seconds"] == 12.3 and s["fact_check_results"] == []

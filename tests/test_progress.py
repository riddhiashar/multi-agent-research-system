from ui.progress import (PipelineTracker, clip, describe_update, iter_updates,
                         render_pipeline_html, stage_progress)


def test_supervisor_plan_and_routing():
    assert describe_update("supervisor", {"plan": ["a", "b", "c"]}) == "Planned 3 sub-questions"
    assert describe_update("supervisor", {"next_agent": "fact_checking"}) == "Next step: Fact-checking"
    assert describe_update("supervisor", None) == "Supervisor decided the next step"


def test_search_line_reports_provider_and_failures():
    ok = {"search_results": [{"sub_question": "How does X work?", "provider": "tavily", "sources": [1, 2, 3, 4]}]}
    bad = {"search_results": [{"sub_question": "Q", "provider": "none", "sources": [], "status": "failed"}]}
    assert "via tavily (4 sources)" in describe_update("web_search_single", ok)
    assert "found nothing" in describe_update("web_search_single", bad)


def test_fact_check_counts_statuses():
    u = {"fact_check_results": [{"status": "CONFIRMED"}, {"status": "UNVERIFIED"}, {"status": "UNVERIFIED"}]}
    assert describe_update("fact_checking", u) == "Checked 3 claims: 1 confirmed, 0 disputed, 2 unverified"


def test_errors_are_flagged_and_unknown_nodes_fall_back():
    assert "⚠ 1 error(s)" in describe_update("data_extraction", {"extracted_facts": [], "errors": [{"x": 1}]})
    assert describe_update("mystery", {}) == "mystery finished"


def test_progress_fraction_only_for_pipeline_nodes():
    assert stage_progress("supervisor") is None
    assert stage_progress("web_search_single") == 0.2 and stage_progress("report_generator") == 1.0


def test_iter_updates_tolerates_odd_shapes():
    assert iter_updates({"a": {"x": 1}, "b": None}) == [("a", {"x": 1}), ("b", None)]
    assert iter_updates({"a": [{"x": 1}, {"x": 2}]}) == [("a", {"x": 1}), ("a", {"x": 2})]
    assert iter_updates("not a dict") == []


def test_clip_breaks_on_word_boundary():
    assert clip("What are the primary technical processes and recovery rates for lithium", 40) == \
        "What are the primary technical…"
    assert clip("short", 40) == "short"


class FakeClock:
    def __init__(self):
        self.t = 100.0

    def __call__(self):
        return self.t


def test_tracker_walks_through_stages_with_timings():
    clock = FakeClock()
    t = PipelineTracker(clock=clock)
    t.start()
    assert t.stages["web_search_single"]["status"] == "running"

    t.on_update("supervisor", {"plan": ["a", "b"]})
    clock.t += 2
    t.on_update("web_search_single", {"search_results": [{"sub_question": "a", "sources": [1]}]})
    assert t.stages["web_search_single"]["status"] == "running"          # 1 of 2 searches
    assert t.stages["web_search_single"]["detail"] == "1/2 queries"
    clock.t += 1
    t.on_update("web_search_single", {"search_results": [{"sub_question": "b", "sources": [1]}]})
    assert t.stages["web_search_single"]["status"] == "done"
    assert t.stages["web_search_single"]["seconds"] == 3.0
    assert t.stages["data_extraction"]["status"] == "running"            # next stage starts automatically

    clock.t += 10
    t.on_update("data_extraction", {"extracted_facts": [1, 2, 3]})
    assert t.stages["data_extraction"]["seconds"] == 10.0
    assert t.fraction_done() == 0.4
    assert [e["node"] for e in t.events][:2] == ["supervisor", "web_search_single"]


def test_render_html_shows_each_state_and_escapes_text():
    clock = FakeClock()
    t = PipelineTracker(clock=clock)
    t.start()
    t.stages["web_search_single"]["detail"] = "<script>alert(1)</script>"
    html_out = render_pipeline_html(t.snapshot())
    assert "pl-running" in html_out and "pl-pending" in html_out
    assert "<script>alert" not in html_out and "&lt;script&gt;" in html_out

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from snap_fixture import make_snap
from ui.runstats import quality_metrics, run_health, run_stats


def test_run_stats_reuse_backend_numbers():
    s = run_stats(make_snap())
    assert s["llm_calls"] == 4 and s["input_tokens"] == 1800 and s["output_tokens"] == 500
    assert s["failed_llm_calls"] == 1 and s["verification_searches"] == 2
    assert (s["cache_hits"], s["cache_misses"]) == (3, 2) and s["search_requests"] == 2
    assert s["sources_cited"] == 6 and s["sources_retrieved"] >= 6


def test_health_is_partial_when_a_stage_failed():
    h = run_health(make_snap())
    assert h["state"] == "partial"
    assert any("summary" in i for i in h["issues"]) and any("fact-checked" in i for i in h["issues"])


def test_health_complete_and_failed():
    snap = make_snap()
    snap["summaries"][1]["status"] = "ok"
    snap["fact_check_results"][2]["reasoning"] = "fine"
    assert run_health(snap)["state"] == "complete"
    snap["final_report"] = ""
    assert run_health(snap)["state"] == "failed"


def test_old_snapshot_without_new_keys_still_works():
    old = {"query": "q", "final_report": "r", "fact_check_results": [{"claim": "c", "status": "UNVERIFIED"}],
           "metrics": {"nodes": {}}}
    assert run_stats(old)["llm_calls"] == 0 and run_health(old)["state"] == "complete"
    assert quality_metrics(old)["claims"] == 1


def test_quality_metrics():
    q = quality_metrics(make_snap())
    assert q["claims"] == 4 and q["citation_coverage"] == 1.0
    assert q["multi_domain_rate"] == 0.5 and q["unsupported_rate"] == 0.5
    assert q["topics_planned"] == 2 and q["topics_summarised"] == 1

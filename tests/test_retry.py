import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from snap_fixture import make_snap
from ui.retry import count_failures, find_failures, retry_failures


def fake_fact_check(rows):
    return [{**r, "status": "CONFIRMED" if r.get("independent_sources", 0) >= 2 else "UNVERIFIED",
             "reasoning": "re-checked", "downgraded": False} for r in rows]


def fake_summarize(q, rows):
    return {"sub_question": q, "summary": f"fresh summary of {len(rows)} claims", "status": "ok"}


def fake_report(query, summaries, results):
    return {"report": f"REBUILT {len(summaries)} {len(results)}"}


def test_find_failures_lists_each_kind():
    f = find_failures(make_snap())
    assert f["summaries"] == ["Q2?"] and len(f["fact_check"]) == 1 and f["verification"] == []
    assert count_failures(make_snap()) == 2


def test_retry_fixes_only_what_failed_and_rebuilds_report():
    calls = {"fc": [], "sum": []}

    def fc(rows):
        calls["fc"].append(len(rows)); return fake_fact_check(rows)

    def sm(q, rows):
        calls["sum"].append(q); return fake_summarize(q, rows)

    out = retry_failures(make_snap(), summarize_fn=sm, fact_check_fn=fc, report_fn=fake_report)
    snap = out["snapshot"]
    assert calls == {"fc": [1], "sum": ["Q2?"]}              # Q1's good summary and good claims were not redone
    assert snap["summaries"][1]["status"] == "ok" and snap["summaries"][0]["summary"] == "Q1 summary."
    assert snap["final_report"].startswith("REBUILT")
    assert out["fixed"] == 2 and out["remaining"] == 0
    assert snap["retries"][0]["fixed"] == 2


def test_retry_that_fails_again_is_reported_honestly():
    def still_failing(rows):
        return [{**r, "status": "UNVERIFIED", "reasoning": "fact-check call failed; defaulted to unverified"} for r in rows]

    def bad_summary(q, rows):
        return {"sub_question": q, "summary": "x", "status": "failed"}

    out = retry_failures(make_snap(), summarize_fn=bad_summary, fact_check_fn=still_failing, report_fn=fake_report)
    assert out["fixed"] == 0 and out["remaining"] == 2


def test_failed_verification_is_retried_and_rechecked():
    snap = make_snap()
    snap["fact_check_results"][1]["verification"] = "search_failed"

    def verify(row):
        return ({**row, "sources": row["sources"] + [{"url": "https://new.org/n", "via": "verification"}],
                 "verification": "added", "verified_by_search": 1, "independent_sources": 2}, 1, False)

    out = retry_failures(snap, summarize_fn=fake_summarize, fact_check_fn=fake_fact_check, report_fn=fake_report, verify_fn=verify)
    row = out["snapshot"]["fact_check_results"][1]
    assert row["verification"] == "added" and len(row["sources"]) == 2


def test_nothing_to_retry_changes_nothing():
    snap = make_snap()
    snap["summaries"][1]["status"] = "ok"; snap["fact_check_results"][2]["reasoning"] = "ok"
    out = retry_failures(snap, summarize_fn=fake_summarize, fact_check_fn=fake_fact_check, report_fn=fake_report)
    assert out["log"] == ["Nothing needed retrying."] and out["snapshot"]["final_report"] == snap["final_report"]


def test_original_snapshot_is_not_mutated():
    snap = make_snap()
    retry_failures(snap, summarize_fn=fake_summarize, fact_check_fn=fake_fact_check, report_fn=fake_report)
    assert snap["summaries"][1]["status"] == "failed"

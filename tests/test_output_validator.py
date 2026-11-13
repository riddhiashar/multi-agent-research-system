import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from snap_fixture import make_snap
from tools import output_validator as ov

GOOD = ("Hydrometallurgical recycling lets plants reach 95% recovery of nickel and cobalt [1]. "
        "Pyrometallurgy recovers lithium poorly, according to a single source. "
        "Sources include https://www.alpha.org/a and https://beta.com/b.")


def snap_with(report):
    s = make_snap()
    s["final_report"] = report
    s["summaries"] = [{"sub_question": "Q1?", "summary": "Q1 summary.", "status": "ok"}]
    return s


def types(v):
    return [i["type"] for i in v["issues"]]


def test_a_faithful_report_passes():
    v = ov.validate_snapshot(snap_with(GOOD))
    assert v["ok"] and v["issues"] == [], v["issues"]
    assert v["stats"]["urls"] == 2 and ov.summary_line(v) == "Passed"


def test_invented_site_is_an_error_and_known_site_other_page_is_a_warning():
    v = ov.validate_snapshot(snap_with("See https://made-up-journal.org/paper and https://alpha.org/other-page for the data on recycling."))
    by = {i["type"]: i for i in v["issues"]}
    assert by["url_unknown"]["severity"] == "error" and by["url_not_retrieved"]["severity"] == "warning"


def test_unsafe_and_malformed_urls_are_rejected():
    for bad in ("javascript:alert(1)", "http://localhost:8501/x", "http://192.168.1.5/admin", "http://user:pw@a.org/x", "http://nodot/x"):
        assert ov.check_url(bad, ov.build_context(make_snap()))["type"] == "url_invalid", bad


def test_figure_not_in_evidence_is_flagged_with_unit_awareness():
    v = ov.validate_snapshot(snap_with("Plants reach 80% recovery of nickel and cobalt with hydrometallurgical recycling."))
    assert "number_unsupported" in types(v)
    ok = ov.validate_snapshot(snap_with("A 95 GWh facility recycles nickel and cobalt with hydrometallurgical recycling."))
    assert "number_unsupported" in types(ok)                 # 95 GWh is not 95%


def test_small_plain_numbers_and_the_runs_own_counts_are_not_flagged():
    v = ov.validate_snapshot(snap_with("There are 3 findings across 2 sub-topics and 4 claims were checked in this run."))
    assert "number_unsupported" not in types(v)


def test_invented_year_is_a_warning():
    v = ov.validate_snapshot(snap_with("Recycling plants will expand a lot by 2041 according to analysts of the market."))
    assert any(i["type"] == "year_unsupported" and i["severity"] == "warning" for i in v["issues"])


def test_sentence_that_contradicts_a_checked_claim_is_an_error_unless_hedged():
    bad = ov.validate_snapshot(snap_with("Recovery rates for nickel and cobalt reach 70% in hydrometallurgical recycling."))
    assert "claim_contradicts" in types(bad)
    hedged = ov.validate_snapshot(snap_with("One source says recovery rates for nickel and cobalt reach 95% in hydrometallurgical recycling, "
                                            "however another reports 70%, so this is disputed."))
    assert "claim_contradicts" not in types(hedged)


def test_calling_an_unverified_claim_confirmed_is_overstated():
    v = ov.validate_snapshot(snap_with("Redwood Materials supplies cathode material to Panasonic, which is confirmed."))
    assert "claim_overstated" in types(v)
    fine = ov.validate_snapshot(snap_with("It is not confirmed that Redwood Materials supplies cathode material to Panasonic."))
    assert "claim_overstated" not in types(fine)
    true_confirmed = ov.validate_snapshot(snap_with("Recovery rates for nickel and cobalt reach 95% in hydrometallurgical recycling, which is confirmed."))
    assert "claim_overstated" not in types(true_confirmed)


def test_finalize_removes_invented_links_keeps_raw_and_adds_notes():
    raw = ("Plants reach 95% recovery of nickel and cobalt in hydrometallurgical recycling. "
           "See [the paper](https://made-up-journal.org/paper) and https://made-up-journal.org/x for details on the topic. "
           "Redwood Materials supplies cathode material to Panasonic, which is confirmed.")
    snap = ov.finalize_snapshot(snap_with(raw))
    assert "made-up-journal.org" not in snap["final_report"].split("**Automated validation notes**")[0]
    assert "link removed" in snap["final_report"]
    assert snap["final_report_raw"] == raw and snap["validation"]["ok"] is False
    assert "claim overstated" in snap["final_report"]
    again = ov.finalize_snapshot(snap)                          # idempotent: no duplicated footer
    assert again["final_report"].count("Automated validation notes") == 1


def test_finalize_sanitises_html_in_summaries_and_report():
    s = snap_with("Fine text with <script>alert(1)</script> inside about nickel recovery.")
    s["summaries"] = [{"sub_question": "Q1?", "summary": "Summary <iframe src=x></iframe> ok", "status": "ok"}]
    out = ov.finalize_snapshot(s)
    assert "<script" not in out["final_report"] and "<iframe" not in out["summaries"][0]["summary"]
    assert {a["type"] for a in out["validation"]["guardrail_actions"]} >= {"script_or_embed_removed"}


def test_live_url_check_uses_the_injected_fetcher_and_never_blocks():
    codes = {"https://www.alpha.org/a": 200, "https://beta.com/b": 404}
    v = ov.validate_snapshot(snap_with(GOOD), live_fetch=lambda u: codes.get(u))
    bad = [i for i in v["issues"] if i["type"] == "url_unreachable"]
    assert len(bad) == 1 and bad[0]["severity"] == "warning" and v["ok"]
    assert ov.validate_snapshot(snap_with(GOOD), live_fetch=lambda u: 403)["issues"] == []      # blocked bots are not failures


def test_empty_report_is_handled():
    out = ov.finalize_snapshot({**make_snap(), "final_report": ""})
    assert out["final_report"] == "" and out["validation"]["ok"]

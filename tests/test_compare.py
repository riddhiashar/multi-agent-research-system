import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from snap_fixture import CLAIM_A, make_snap
from ui.compare import compare_runs


def test_compare_finds_new_changed_and_gained():
    a, b = make_snap(), make_snap()
    b["fact_check_results"][1]["status"] = "CONFIRMED"; b["fact_check_results"][1]["independent_sources"] = 2
    b["fact_check_results"][3]["independent_sources"] = 3
    b["fact_check_results"].append({"claim": "A brand new finding about direct cathode recycling.", "status": "UNVERIFIED",
                                    "sub_question": "Q1?", "independent_sources": 1, "sources": [{"url": "https://n.org/x"}]})
    del b["fact_check_results"][2]
    c = compare_runs(a, b)
    assert c["same_question"]
    assert [x["claim"][:20] for x in c["new_claims"]] == ["A brand new finding "]
    assert len(c["dropped_claims"]) == 1
    assert c["status_changed"][0]["from"] == "UNVERIFIED" and c["status_changed"][0]["to"] == "CONFIRMED"
    assert c["gained_evidence"][0]["to"] == 3
    assert "https://n.org/x" in c["new_sources"]


def test_compare_identical_runs_is_all_unchanged():
    c = compare_runs(make_snap(), make_snap())
    assert c["unchanged"] == 4 and not c["new_claims"] and not c["status_changed"]
    assert all(r["change"] == 0 for r in c["table"])


def test_reworded_claim_is_matched_not_reported_as_new():
    a, b = make_snap(), make_snap()
    b["fact_check_results"][0]["claim"] = CLAIM_A.replace("reach", "can reach")
    assert not compare_runs(a, b)["new_claims"]

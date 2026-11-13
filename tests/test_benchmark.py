import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from snap_fixture import make_snap

from benchmark import reference_recall, summarise_run, to_markdown


def test_reference_recall_counts_only_claims_that_state_the_fact():
    snap = make_snap()
    out = reference_recall(["Recovery rates for nickel and cobalt reach 95% in recycling.",
                            "CATL sells sodium-ion batteries in 2030."], snap["fact_check_results"])
    assert out["reference_claims"] == 2 and out["recalled"] == 1 and out["recall"] == 0.5
    assert out["recalled_and_confirmed"] == 1


def test_summary_row_and_markdown_table():
    row = summarise_run("Q?", make_snap(), ["Pyrometallurgy recovers lithium poorly."])
    assert row["claims"] == 4 and row["confirmed"] == 1 and row["status"] == "partial" and row["recall"] == 1.0
    md = to_markdown([row, {"question": "bad", "error": "RuntimeError: x"}])
    assert "| Q? | partial | 4 | 1 |" in md and "crashed" in md and "Recalled" in md

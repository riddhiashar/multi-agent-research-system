import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).parent))
import csv
import io
import json

import pytest

from snap_fixture import make_snap
from ui.report import build_report, claims_export, evidence_csv, pdf_text, to_markdown


def test_report_has_every_required_section_in_order():
    doc = build_report(make_snap())
    assert [s["title"] for s in doc["sections"]] == [
        "Executive summary", "Research methodology", "Findings by sub-topic", "Comparison of sub-topics",
        "Evidence and citations", "Limitations and research gaps", "Conclusion", "Generated narrative report"]
    info = dict(doc["info"])
    assert info["Claims checked"] == "4" and info["Sub-topics researched"] == "2" and info["Run status"] == "Partial"


def test_report_numbers_are_the_real_ones():
    md = to_markdown(build_report(make_snap()))
    assert "1 confirmed" in md and "3 unverified" in md
    assert "2 were searched for a second source" in md or "2 were searched" in md
    assert "[1]" in md and "alpha.org" in md
    assert "Summary unavailable" in md           # the failed sub-topic is stated, not hidden
    assert "do not prove" not in md and "does not prove none exist" in md


def test_report_without_confirmed_claims_does_not_overclaim():
    snap = make_snap()
    for r in snap["fact_check_results"]:
        r["status"] = "UNVERIFIED"
    assert "unconfirmed lead" in to_markdown(build_report(snap))


def test_pdf_is_valid_and_contains_text():
    pytest.importorskip("reportlab")
    from ui.report import to_pdf
    snap = make_snap()
    snap["fact_check_results"][0]["claim"] += " “quoted” → <b>& ₂"   # tricky characters
    data = to_pdf(build_report(snap))
    assert data[:5] == b"%PDF-" and len(data) > 2000
    pypdf = pytest.importorskip("pypdf")
    text = " ".join(p.extract_text() for p in pypdf.PdfReader(io.BytesIO(data)).pages)
    assert "Executive summary" in text and "Research methodology" in text


def test_pdf_text_is_latin1_safe_and_escaped():
    out = pdf_text("a → b <i> & 中")
    out.encode("latin-1")
    assert "&lt;i&gt;" in out and "&amp;" in out


def test_json_export_uses_the_real_schema():
    data = claims_export(make_snap())
    c = data["claims"][0]
    assert c["status"] == "CONFIRMED" and c["independent_domains"] == 2 and c["verified_by_search"] is True
    assert c["sources"][1]["origin"] == "verification"
    json.dumps(data)


def test_csv_has_one_row_per_source_and_blocks_formula_injection():
    snap = make_snap()
    snap["fact_check_results"][1]["sources"][0]["title"] = "=HYPERLINK(\"http://evil\")"
    rows = list(csv.DictReader(io.StringIO(evidence_csv(snap))))
    assert len(rows) == 6
    assert any(r["source_title"].startswith("'=") for r in rows)
    assert all(not r["source_title"].startswith("=") for r in rows)


def test_emoji_variation_selector_does_not_leave_a_stray_question_mark():
    assert pdf_text("⚠️ Disputed") == "[!] Disputed"

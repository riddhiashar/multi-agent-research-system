from ui.charts import topic_rows, verdict_rows


def test_verdict_rows_keep_the_zero():
    rows = verdict_rows({"CONFIRMED": 7, "UNVERIFIED": 32})
    assert [(r["Verdict"], r["Claims"]) for r in rows] == [("Confirmed", 7), ("Disputed", 0), ("Unverified", 32)]


def test_topic_rows_are_long_format_with_every_verdict():
    rows = topic_rows([{"topic": "Q1: a", "CONFIRMED": 1, "DISPUTED": 0, "UNVERIFIED": 2}])
    assert len(rows) == 3 and sum(r["Claims"] for r in rows) == 3 and {r["Sub-topic"] for r in rows} == {"Q1: a"}

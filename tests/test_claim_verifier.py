import pytest

from tools.claim_verifier import (assess, assess_text, compare_quantities, direction, extract_quantities, key_entities,
                                  query_entities)

ENTS = {"tesla", "catl", "redwood"}


def rel(claim, sentence, **kw):
    return assess(claim, sentence, entities=ENTS, **kw)["relation"]


def test_quantities_units_scale_ranges_and_years():
    q = extract_quantities("Costs fell 30% to $2.5 billion between 2020-2025, a 95-99% recovery and 2,000 MWh.")
    fams = {(x["family"], round(x["lo"]), round(x["hi"])) for x in q["quantities"]}
    assert ("pct", 30, 30) in fams and ("usd", 2_500_000_000, 2_500_000_000) in fams
    assert ("pct", 95, 99) in fams and ("wh", 2_000_000_000, 2_000_000_000) in fams
    assert {2020, 2025} <= q["years"]


def test_same_number_different_unit_is_not_a_match():
    c, s = extract_quantities("95 GWh of cells")["quantities"], extract_quantities("95% of cells")["quantities"]
    assert compare_quantities(c, s)["status"] == "absent"


def test_unit_conversion_and_rounding_match():
    a = extract_quantities("2 GWh")["quantities"]
    assert compare_quantities(a, extract_quantities("2000 MWh")["quantities"])["status"] == "match"
    assert compare_quantities(extract_quantities("9.5 million")["quantities"],
                              extract_quantities("9,500,000")["quantities"])["status"] == "match"


def test_conflict_needs_the_same_unit():
    c = extract_quantities("recovery of 95%")["quantities"]
    assert compare_quantities(c, extract_quantities("recovery of 70%")["quantities"])["status"] == "conflict"
    assert compare_quantities(c, extract_quantities("a 70 tonne load")["quantities"])["status"] == "absent"


@pytest.mark.parametrize("claim,sentence,expected", [
    ("Tesla recycles 92% of materials.", "CATL recycles 92% of materials.", "unrelated"),       # wrong subject
    ("Tesla recycles 92% of materials.", "Tesla recycles 92% of materials in Nevada.", "supports"),
    ("Tesla recycles 92% of materials.", "Tesla recycles 70% of materials.", "contradicts"),    # figure
    ("Tesla recycles 92% of materials.", "Tesla does not recycle 92% of materials.", "contradicts"),  # negation
    ("Costs fall 30% at scale.", "Costs rise 30% at scale.", "contradicts"),                    # direction
    ("Market reaches $2 billion by 2030.", "Market reaches $2 billion by 2035.", "uncertain"),  # period
    ("Plants recover up to 95% of lithium.", "Plants recover at least 95% of lithium.", "uncertain"),  # bounds
    ("Nickel recovery reaches 95-99%.", "Nickel recovery reaches 97%.", "supports"),            # inside range
])
def test_relations(claim, sentence, expected):
    assert rel(claim, sentence) == expected


def test_reasons_explain_the_decision():
    a = assess("Tesla recycles 92% of materials.", "Tesla recycles 70% of materials.", entities=ENTS)
    assert "figure differs" in a["reasons"][0] and "92%" in a["reasons"][0] and "70%" in a["reasons"][0]


def test_known_limits_fail_safe_never_false_support():
    # paraphrase and pronoun subject are missed (conservative), but must never be reported as 'supports'
    assert rel("CATL began mass production of sodium-ion batteries.", "CATL started producing Na-ion cells at scale.") != "supports"
    assert rel("Tesla recycles 92% of materials.", "The company recycles 92% of materials.") != "supports"


def test_entities_from_claim_and_question():
    assert {"catl", "iea"} <= key_entities("The IEA says CATL sells cells.")
    assert "tesla" in key_entities("Tesla recycles cells.", extra={"tesla"})
    assert {"tesla", "redwood"} <= query_entities("Compare Tesla and Redwood Materials recycling")
    assert "gwh" not in key_entities("The plant makes 5 GWh a year")


def test_direction():
    assert direction("costs increase") == "up" and direction("costs decrease") == "down" and direction("costs change") is None


def test_assess_text_picks_support_and_conflict_separately():
    text = "Tesla recycles 70% of materials. Tesla recycles 92% of materials in Nevada."
    a = assess_text("Tesla recycles 92% of materials.", text, entities=ENTS)
    assert "92%" in a["supports"]["sentence"] and "70%" in a["contradicts"]["sentence"]

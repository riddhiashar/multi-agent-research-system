from tools.claim_merge import claims_equivalent, merge_duplicate_claims, expand_verdicts


def src(url, i=1):
    return {"id": i, "title": url, "url": url}


A = "Redwood Materials recovers 95% of cobalt and nickel from used lithium-ion batteries"
A2 = "Redwood Materials recovers 95% of the cobalt and nickel in used lithium-ion batteries"


def test_rephrased_claims_are_equivalent():
    assert claims_equivalent(A, A2)


def test_different_numbers_never_merge():
    assert not claims_equivalent(A, A.replace("95%", "70%"))


def test_negation_never_merges():
    assert not claims_equivalent("Solid-state cells do not catch fire during nail penetration tests easily",
                                 "Solid-state cells catch fire during nail penetration tests easily")


def test_short_claims_never_merge():
    assert not claims_equivalent("Claim A", "Claim A")


def test_merge_pools_sources_and_tracks_origins():
    facts = [
        {"claim": A, "sub_question": "Q1", "sources": [src("https://redwoodmaterials.com/x")]},
        {"claim": A2, "sub_question": "Q2", "sources": [src("https://reuters.com/y")]},
        {"claim": "Totally different statement about CATL sodium cells and their price range here",
         "sub_question": "Q1", "sources": [src("https://a.com/1")]},
    ]
    merged = merge_duplicate_claims(facts)
    assert len(merged) == 2
    m = merged[0]
    assert m["merged_count"] == 2
    assert {s["url"] for s in m["sources"]} == {"https://redwoodmaterials.com/x", "https://reuters.com/y"}
    assert [s["id"] for s in m["sources"]] == [1, 2]
    assert {o["sub_question"] for o in m["origins"]} == {"Q1", "Q2"}
    assert "origins" not in merged[1]            # singleton untouched


def test_same_source_url_not_double_counted():
    facts = [{"claim": A, "sub_question": "Q1", "sources": [src("https://a.com/p")]},
             {"claim": A2, "sub_question": "Q2", "sources": [src("https://a.com/p")]}]
    assert len(merge_duplicate_claims(facts)[0]["sources"]) == 1


def test_duplicates_inside_one_subquestion_collapse():
    facts = [{"claim": A, "sub_question": "Q1", "sources": [src("https://a.com/p")]},
             {"claim": A2, "sub_question": "Q1", "sources": [src("https://b.com/p")]}]
    merged = merge_duplicate_claims(facts)
    assert len(merged[0]["origins"]) == 1
    out = expand_verdicts([{**merged[0], "status": "CONFIRMED"}])
    assert len(out) == 1


def test_expand_gives_every_subquestion_the_verdict():
    facts = [{"claim": A, "sub_question": "Q1", "sources": [src("https://a.com/p")]},
             {"claim": A2, "sub_question": "Q2", "sources": [src("https://b.com/p")]}]
    merged = merge_duplicate_claims(facts)
    out = expand_verdicts([{**merged[0], "status": "CONFIRMED"}])
    assert sorted(r["sub_question"] for r in out) == ["Q1", "Q2"]
    assert all(r["status"] == "CONFIRMED" and "origins" not in r for r in out)

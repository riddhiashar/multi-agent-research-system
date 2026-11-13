from tools.verification import supports, verify_claim, verify_claims

CLAIM = "Hydrometallurgical processing achieves a 95% recovery rate for lithium."


def cand(url, text, title="t"):
    return {"title": title, "url": url, "content": text}


def test_supports_requires_same_figures():
    assert supports(CLAIM, "Hydrometallurgical processes reach a 95% recovery rate for lithium in tests.")
    assert not supports(CLAIM, "Hydrometallurgical processes reach a 70% recovery rate for lithium.")


def test_supports_is_sentence_level_for_negation():
    text = "This is not a small market. Hydrometallurgical processing achieves a 95% recovery rate for lithium."
    assert supports(CLAIM, text)           # the stray 'not' is in a different sentence
    assert not supports(CLAIM, "Hydrometallurgical processing does not achieve a 95% recovery rate for lithium.")


def test_claims_without_figures_need_close_wording():
    claim = "Redwood Materials operates the largest lithium-ion battery recycling facility outside Asia"
    assert not supports(claim, "Redwood Materials launched a second-life battery storage partnership.")
    assert supports(claim, "Redwood Materials operates the largest lithium-ion battery recycling facility outside Asia today.")


def test_verify_adds_only_new_supporting_domains():
    fact = {"claim": CLAIM, "sources": [{"id": 1, "title": "a", "url": "https://a.org/x"}]}
    calls = []

    def search(q, exclude_domains=()):
        calls.append(exclude_domains)
        return [cand("https://a.org/y", "Hydrometallurgical processing achieves a 95% recovery rate for lithium."),
                cand("https://b.com/z", "Hydrometallurgical processing achieves a 95% recovery rate for lithium."),
                cand("https://c.net/q", "Unrelated text about cathodes.")]

    new, added, failed = verify_claim(fact, search)
    assert calls == [["a.org"]] and added == 1 and not failed
    assert [s["url"] for s in new["sources"]] == ["https://a.org/x", "https://b.com/z"]
    assert new["sources"][1]["via"] == "verification" and [s["id"] for s in new["sources"]] == [1, 2]


def test_search_failure_leaves_fact_unchanged():
    fact = {"claim": CLAIM, "sources": [{"id": 1, "title": "a", "url": "https://a.org/x"}]}

    def boom(q, exclude_domains=()):
        raise RuntimeError("down")

    new, added, failed = verify_claim(fact, boom)
    assert new["verification"] == "search_failed" and new["sources"] == fact["sources"] and added == 0 and failed


def test_verify_claims_prioritises_figures_and_respects_cap():
    def mk(claim, n):
        return {"claim": claim, "sources": [{"id": 1, "title": "a", "url": f"https://s{n}.org/x"}]}
    facts = [mk("Plain statement about recycling policy and regulation across regions here", 1),
             mk("Plant capacity reaches 50000 tonnes per year in Nevada operations", 2),
             mk("Another plain statement about supply chain integration strategy overall", 3)]
    seen = []

    def search(q, exclude_domains=()):
        seen.append(q)
        return []

    _, stats = verify_claims(facts, search, max_claims=2)
    assert stats["tried"] == 2 and seen[0].startswith("Plant capacity")


def test_already_corroborated_claims_not_searched():
    fact = {"claim": CLAIM, "sources": [{"id": 1, "title": "a", "url": "https://a.org/x"},
                                         {"id": 2, "title": "b", "url": "https://b.com/x"}]}
    _, stats = verify_claims([fact], lambda q, exclude_domains=(): 1 / 0)
    assert stats["tried"] == 0


def test_upgraded_counted_when_claim_reaches_two_domains():
    fact = {"claim": CLAIM, "sources": [{"id": 1, "title": "a", "url": "https://a.org/x"}]}
    search = lambda q, exclude_domains=(): [cand("https://b.com/z", CLAIM)]
    out, stats = verify_claims([fact], search)
    assert {k: stats[k] for k in ("tried", "upgraded", "sources_added", "failed")} == {"tried": 1, "upgraded": 1, "sources_added": 1, "failed": 0}


def test_verification_records_the_evidence_sentence():
    fact = {"claim": CLAIM, "sources": [{"id": 1, "title": "a", "url": "https://a.org/x"}]}
    text = "Intro text. Hydrometallurgical processing achieves a 95% recovery rate for lithium. Outro."
    new, _, _ = verify_claim(fact, lambda q, exclude_domains=(): [cand("https://b.com/z", text)])
    assert new["sources"][1]["evidence"] == "Hydrometallurgical processing achieves a 95% recovery rate for lithium."


def test_verification_budget_is_shared_across_subquestions():
    def mk(sq, n):
        return {"claim": f"Plant {sq} capacity reaches {n}0000 tonnes per year at site number {n}",
                "sub_question": sq,
                "sources": [{"id": 1, "title": "a", "url": f"https://{sq.lower()}{n}.org/x"}]}
    facts = [mk("Q1", n) for n in range(1, 5)] + [mk("Q2", n) for n in range(1, 5)]
    tried = []

    def search(q, exclude_domains=()):
        tried.append(q)
        return []

    verify_claims(facts, search, max_claims=4)
    assert sum("Q1" in q for q in tried) == 2 and sum("Q2" in q for q in tried) == 2


# ---- Day 27: meaning, numbers and context -------------------------------------------------------
def test_new_domain_with_the_wrong_subject_or_unit_is_rejected():
    fact = {"claim": "Tesla recycles 92% of battery materials.", "sources": [{"id": 1, "title": "a", "url": "https://a.org/x"}]}
    search = lambda q, exclude_domains=(): [cand("https://b.com/1", "CATL recycles 92% of battery materials."),
                                            cand("https://c.net/2", "Tesla ships 92 GWh of battery materials.")]
    new, added, _ = verify_claim(fact, search, entities={"tesla", "catl"})
    assert added == 0 and new["verification"] == "no_match"


def test_contradicting_sentence_is_recorded_as_conflict_not_dropped():
    fact = {"claim": "Tesla recycles 92% of battery materials.", "sources": [{"id": 1, "title": "a", "url": "https://a.org/x"}]}
    search = lambda q, exclude_domains=(): [cand("https://b.com/1", "Tesla recycles 70% of battery materials.")]
    new, added, _ = verify_claim(fact, search, entities={"tesla"})
    assert added == 0 and new["conflicting_evidence"][0]["domain"] == "b.com"
    assert "figure differs" in new["conflicting_evidence"][0]["reason"]


def test_verification_source_records_why_it_was_accepted():
    fact = {"claim": CLAIM, "sources": [{"id": 1, "title": "a", "url": "https://a.org/x"}]}
    new, _, _ = verify_claim(fact, lambda q, exclude_domains=(): [cand("https://b.com/z", CLAIM)])
    s = new["sources"][1]
    assert s["relation"] == "supports" and s["method"] == "rules" and "figures match" in s["reasons"][0]


def test_judge_can_rescue_a_paraphrase_only_when_guards_pass():
    claim = "CATL started mass production of sodium-ion batteries"
    fact = {"claim": claim, "sources": [{"id": 1, "title": "a", "url": "https://a.org/x"}]}
    para = "CATL started mass production of sodium cells for passenger cars."          # related, not conclusive
    judge = lambda c, sents: [{"relation": "supports", "why": "same event"} for _ in sents]
    search = lambda q, exclude_domains=(): [cand("https://b.com/z", para)]
    no_judge, added0, _ = verify_claim(fact, search, entities={"catl"})
    with_judge, added1, _ = verify_claim(fact, search, entities={"catl"}, judge_fn=judge)
    assert added0 == 0 and added1 == 1 and with_judge["sources"][1]["method"] == "llm"


def test_judge_cannot_override_a_figure_mismatch():
    fact = {"claim": CLAIM, "sources": [{"id": 1, "title": "a", "url": "https://a.org/x"}]}
    wrong = "Hydrometallurgical processes reach a 70% recovery rate for lithium."
    yes = lambda c, sents: [{"relation": "supports", "why": "x"} for _ in sents]
    new, added, _ = verify_claim(fact, lambda q, exclude_domains=(): [cand("https://b.com/z", wrong)], judge_fn=yes)
    assert added == 0

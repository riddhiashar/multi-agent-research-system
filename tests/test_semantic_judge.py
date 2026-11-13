import json

from tools.semantic_judge import build_prompt, judge, parse


def test_prompt_contains_claim_and_numbered_sentences():
    p = build_prompt("Claim X", ["s one", "s two"])
    assert "Claim X" in p and "0. s one" in p and "1. s two" in p and "strict" in p.lower()


def test_parse_handles_fences_and_bad_labels():
    raw = "```json\n" + json.dumps([{"i": 0, "relation": "supports", "why": "same"}, {"i": 1, "relation": "banana"}]) + "\n```"
    out = parse(raw, 2)
    assert out[0]["relation"] == "supports" and out[1]["relation"] == "unrelated"


def test_garbage_or_failure_returns_nothing():
    assert parse("not json", 2) == []
    assert judge("c", ["s"], lambda p: 1 / 0) == []
    assert judge("c", [], lambda p: "[]") == []


def test_judge_caps_sentences_per_call():
    seen = {}
    judge("c", [f"s{i}" for i in range(9)], lambda p: seen.setdefault("p", p) and "[]")
    assert "3. s3" in seen["p"] and "4. s4" not in seen["p"]

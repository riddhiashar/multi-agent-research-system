from tools.verifier_eval import evaluate


def test_bundled_cases_all_pass_and_nothing_is_falsely_supported():
    r = evaluate()
    assert r["cases"] >= 30
    assert r["false_support"] == 0, r["wrong"]
    assert r["accuracy"] == 1.0, r["wrong"]

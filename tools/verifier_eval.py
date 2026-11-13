"""Measure the verifier against hand-labelled (claim, sentence) pairs (Day 27).
Run:  python -m tools.verifier_eval            (prints accuracy and a confusion table)
The bundled cases are a regression suite written by the developer, NOT an independent benchmark: add pairs taken from
your own real runs (cases where the explorer got it wrong) to keep it honest."""
import json
from collections import Counter
from pathlib import Path
from typing import Dict, List

from tools.claim_verifier import assess

DEFAULT_PATH = Path(__file__).resolve().parent.parent / "evaluation" / "verifier_cases.json"
LABELS = ["supports", "contradicts", "uncertain", "unrelated"]


def evaluate(path=DEFAULT_PATH) -> Dict:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    ents = data.get("entities_from_question", [])
    confusion, wrong = Counter(), []
    for c in data["cases"]:
        got = assess(c["claim"], c["sentence"], entities=ents)["relation"]
        confusion[(c["expected"], got)] += 1
        if got != c["expected"]:
            wrong.append({**c, "got": got})
    n = len(data["cases"])
    # the error that matters most: calling a sentence supportive when it is not
    false_support = sum(v for (exp, got), v in confusion.items() if got == "supports" and exp != "supports")
    return {"cases": n, "correct": n - len(wrong), "accuracy": round((n - len(wrong)) / n, 3) if n else 0.0,
            "false_support": false_support, "confusion": confusion, "wrong": wrong}


def main() -> None:
    r = evaluate()
    print(f"{r['correct']}/{r['cases']} correct ({r['accuracy']:.0%}); false 'supports': {r['false_support']}")
    print(f"{'expected':<13}" + "".join(f"{g:>13}" for g in LABELS))
    for exp in LABELS:
        print(f"{exp:<13}" + "".join(f"{r['confusion'].get((exp, g), 0):>13}" for g in LABELS))
    for w in r["wrong"]:
        print(f"WRONG expected {w['expected']}, got {w['got']}: {w['claim']} | {w['sentence']}")


if __name__ == "__main__":
    main()

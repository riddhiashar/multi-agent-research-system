"""Optional LLM judge for sentences the rules could not decide (Day 27).

The rules in claim_verifier.py are strict on purpose: a paraphrase ("began mass production" vs "started producing
at scale") scores as 'uncertain'. This asks the LLM ONE small question per claim: does each of these sentences
support, contradict or not address the claim? Output is a few tokens per sentence, so it fits the free tier.

Safety: the model can only decide pairs whose deterministic guards already passed (figures match or the claim has
none, right subject, same polarity, not an opposite direction) - see `judgeable` in claim_verifier.assess.
A failed or unparseable reply changes nothing: it never upgrades a claim."""
import json
import os
import re
from typing import Callable, Dict, List

MAX_SENTENCES_PER_CALL = 4
MAX_JUDGE_CALLS = 6


def enabled() -> bool:
    return os.getenv("SEMANTIC_JUDGE", "false").lower() == "true"


def build_prompt(claim: str, sentences: List[str]) -> str:
    listed = "\n".join(f"{i}. {s[:300]}" for i, s in enumerate(sentences))
    return f"""You compare ONE claim with source sentences. For each sentence decide:
- "supports": it states the same fact (same subject, same figures and units, same time period, same direction)
- "contradicts": it states something incompatible (different figure, opposite meaning, negation)
- "unrelated": it does not state or contradict the claim, or only partly covers it
Be strict: if a figure, unit, subject or time period differs, it does not support.

Claim: "{claim}"

Sentences:
{listed}

Respond with ONLY a JSON array: [{{"i": 0, "relation": "supports", "why": "max 10 words"}}, ...]"""


def parse(raw: str, n: int) -> List[Dict]:
    cleaned = re.sub(r"^```(json)?", "", (raw or "").strip()).strip()
    cleaned = re.sub(r"```$", "", cleaned).strip()
    try:
        data = json.loads(cleaned)
    except (ValueError, TypeError):
        return []
    out = {}
    if isinstance(data, list):
        for item in data:
            if isinstance(item, dict) and isinstance(item.get("i"), int) and 0 <= item["i"] < n:
                rel = str(item.get("relation", "")).lower()
                if rel in ("supports", "contradicts", "unrelated"):
                    out[item["i"]] = {"relation": rel, "why": str(item.get("why", ""))[:80]}
    return [out.get(i, {"relation": "unrelated", "why": "not returned"}) for i in range(n)] if out else []


def judge(claim: str, sentences: List[str], llm_fn: Callable[[str], str]) -> List[Dict]:
    """One verdict per sentence, or [] if the call failed (callers then keep the rule-based result)."""
    sentences = sentences[:MAX_SENTENCES_PER_CALL]
    if not sentences:
        return []
    try:
        return parse(llm_fn(build_prompt(claim, sentences)), len(sentences))
    except Exception:
        return []

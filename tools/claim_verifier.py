"""Does this sentence actually say what the claim says? (Day 27)

Word overlap alone is fooled by "Tesla recycles 92%" vs "CATL recycles 92%" (wrong company), "95 GWh" vs "95%"
(wrong unit), "by 2030" vs "by 2035" (wrong period) or "increases" vs "decreases". This module compares claim and
sentence on five things, in code, with no LLM and no network:

  1. quantities with units, scale and ranges (95-99% vs 97%, 9.5 million vs 9,500,000, $2bn vs 2 billion dollars)
  2. the subject (named entities and acronyms must be present)
  3. polarity (negation) and direction (increase vs decrease)
  4. time scope (years) and bounds ("up to" vs "at least")
  5. enough shared content words

assess() returns one of: supports | contradicts | uncertain | unrelated, plus the reasons, so the UI can show WHY.
An optional LLM judge (tools/semantic_judge.py) may only upgrade `uncertain` pairs whose deterministic guards
passed: the model proposes, the code disposes."""
import math
import re
from typing import Dict, Iterable, List, Optional, Set

from tools.claim_merge import _profile

REL_TOL = 0.01
MIN_OVERLAP_NUMBERS = 0.5
MIN_OVERLAP_WORDS = 0.75
MIN_RELATED = 0.35          # below this the sentence is simply about something else
MIN_JUDGEABLE = 0.25        # below this not even an LLM judge is asked
ENTITY_MATCH = 0.6

_SCALE = {"thousand": 1e3, "million": 1e6, "mn": 1e6, "billion": 1e9, "bn": 1e9, "trillion": 1e12}
_UNITS = {  # surface form -> (family, factor)
    "%": ("pct", 1), "percent": ("pct", 1), "pct": ("pct", 1), "per cent": ("pct", 1), "percentage points": ("pct", 1),
    "wh": ("wh", 1), "kwh": ("wh", 1e3), "mwh": ("wh", 1e6), "gwh": ("wh", 1e9), "twh": ("wh", 1e12),
    "kw": ("w", 1e3), "mw": ("w", 1e6), "gw": ("w", 1e9),
    "kg": ("kg", 1), "tonne": ("kg", 1e3), "tonnes": ("kg", 1e3), "ton": ("kg", 1e3), "tons": ("kg", 1e3),
    "km": ("m", 1e3), "mile": ("m", 1609), "miles": ("m", 1609),
    "year": ("year", 1), "years": ("year", 1), "month": ("year", 1 / 12), "months": ("year", 1 / 12),
    "day": ("day", 1), "days": ("day", 1), "hour": ("hour", 1), "hours": ("hour", 1),
    "cycle": ("cycle", 1), "cycles": ("cycle", 1),
    "usd": ("usd", 1), "dollar": ("usd", 1), "dollars": ("usd", 1), "euro": ("eur", 1), "euros": ("eur", 1), "eur": ("eur", 1),
}
_CUR = {"$": "usd", "€": "eur", "£": "gbp"}
_NUM = r"\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?"
_UNIT_RE = "|".join(sorted((re.escape(u) for u in _UNITS if u != "%"), key=len, reverse=True))
_QUANTITY = re.compile(
    rf"(?<![\w.])(?P<cur>[$€£])?\s?(?P<a>{_NUM})(?:\s*(?:-|–|—|to)\s*(?P<b>{_NUM}))?"
    rf"\s*(?P<scale>thousand|million|billion|trillion|mn|bn)?\s*(?P<unit>%|(?:{_UNIT_RE})(?![A-Za-z]))?", re.I)

_UP = ("increas", "rise", "rose", "grow", "grew", "higher", "gain", "improv", "boost", "expand", "surge", "climb", "exceed",
       "outperform", "more ")
_DOWN = ("decreas", "reduc", "fall", "fell", "drop", "lower", "declin", "cut", "shrink", "lose", "lost", "fewer",
         "underperform", "less ")
_UPPER = re.compile(r"\b(up to|at most|no more than|maximum|max|under|below|less than|fewer than)\b", re.I)
_LOWER = re.compile(r"\b(at least|more than|over|above|minimum|exceed\w*|greater than|no less than)\b", re.I)
_GENERIC_CAPS = {"EV", "EVS", "AI", "LLM", "LLMS", "US", "USA", "UK", "EU", "CEO", "R&D", "GDP", "CO2", "NLP"}
_NOT_ENTITIES = {"january", "february", "march", "april", "may", "june", "july", "august", "september", "october",
                 "november", "december", "monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"}


# ----------------------------------------------------------------------------- quantities
def extract_quantities(text: str) -> Dict:
    """-> {"quantities": [{"lo","hi","family","raw"}], "years": {2030, ...}}. 4-digit numbers 1900-2100 with no
    unit are years, not quantities."""
    quantities, years = [], set()
    for m in _QUANTITY.finditer((text or "").replace("−", "-")):
        a = float(m.group("a").replace(",", ""))
        b = float(m.group("b").replace(",", "")) if m.group("b") else a
        unit = (m.group("unit") or "").lower()
        scale = _SCALE.get((m.group("scale") or "").lower(), 1)
        cur = _CUR.get(m.group("cur") or "")
        if not unit and not m.group("scale") and not cur:
            if 1900 <= a <= 2100 and float(a).is_integer() and 1900 <= b <= 2100:
                years.update({int(a), int(b)})
                continue
        family, factor = _UNITS.get(unit, ("", 1)) if unit else ("", 1)
        if cur and not unit:
            family = cur
        lo, hi = sorted((a * scale * factor, b * scale * factor))
        quantities.append({"lo": lo, "hi": hi, "family": family, "raw": m.group(0).strip()})
    return {"quantities": quantities, "years": years}


def _close(x: float, y: float) -> bool:
    return math.isclose(x, y, rel_tol=REL_TOL, abs_tol=1e-9)


def _contained(inner: Dict, outer: Dict) -> bool:
    """claim interval inside the sentence interval (with rounding tolerance)."""
    return (inner["lo"] >= outer["lo"] or _close(inner["lo"], outer["lo"])) and \
           (inner["hi"] <= outer["hi"] or _close(inner["hi"], outer["hi"]))


def _disjoint(a: Dict, b: Dict) -> bool:
    return a["hi"] < b["lo"] and not _close(a["hi"], b["lo"]) or b["hi"] < a["lo"] and not _close(b["hi"], a["lo"])


def compare_quantities(claim_q: List[Dict], sent_q: List[Dict]) -> Dict:
    """status: n/a (claim has no figures) | match | partial (sentence gives a narrower/overlapping figure)
    | conflict (same unit, different figure) | absent (the figure is not stated)."""
    if not claim_q:
        return {"status": "n/a", "detail": ""}
    worst, details = "match", []
    order = {"match": 0, "partial": 1, "absent": 2, "conflict": 3}
    for c in claim_q:
        same = [s for s in sent_q if c["family"] == "" or s["family"] == c["family"]]
        if any(_contained(c, s) for s in same):
            status, detail = "match", f"{c['raw']} stated"
        elif c["lo"] < c["hi"] and any(_contained(s, c) for s in same):
            status, detail = "match", f"source figure lies inside the claimed range {c['raw']}"
        elif any(not _disjoint(c, s) for s in same):
            status, detail = "partial", f"{c['raw']} only partly covered"
        elif c["family"] and same:
            status, detail = "conflict", f"claim says {c['raw']}, source says {', '.join(s['raw'] for s in same[:2])}"
        else:
            status, detail = "absent", f"{c['raw']} not stated"
        details.append(detail)
        if order[status] > order[worst]:
            worst = status
    return {"status": worst, "detail": "; ".join(details)}


# ----------------------------------------------------------------------------- subject / entities
def key_entities(text: str, extra: Iterable[str] = ()) -> Set[str]:
    """Names the claim is about: acronyms, capitalised words (not sentence-initial), a capitalised first word
    followed by another capital, and any word in `extra` (e.g. names taken from the research question)."""
    extra = {e.lower() for e in extra}
    words = re.findall(r"[A-Za-z][A-Za-z0-9&\-]*(?:'s)?", text or "")
    out = set()
    for i, raw in enumerate(words):
        w = re.sub(r"'s$", "", raw)
        low = w.lower()
        if low in _NOT_ENTITIES or low in _UNITS or w.upper() in _GENERIC_CAPS:
            continue
        if low in extra:
            out.add(low)
        elif w.isupper() and len(w) >= 2:
            out.add(low)
        elif w[0].isupper() and i > 0:
            out.add(low)
        elif i == 0 and (any(ch.isupper() for ch in w[1:]) or (len(words) > 1 and words[1][:1].isupper())):
            out.add(low)
    return out


def query_entities(query: str) -> Set[str]:
    """Names in the research question: authoritative subjects for every claim in the run."""
    return {e for e in key_entities(query or "") if len(e) > 2}


def entity_check(entities: Set[str], text: str) -> Dict:
    if not entities:
        return {"ok": True, "missing": []}
    low = (text or "").lower()
    missing = sorted(e for e in entities if not re.search(rf"\b{re.escape(e)}", low))
    return {"ok": (len(entities) - len(missing)) / len(entities) >= ENTITY_MATCH, "missing": missing}


# ----------------------------------------------------------------------------- polarity, direction, bounds
def direction(text: str) -> Optional[str]:
    low = (text or "").lower() + " "
    up, down = any(k in low for k in _UP), any(k in low for k in _DOWN)
    return "up" if up and not down else "down" if down and not up else None


def bound(text: str) -> str:
    if _UPPER.search(text or ""):
        return "upper"
    if _LOWER.search(text or ""):
        return "lower"
    return "exact"


# ----------------------------------------------------------------------------- the assessment
def _result(relation, score, reasons, numbers="n/a", entities_ok=True, judgeable=False) -> Dict:
    return {"relation": relation, "score": round(score, 2), "reasons": reasons, "numbers": numbers,
            "entities_ok": entities_ok, "judgeable": judgeable}


def assess(claim: str, sentence: str, title: str = "", entities: Iterable[str] = ()) -> Dict:
    c, s = _profile(claim), _profile(sentence)
    if not c["tokens"]:
        return _result("unrelated", 0.0, ["empty claim"])
    overlap = len(c["tokens"] & s["tokens"]) / len(c["tokens"])
    qc, qs = extract_quantities(claim), extract_quantities(sentence)
    ents = entity_check(key_entities(claim, entities), f"{sentence} {title}")
    nums = compare_quantities(qc["quantities"], qs["quantities"])
    min_overlap = MIN_OVERLAP_NUMBERS if qc["quantities"] else MIN_OVERLAP_WORDS
    reasons: List[str] = []

    if overlap < MIN_RELATED and nums["status"] != "conflict":
        return _result("unrelated", overlap, ["shares too little wording with the claim"], nums["status"], ents["ok"])
    if not ents["ok"]:
        return _result("unrelated", overlap, [f"about a different subject (missing: {', '.join(ents['missing'])})"],
                       nums["status"], False)
    if c["negated"] != s["negated"]:
        return _result("contradicts" if overlap >= MIN_OVERLAP_NUMBERS else "unrelated", overlap,
                       ["negation differs: the sentence says the opposite"], nums["status"], True)
    dc, ds = direction(claim), direction(sentence)
    if dc and ds and dc != ds and overlap >= MIN_OVERLAP_NUMBERS:
        return _result("contradicts", overlap, [f"opposite direction (claim: {dc}, source: {ds})"], nums["status"], True)
    if nums["status"] == "conflict" and overlap >= MIN_OVERLAP_NUMBERS:
        return _result("contradicts", overlap, [f"figure differs: {nums['detail']}"], "conflict", True)

    guards_ok = nums["status"] in ("match", "n/a") and overlap >= MIN_JUDGEABLE
    if nums["status"] in ("absent", "partial", "conflict"):
        return _result("uncertain" if overlap >= min_overlap else "unrelated", overlap,
                       [f"figure not confirmed: {nums['detail']}"], nums["status"], True, False)
    if qc["years"] and qs["years"] and not qc["years"] <= qs["years"]:
        return _result("uncertain", overlap, [f"different time period (claim {sorted(qc['years'])}, source {sorted(qs['years'])})"],
                       nums["status"], True, guards_ok)
    bc, bs = bound(claim), bound(sentence)
    if {bc, bs} == {"upper", "lower"}:
        return _result("uncertain", overlap, ["claim and source use opposite bounds (up to vs at least)"], nums["status"], True, guards_ok)
    if overlap >= min_overlap:
        if nums["status"] == "match":
            reasons.append(f"figures match ({nums['detail']})")
        if ents["ok"] and key_entities(claim, entities):
            reasons.append("subject present")
        return _result("supports", overlap, reasons or ["wording matches"], nums["status"], True, False)
    return _result("uncertain", overlap, ["related, but wording differs too much to be sure"], nums["status"], True, guards_ok)


# ----------------------------------------------------------------------------- over a whole text
def split_sentences(text: str) -> List[str]:
    return [x.strip() for x in re.split(r"(?<=[.!?])\s+|\n+", text or "") if x.strip()]


def assess_text(claim: str, text: str, title: str = "", entities: Iterable[str] = ()) -> Dict:
    """Best supporting sentence, best contradicting sentence and judgeable-uncertain sentences in `text`."""
    best = {"supports": None, "contradicts": None, "uncertain": [], "closest": None}
    for sentence in split_sentences(text):
        a = {**assess(claim, sentence, title, entities), "sentence": sentence}
        rel = a["relation"]
        if rel in ("supports", "contradicts"):
            if best[rel] is None or a["score"] > best[rel]["score"]:
                best[rel] = a
        elif rel == "uncertain" and a["judgeable"]:
            best["uncertain"].append(a)
        if rel != "unrelated" and (best["closest"] is None or a["score"] > best["closest"]["score"]):
            best["closest"] = a
    best["uncertain"].sort(key=lambda a: -a["score"])
    return best

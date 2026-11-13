import json
import os
import re
from typing import Dict, List

from tools.llm_clients import call_llm, search_candidates
from tools.logger import log_agent_event
from tools.evidence import independent_source_count, MIN_INDEPENDENT_SOURCES, source_domain
from tools.claim_merge import merge_duplicate_claims, expand_verdicts
from tools import evidence_check, semantic_judge, verification
from tools.claim_verifier import query_entities

AGENT_NAME = "fact_checking_agent"
MAX_FACTS_PER_BATCH = 15


def _evidence_lines(f: Dict) -> str:
    """Up to two sentences that state the claim and one that contradicts it, so the model judges from evidence
    instead of from a source count alone."""
    lines = []
    for s in [s for s in f.get("sources", []) if s.get("relation") == "supports" and s.get("evidence")][:2]:
        lines.append(f'     supports ({source_domain(s)}): "{s["evidence"][:160]}"')
    for c in (f.get("conflicting_evidence") or [])[:1]:
        lines.append(f'     CONFLICT ({c["domain"]}): "{c["evidence"][:160]}" [{c["reason"][:80]}]')
    return ("\n" + "\n".join(lines)) if lines else ""



def _build_fact_check_prompt(facts: List[Dict]) -> str:
    numbered = "\n".join(
        f'{i}. "{f["claim"]}" (independent sources: {independent_source_count(f.get("sources", []))})' + _evidence_lines(f)
        for i, f in enumerate(facts)
    )
    if os.getenv("DISPUTE_ON_CONFLICT", "false").lower() == "true":
        dispute_rule = ("contradicts another claim in this list, or a CONFLICT line shows a source stating different "
                        "figures or the opposite about the SAME fact")
    else:
        dispute_rule = ("directly contradicts another claim in this list. CONFLICT lines are rule-based flags shown for "
                        "information only: do NOT mark a claim DISPUTED because of a CONFLICT line")
    return f"""You are fact-checking a set of claims gathered during research. For EACH
numbered claim below, classify it as exactly one of:

- CONFIRMED: at least {MIN_INDEPENDENT_SOURCES} independent sources (see the count) and nothing in this list contradicts it
- DISPUTED: {dispute_rule}
- UNVERIFIED: fewer than {MIN_INDEPENDENT_SOURCES} independent sources, and no contradiction

Some claims show the source sentences found for them. Judge by meaning, figures, units and time period, not by
wording alone. A claim whose sentences give different figures than the claim is not confirmed.

Claims:
{numbered}

Respond with ONLY a JSON array: [{{"index": 0, "status": "CONFIRMED", "reasoning": "one short sentence"}}, ...]
"""
def _parse_fact_check_response(raw_text: str, expected_count: int) -> List[Dict]:
    cleaned = re.sub(r"^```(json)?", "", raw_text.strip()).strip()
    cleaned = re.sub(r"```$", "", cleaned).strip()
    try:
        parsed = json.loads(cleaned)
        if not isinstance(parsed, list):
            return []
        results = {}
        for item in parsed:
            idx = item.get("index")
            if isinstance(idx, int) and 0 <= idx < expected_count:
                status = item.get("status", "UNVERIFIED").upper()
                if status not in ("CONFIRMED", "DISPUTED", "UNVERIFIED"):
                    status = "UNVERIFIED"
                results[idx] = {"status": status, "reasoning": item.get("reasoning", "")}
        return [results.get(i, {"status": "UNVERIFIED", "reasoning": "not returned by model"})
                for i in range(expected_count)]
    except json.JSONDecodeError:
        return []


def apply_evidence_gate(fact: Dict, verdict: Dict) -> Dict:
    """Deterministic checks applied AFTER the LLM verdict, whatever the model said:
    - a claim cannot stay CONFIRMED without enough independent source domains;
    - EVIDENCE_GATE=strict: those domains must also have a sentence that states the claim (right figures,
      subject, period), so a source that merely mentions the topic does not count;
    - DISPUTE_ON_CONFLICT=true: a source sentence that contradicts the claim makes it DISPUTED."""
    count = independent_source_count(fact.get("sources", []))
    effective = count
    if os.getenv("EVIDENCE_GATE", "domains").lower() == "strict" and fact.get("supported_domains") is not None:
        effective = min(count, fact["supported_domains"])
    status, reasoning = verdict["status"], verdict["reasoning"]
    downgraded = False
    if status == "CONFIRMED" and effective < MIN_INDEPENDENT_SOURCES:
        status = "UNVERIFIED"
        why = (f"{count} independent source(s), need {MIN_INDEPENDENT_SOURCES}" if effective == count else
               f"only {effective} of {count} cited domains have a sentence that states the claim")
        reasoning = f"[downgraded: {why}] {reasoning}"
        downgraded = True
    conflicts = fact.get("conflicting_evidence") or []
    if conflicts and status != "DISPUTED" and os.getenv("DISPUTE_ON_CONFLICT", "false").lower() == "true":
        status = "DISPUTED"
        reasoning = f"[conflicting evidence: {conflicts[0]['domain']} - {conflicts[0]['reason'][:90]}] {reasoning}"
    return {**fact, "status": status, "reasoning": reasoning,
            "independent_sources": count, "downgraded": downgraded}


def fact_check_batch(facts: List[Dict]) -> List[Dict]:
    if not facts:
        return []
    try:
        raw_output = call_llm(_build_fact_check_prompt(facts))
        verdicts = _parse_fact_check_response(raw_output, len(facts))
        if not verdicts:
            raise ValueError("fact-check response could not be parsed")
    except Exception as e:
        log_agent_event(AGENT_NAME, "batch_failed", error=f"{type(e).__name__}: {e}", batch_size=len(facts))
        verdicts = [{"status": "UNVERIFIED", "reasoning": "fact-check call failed; defaulted to unverified"}
                    for _ in facts]

    checked = [apply_evidence_gate(f, v) for f, v in zip(facts, verdicts)]

    counts: Dict[str, int] = {}
    for c in checked:
        counts[c["status"]] = counts.get(c["status"], 0) + 1
    log_agent_event(AGENT_NAME, "batch_done", downgraded=sum(1 for c in checked if c["downgraded"]), **counts)
    return checked


def fact_check_facts(facts: List[Dict]) -> List[Dict]:
    all_checked = []
    for i in range(0, len(facts), MAX_FACTS_PER_BATCH):
        all_checked.extend(fact_check_batch(facts[i:i + MAX_FACTS_PER_BATCH]))
    return all_checked


def fact_checking_node(state: Dict) -> Dict:
    extracted_facts = state.get("extracted_facts", [])
    if not extracted_facts:
        return {
            "errors": [{"agent": AGENT_NAME, "error": "no extracted_facts to check"}],
            "model_usage_log": [{"agent": AGENT_NAME, "status": "skipped_no_input"}],
        }

    # Day 24: check each distinct claim once, with its sources pooled across sub-questions,
    # then hand every originating sub-question its own copy of the verdict.
    merged = merge_duplicate_claims(extracted_facts)
    log_agent_event(AGENT_NAME, "claims_merged", facts_in=len(extracted_facts), distinct_claims=len(merged))

    # Day 27: check the CITATIONS (does the cited sentence really state the claim?) and scan every snippet of the
    # run for CONTRADICTIONS, including ones from other sub-topics that the batched LLM check never sees.
    entities = query_entities(state.get("query", ""))
    records = evidence_check.snippet_records(state.get("search_results", []))
    merged = evidence_check.attach_citation_checks(merged, records, entities)
    merged = evidence_check.attach_conflicts(merged, records, entities)

    # Day 25: for under-sourced claims, search for corroboration on OTHER domains (opt-in).
    vstats = {}
    if verification.enabled():
        judge_fn, budget = None, [semantic_judge.MAX_JUDGE_CALLS]
        if semantic_judge.enabled():
            def judge_fn(claim, sentences):
                if budget[0] <= 0:
                    return []
                budget[0] -= 1
                return semantic_judge.judge(claim, sentences, call_llm)
        merged, vstats = verification.verify_claims(merged, search_candidates, entities=entities, judge_fn=judge_fn)
        merged = evidence_check.attach_citation_checks(merged, records, entities)   # new sources count as supported
        log_agent_event(AGENT_NAME, "verification_done", **vstats)

    cstats = evidence_check.citation_stats(merged)
    checked = expand_verdicts(fact_check_facts(merged))
    return {
        "fact_check_results": checked,
        "model_usage_log": [{"agent": AGENT_NAME, "provider": "groq", "facts_checked": len(merged),
                             "facts_in": len(extracted_facts), **{f"verify_{k}": v for k, v in vstats.items()},
                             **{f"citations_{k}": v for k, v in cstats.items()}}],
    }

"""Run the full pipeline on several different research questions and write the results to docs/ (Day 27).

    python benchmark.py                       # 3 built-in questions from different domains
    python benchmark.py --file eval/questions.json --pause 30

questions.json: [{"question": "...", "reference_claims": ["optional hand-written facts the report should contain"]}]
Without reference_claims you get automatic signals only (coverage, unverified breakdown, failures). With them you
also get reference-claim recall, which is the first real measure of 'did it find the important facts'.
Citation coverage is NOT citation correctness; correctness still needs hand checking of a sample."""
import argparse
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

from tools.evidence_trace import category_counts
from tools.verification import matching_sentence
from ui import runstats

DEFAULT_QUESTIONS = [
    {"question": "Compare battery recycling approaches of Tesla, Redwood Materials and CATL"},
    {"question": "What are the main techniques for reducing hallucinations in large language models, and how effective are they?"},
    {"question": "How do heat pumps compare with gas boilers for home heating in cold climates in terms of cost and emissions?"},
]


def reference_recall(reference_claims: List[str], results: List[Dict]) -> Dict:
    """A reference fact counts as recalled if some extracted claim states it (same figures, same polarity,
    enough shared words); 'confirmed' if that claim also passed the two-domain gate."""
    recalled = confirmed = 0
    for ref in reference_claims or []:
        hits = [r for r in results if matching_sentence(ref, r.get("claim", ""))]
        recalled += bool(hits)
        confirmed += any(r.get("status") == "CONFIRMED" for r in hits)
    n = len(reference_claims or [])
    return {"reference_claims": n, "recalled": recalled, "recall": round(recalled / n, 3) if n else None,
            "recalled_and_confirmed": confirmed}


def summarise_run(question: str, snap: Dict, reference_claims: Optional[List[str]] = None) -> Dict:
    results = snap.get("fact_check_results") or []
    cats, q, s, h = category_counts(results), runstats.quality_metrics(snap), runstats.run_stats(snap), runstats.run_health(snap)
    row = {"question": question, "status": h["state"], "issues": h["issues"], "claims": len(results),
           "confirmed": cats["confirmed"], "disputed": cats["disputed"],
           "unverified_insufficient": cats["insufficient_sources"], "unverified_inconclusive": cats["inconclusive"],
           "unverified_failed": cats["failed"], "citation_coverage": q["citation_coverage"],
           "evidence_sentence_coverage": q["evidence_sentence_coverage"], "unsupported_rate": q["unsupported_rate"],
           "topics_summarised": q["topics_summarised"], "topics_planned": q["topics_planned"],
           "llm_calls": s["llm_calls"], "output_tokens": s["output_tokens"], "unrecovered_failures": s["unrecovered_failures"],
           "seconds": round(s["wall_seconds"] or 0)}
    if reference_claims:
        row.update(reference_recall(reference_claims, results))
    return row


def to_markdown(rows: List[Dict]) -> str:
    out = [f"# Benchmark results", "", f"Generated {datetime.now():%Y-%m-%d %H:%M}. One run per question, free-tier APIs, "
           "so counts vary a little between runs.", "",
           "| Question | Status | Claims | Confirmed | Unverified (few sources / inconclusive / failed) | Citation cov. | Evidence-sentence cov. | Topics summarised | Failures | Time |",
           "|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        if "error" in r:
            out.append(f"| {r['question']} | **crashed** | - | - | - | - | - | - | {r['error']} | - |")
            continue
        out.append(f"| {r['question']} | {r['status']} | {r['claims']} | {r['confirmed']} | "
                   f"{r['unverified_insufficient']} / {r['unverified_inconclusive']} / {r['unverified_failed']} | "
                   f"{r['citation_coverage']:.0%} | {r['evidence_sentence_coverage']:.0%} | "
                   f"{r['topics_summarised']}/{r['topics_planned']} | {r['unrecovered_failures']} | {r['seconds']}s |")
    rec = [r for r in rows if r.get("recall") is not None]
    if rec:
        out += ["", "| Question | Reference claims | Recalled | Recall | Recalled and confirmed |", "|---|---|---|---|---|"]
        out += [f"| {r['question']} | {r['reference_claims']} | {r['recalled']} | {r['recall']:.0%} | {r['recalled_and_confirmed']} |" for r in rec]
    out += ["", "Citation coverage = claims with at least one source. Evidence-sentence coverage = claims where a stored "
            "source sentence states the claim (a proxy for citation correctness, not proof of it)."]
    return "\n".join(out) + "\n"


def run_one(question: str, graph, verify: bool):
    import os
    from tools import cache
    from tools.metrics import RunMetrics, evidence_stats
    from ui.snapshot import build_snapshot

    os.environ["VERIFY_BY_SEARCH"] = "true" if verify else "false"
    metrics = RunMetrics()
    state = {"query": question, "plan": [], "next_agent": "", "iteration_count": 0, "needs_more_research": False,
             "search_results": [], "extracted_facts": [], "fact_check_results": [], "summaries": [], "errors": [],
             "final_report": None, "model_usage_log": []}
    before, started = cache.stats(), time.time()
    final = graph.invoke(state, config={"callbacks": [metrics]})
    after = cache.stats()
    return build_snapshot(question, final, metrics=metrics.summary(), evidence=evidence_stats(final),
                          llm_errors=metrics.error_log, options={"verify": verify},
                          cache={k: after[k] - before[k] for k in after}, wall_seconds=time.time() - started)


def main() -> None:
    from dotenv import load_dotenv
    load_dotenv()
    from graph.build_graph import build_graph
    from ui import history

    ap = argparse.ArgumentParser()
    ap.add_argument("--file", help="JSON list of {question, reference_claims?}")
    ap.add_argument("--pause", type=int, default=20, help="seconds between runs (free-tier rate limits)")
    ap.add_argument("--no-verify", action="store_true")
    args = ap.parse_args()
    items = json.loads(Path(args.file).read_text(encoding="utf-8")) if args.file else DEFAULT_QUESTIONS

    graph, rows = build_graph(), []
    for i, item in enumerate(items):
        q = item["question"]
        print(f"[{i + 1}/{len(items)}] {q}")
        try:
            snap = run_one(q, graph, verify=not args.no_verify)
            history.save_snapshot(snap)
            rows.append(summarise_run(q, snap, item.get("reference_claims")))
            r = rows[-1]
            print(f"    {r['status']}: {r['confirmed']}/{r['claims']} confirmed, {r['seconds']}s")
        except Exception as e:   # one bad question must not lose the others
            rows.append({"question": q, "error": f"{type(e).__name__}: {str(e)[:120]}"})
            print(f"    crashed: {rows[-1]['error']}")
        if i < len(items) - 1:
            time.sleep(args.pause)

    Path("docs").mkdir(exist_ok=True)
    Path("docs/benchmark_results.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    Path("docs/benchmark_results.md").write_text(to_markdown(rows), encoding="utf-8")
    print("Wrote docs/benchmark_results.md and docs/benchmark_results.json")


if __name__ == "__main__":
    main()

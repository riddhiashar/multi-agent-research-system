"""Multi-Agent Research & Intelligence System - Streamlit front end (v3).

Run from the project root:   streamlit run app.py
Tabs: Report / Evidence Explorer / Analytics / Research History. All logic lives in tested modules
(ui/, tools/); this file only lays things out."""
import json
import os
import time

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

from tools import cache, guardrails, output_validator  # noqa: E402
from tools.evidence_trace import CATEGORY_LABEL, CATEGORY_ORDER, claim_funnel, trace_all  # noqa: E402
from tools.metrics import RunMetrics, evidence_stats  # noqa: E402
from ui import charts, compare, history, report, retry, runstats, views  # noqa: E402
from ui.progress import PipelineTracker, iter_updates, render_pipeline_html  # noqa: E402
from ui.snapshot import build_snapshot  # noqa: E402

st.set_page_config(page_title="Multi-Agent Research System", page_icon="🔬", layout="wide")

PRESETS = [
    "Compare battery recycling approaches of Tesla, Redwood Materials and CATL",
    "What are the safety risks of solid-state EV batteries?",
    "What is CATL's sodium-ion battery strategy and how does it compare to lithium-ion?",
]
REQUIRED_KEYS = ("GROQ_API_KEY", "GEMINI_API_KEY", "TAVILY_API_KEY")

st.markdown("""<style>
.hero h1{margin-bottom:0}
.hero p{opacity:.75;margin-top:.25rem}
.src{font-size:.88rem;margin:6px 0 0 0}
.quote{font-size:.82rem;opacity:.85;margin:2px 0 6px 18px;border-left:3px solid rgba(128,128,128,.35);padding-left:8px}
.qlabel{font-size:.72rem;opacity:.7;text-transform:uppercase;letter-spacing:.03em}
.tag{display:inline-block;padding:0 8px;margin-left:4px;border-radius:999px;font-size:.72rem;border:1px solid rgba(128,128,128,.4)}
</style>""", unsafe_allow_html=True)

st.session_state.setdefault("query", PRESETS[0])


def set_query(q: str) -> None:
    st.session_state["query"] = q


def rerun() -> None:
    (getattr(st, "rerun", None) or st.experimental_rerun)()


def open_run(path) -> None:
    st.session_state["snapshot"] = history.load_snapshot(path)
    st.session_state["snapshot_path"] = str(path)
    st.session_state.pop("retry_log", None)


def delete_run(path) -> None:
    history.delete_snapshot(path)
    if st.session_state.get("snapshot_path") == str(path):
        st.session_state.pop("snapshot_path", None)


@st.cache_resource
def get_graph():
    from graph.build_graph import build_graph
    return build_graph()


def show_chart(make, fallback) -> None:
    """Altair chart with a plain-Streamlit fallback, so a chart problem never breaks the page."""
    try:
        st.altair_chart(make())
    except Exception:
        fallback()


# ------------------------------------------------------------------ sidebar
with st.sidebar:
    st.markdown("### 🔬 Research Intelligence")
    st.caption("Supervisor-orchestrated multi-agent pipeline built with LangGraph.")
    st.markdown("**Example questions**")
    for i, q in enumerate(PRESETS):
        st.button(q, key=f"preset{i}", on_click=set_query, args=(q,))
    st.markdown("**Options**")
    verify = st.checkbox("Verify single-source claims by search", value=True,
                         help="Searches for a second independent source for a limited number of claims per run, "
                              "spread across sub-topics.")
    use_cache = st.checkbox("Cache search results (24h)", value=True, help="Repeated queries skip the search APIs.")
    strict = st.checkbox("Strict evidence check", value=False,
                         help="A claim is confirmed only if two independent domains each have a sentence that states "
                              "it (same figures, units, subject and period). Off: two cited domains are enough.")
    judge = st.checkbox("LLM judge for paraphrases", value=False,
                        help="Lets the model decide sentences the rules cannot (e.g. rephrased claims). It can never "
                             "override a figure, subject or negation mismatch. Costs a few extra tokens.")
    dispute = st.checkbox("Mark conflicting claims as disputed", value=False,
                          help="If a retrieved sentence states a different figure or the opposite, the claim becomes "
                               "DISPUTED. Conflicts are always shown either way.")
    st.caption("Saved runs are listed in the **Research History** tab.")

# ------------------------------------------------------------------ hero + input
st.markdown("""<div class="hero"><h1>Multi-Agent Research & Intelligence System</h1>
<p>Ask a research question. Specialised agents search, extract, fact-check, summarise and write a report,
and every claim is traced to its sources.</p></div>""", unsafe_allow_html=True)

st.text_area("Research question", key="query", height=80, max_chars=views.MAX_QUESTION_CHARS)
c_run, c_note = st.columns([1, 5])
run_clicked = c_run.button("▶ Run research", type="primary")
c_note.caption("A full run takes about 2–4 minutes on free-tier APIs. Cached searches are instant.")


# ------------------------------------------------------------------ run
def run_pipeline(query: str):
    missing = [k for k in REQUIRED_KEYS if not os.getenv(k)]
    if missing:
        st.error(f"Missing API keys in .env: {', '.join(missing)}")
        return None, None
    os.environ["VERIFY_BY_SEARCH"] = "true" if verify else "false"
    os.environ["SEARCH_CACHE"] = "true" if use_cache else "false"
    os.environ["EVIDENCE_GATE"] = "strict" if strict else "domains"
    os.environ["SEMANTIC_JUDGE"] = "true" if judge else "false"
    os.environ["DISPUTE_ON_CONFLICT"] = "true" if dispute else "false"
    try:
        graph = get_graph()
    except Exception as e:
        st.error(f"Could not start the graph: {type(e).__name__}: {e}")
        return None, None

    metrics, tracker = RunMetrics(), PipelineTracker()
    initial_state = {
        "query": query, "plan": [], "next_agent": "", "iteration_count": 0,
        "needs_more_research": False, "search_results": [], "extracted_facts": [],
        "fact_check_results": [], "summaries": [], "errors": [], "final_report": None,
        "model_usage_log": [],
    }
    holder, bar = st.empty(), st.progress(0.0)
    tracker.start()
    holder.markdown(render_pipeline_html(tracker.snapshot()), unsafe_allow_html=True)
    cache_before, started, final = cache.stats(), time.time(), None
    try:
        with st.expander("Agent activity", expanded=True):
            for mode, chunk in graph.stream(initial_state, config={"callbacks": [metrics]},
                                            stream_mode=["updates", "values"]):
                if mode == "values":
                    final = chunk
                    continue
                for node, update in iter_updates(chunk):
                    text = tracker.on_update(node, update)
                    st.write(f"`{tracker.events[-1]['t']:>6.1f}s`  **{tracker.events[-1]['label']}** — {text}")
                    holder.markdown(render_pipeline_html(tracker.snapshot()), unsafe_allow_html=True)
                    bar.progress(tracker.fraction_done())
    except Exception as e:
        hint = " The free-tier model is rate limited; wait a minute and retry." if "429" in str(e) else ""
        st.error(f"Run failed: {type(e).__name__}: {e}.{hint}")
        return None, None
    bar.progress(1.0)
    if not final:
        st.error("The graph finished without returning a final state.")
        return None, None

    cache_after = cache.stats()
    snapshot = build_snapshot(
        query, final, metrics=metrics.summary(), evidence=evidence_stats(final), llm_errors=metrics.error_log,
        stages=tracker.snapshot(), events=tracker.events,
        options={"verify": verify, "cache": use_cache, "strict": strict, "judge": judge, "dispute": dispute},
        cache={k: cache_after[k] - cache_before[k] for k in cache_after}, wall_seconds=time.time() - started)
    snapshot = output_validator.finalize_snapshot(snapshot, live_fetch=live_url_check())   # sanitise + validate + repair
    return snapshot, history.save_snapshot(snapshot)


def live_url_check():
    return output_validator.http_head if os.getenv("VALIDATE_URLS_LIVE", "false").lower() == "true" else None


if run_clicked:
    check = guardrails.check_input(st.session_state["query"])
    if not check["ok"]:
        st.warning(check["error"])
    elif not (limiter := guard.RunLimiter()).allowed():
        st.warning("Today's run limit for this demo has been reached. Saved runs can still be opened in Research History.")
    else:
        limiter.record()
        snap_new, path_new = run_pipeline(check["clean"])
        if snap_new:
            st.session_state["snapshot"], st.session_state["snapshot_path"] = snap_new, str(path_new)
            st.session_state.pop("retry_log", None)


# ------------------------------------------------------------------ tab: report
def validation_panel(snap: dict) -> None:
    v = snap.get("validation") or output_validator.validate_snapshot(snap)
    s = v.get("stats") or {}
    if not v["issues"]:
        st.success("Output validation passed: every link, figure and confirmed/disputed statement in the generated text "
                   "agrees with this run's evidence. (Rule-based check, not proof that every sentence is true.)")
    elif v["ok"]:
        st.info(f"Output validation: {s.get('warnings', 0)} warning(s), no blocking problems.")
    else:
        st.warning(f"Output validation found {s.get('errors', 0)} problem(s) in the generated text. Unverifiable links were "
                   "removed and the issues are listed below and at the end of the narrative.")
    acts = v.get("guardrail_actions") or []
    if acts:
        st.caption("Output guardrails applied: " + ", ".join(f"{a['type'].replace('_', ' ')} x{a['count']}" for a in acts))
    if v["issues"]:
        with st.expander(f"Validation details ({len(v['issues'])})"):
            for i in v["issues"]:
                icon = "🔴" if i["severity"] == "error" else "🟡"
                st.markdown(f"{icon} **{i['type'].replace('_', ' ')}** ({i['where']}): {i['detail']}")
                st.caption(f"“{i['text']}”")


def tab_report(snap: dict) -> None:
    doc = report.build_report(snap)
    validation_panel(snap)
    left, right = st.columns(2)
    for i, (k, v) in enumerate(doc["info"]):
        (left if i % 2 == 0 else right).markdown(f"**{k}:** {v}")

    md = report.to_markdown(doc)
    d = st.columns(4)
    d[0].download_button("⬇ Markdown", md, file_name="research_report.md", mime="text/markdown")
    try:
        d[1].download_button("⬇ PDF", report.to_pdf(doc), file_name="research_report.pdf", mime="application/pdf")
    except ImportError:
        d[1].caption("PDF needs `pip install reportlab`")
    except Exception as e:  # a PDF problem must never hide the report itself
        d[1].caption(f"PDF unavailable: {type(e).__name__}")
    d[2].download_button("⬇ Claims + evidence (JSON)", json.dumps(report.claims_export(snap), indent=2),
                         file_name="claims_evidence.json", mime="application/json")
    d[3].download_button("⬇ Evidence table (CSV)", report.evidence_csv(snap), file_name="evidence.csv", mime="text/csv")

    term = st.text_input("Search within the report", placeholder="e.g. cobalt").strip().lower()
    if term:
        hits = [ln.strip() for ln in md.splitlines() if term in ln.lower() and not ln.startswith("|---")]
        st.caption(f"{len(hits)} matching line(s)")
        for ln in hits[:30]:
            st.markdown(f"- {ln.lstrip('-# ')}")

    for sec in doc["sections"]:
        if sec["title"] in ("Evidence and citations", "Generated narrative report"):
            with st.expander(sec["title"], expanded=sec["title"] == "Generated narrative report"):
                st.markdown(report.section_markdown(sec, level=3))
        else:
            st.markdown(report.section_markdown(sec))


# ------------------------------------------------------------------ tab: evidence explorer
def tab_evidence(snap: dict) -> None:
    results = snap.get("fact_check_results") or []
    if not results:
        st.info("No claims were produced in this run.")
        return
    traced = trace_all(results, snap.get("search_results") or [], snap.get("query", ""))
    ev = snap.get("evidence") or {}
    f = claim_funnel(ev.get("extracted_claims", 0), results, snap.get("usage_log") or [])
    counts = views.status_counts(results)
    cats = {c: sum(1 for t in traced if t["category"] == c) for c in CATEGORY_ORDER}

    st.caption("Two independent domains are not always two independent pieces of research: two sites can repeat "
               "the same upstream figure. Evidence sentences are shown so you can judge that yourself.")

    a, b = st.columns(2)
    with a:
        st.markdown("**Where the numbers come from**")
        line = f"{f['extracted']} claims were extracted" if f["extracted"] else "Extracted-claim count was not saved"
        st.markdown(f"- {line}; duplicates across sub-topics are merged before checking, giving **{f['distinct']} distinct claims**.\n"
                    f"- A merged claim is shown once for each sub-topic that raised it, so **{f['rows']} claims** are checked "
                    f"and listed below" + (f" ({f['merged_away']} fewer than extracted)." if f["merged_away"] else "."))
    with b:
        st.markdown("**What verification search did**")
        if f["verification_recorded"]:
            st.markdown(f"- {f['initially_under_supported']} distinct claims started with fewer than 2 independent domains.\n"
                        f"- {f['verification_tried']} were searched for a second source (limited per-run budget).\n"
                        f"- **{f['verification_upgraded']}** reached 2 independent domains; {f['verification_sources_added']} sources added"
                        + (f"; {f['verification_failed']} searches errored." if f["verification_failed"] else "."))
        else:
            st.caption("Verification statistics were not recorded for this run.")

    with st.expander("How each source is checked"):
        st.markdown("A source counts as **supporting** a claim only if one of its sentences passes these checks, done in code "
                    "with no LLM:\n- **Figures with units**: 95% is not 95 GWh; 9.5 million equals 9,500,000; a figure inside a "
                    "claimed range counts; a different figure is a conflict.\n- **Subject**: names in the claim (e.g. Tesla) "
                    "must appear, so a sentence about CATL does not support a claim about Tesla.\n- **Meaning**: negation and "
                    "direction (increase vs decrease) must agree.\n- **Scope**: the time period (by 2030 vs 2035) and bounds "
                    "(up to vs at least) must be compatible.\nSentences that fail are shown as *related but not conclusive* or "
                    "*contradicts*. An optional LLM judge may rescue paraphrases, but never overrides these checks.")

    st.markdown("**Why claims got their verdict**")
    st.dataframe(pd.DataFrame([{"Verdict": CATEGORY_LABEL[c], "Claims": cats[c], "Share": f"{cats[c] / len(results):.0%}"}
                               for c in CATEGORY_ORDER]), hide_index=True)
    if cats["disputed"] == 0:
        st.caption("**0 disputed** means no two claims in the same fact-check batch contradicted each other. It does not "
                   "prove no contradictions exist: claims in different batches are never compared.")

    c1, c2 = st.columns([2, 3])
    with c1:
        st.markdown("**Verdict distribution**")
        st.caption(f"Number of the {len(results)} checked claims in each verdict. Zero values are shown.")
        rows = charts.verdict_rows(counts)
        show_chart(lambda: charts.verdict_chart(counts), lambda: st.bar_chart(pd.DataFrame(rows).set_index("Verdict")))
    with c2:
        st.markdown("**Claims per sub-topic**")
        st.caption("Each bar is one sub-topic, split by verdict. Hover for exact counts.")
        by_topic = views.status_by_topic(results)
        show_chart(lambda: charts.topic_chart(by_topic),
                   lambda: st.bar_chart(pd.DataFrame(by_topic).set_index("topic")))

    st.markdown("**Claims**")
    f1, f2, f3, f4 = st.columns([3, 3, 1.5, 2])
    labels = {CATEGORY_LABEL[c]: c for c in CATEGORY_ORDER}
    chosen = f1.multiselect("Verdict", list(labels), default=list(labels))
    topics = views.topic_options(results)
    topic = f2.selectbox("Sub-topic", [None] + topics, format_func=lambda q: "All" if q is None else views.short_topic(q, 60))
    verified_only = f3.checkbox("Verified by search")
    text = f4.text_input("Search claims").strip().lower()
    shown = [t for t in traced if t["category"] in {labels[x] for x in chosen} and (topic is None or t["topic"] == topic)
             and (not verified_only or t["verified_by_search"]) and (not text or text in t["claim"].lower())]
    st.caption(f"Showing {len(shown)} of {len(traced)} claims")

    for t in shown:
        with st.expander(f"{views.STATUS_BADGE.get(t['status'], '⚪')} {t['claim']}"):
            st.markdown(f"**{t['category_label']}** — {t['explanation']}")
            ver = {"added": "Yes: a source was added", "no_match": "Searched, nothing found", "search_failed": "Search errored"}.get(
                t["verification"], "Not searched")
            st.markdown(f"- Independent domains: **{t['independent_domains']}**\n"
                        f"- Verification by search: {ver}\n"
                        f"- Original research sources: {t['original_sources']}\n"
                        f"- Additional verification sources: {t['verification_sources']}"
                        + (f"\n- Cited domains whose sentence states the claim: **{t['supported_domains']}**"
                           if t.get("supported_domains") is not None else "")
                        + (f"\n- Merged from {t['merged_count']} duplicate claims" if t["merged_count"] > 1 else "")
                        + ("\n- Downgraded by the evidence gate (model said confirmed)" if t["downgraded"] else ""))
            if t["shared_origin_warning"]:
                st.warning("Two sources use near-identical wording. They may repeat one upstream statement, so treat "
                           "this as weaker than two independent studies.")
            if t.get("conflicts"):
                st.error("Possible conflicting evidence (rule-based, check the sentence yourself):")
                for c in t["conflicts"]:
                    st.markdown(f"- **{c['domain']}**: “{c['evidence']}” — {c['reason']}")
            if t["sources"]:
                st.markdown("**Sources and the sentence checked against the claim**")
                for s in t["sources"]:
                    st.markdown(views.source_card_html(s), unsafe_allow_html=True)
            else:
                st.caption("No sources were recorded for this claim.")


# ------------------------------------------------------------------ tab: analytics
def tab_analytics(snap: dict) -> None:
    s, q, m = runstats.run_stats(snap), runstats.quality_metrics(snap), snap.get("metrics") or {}
    st.markdown("**Pipeline run**")
    st.markdown(render_pipeline_html(snap.get("stages") or []), unsafe_allow_html=True)

    st.markdown("**Usage**")
    r1 = st.columns(4)
    r1[0].metric("LLM calls", s["llm_calls"], help="Every model call, including failed attempts and retries.")
    r1[1].metric("Input tokens", f"{s['input_tokens']:,}")
    r1[2].metric("Output tokens", f"{s['output_tokens']:,}")
    r1[3].metric("Failed LLM calls", s["failed_llm_calls"],
                 help=f"Includes {s['rate_limited_calls']} rate-limit (429) errors. Many are retried automatically, so a "
                      "failed call is not always a lost result.")
    r2 = st.columns(4)
    r2[0].metric("Search requests", s["search_requests"], help="One per sub-question.")
    r2[1].metric("Verification searches", s["verification_searches"], help="Extra searches for a second source.")
    r2[2].metric("Sources retrieved", s["sources_retrieved"], help=f"Distinct URLs seen; {s['sources_cited']} are cited by claims.")
    r2[3].metric("Unrecovered failures", s["unrecovered_failures"],
                 help="Failed summaries, claims that could not be checked, errored verification searches and agent errors.")
    r3 = st.columns(4)
    r3[0].metric("Cache hits", s["cache_hits"], help="Searches answered from the 24h disk cache.")
    r3[1].metric("Cache misses", s["cache_misses"], help="Searches sent to a provider.")
    r3[2].metric("Peak output tokens / 60s", s["peak_output_60s"],
                 help="Estimate of output-token activity in the busiest 60-second window. It is not the provider's "
                      "exact quota measurement.")
    r3[3].metric("Run time", f"{s['wall_seconds']:.0f}s")

    st.markdown("**Research quality signals**")
    st.caption("Computed automatically from this run. Citation coverage only shows a source is attached; evidence-sentence "
               "coverage shows a stored sentence states the claim. Neither proves correctness: that needs hand-labelled data.")
    qc = st.columns(6)
    qc[0].metric("Citation coverage", f"{q['citation_coverage']:.0%}", help="Claims with at least one source.")
    qc[1].metric("Evidence-sentence coverage", f"{q['evidence_sentence_coverage']:.0%}",
                 help="Claims where a stored source sentence states the claim.")
    qc[2].metric("2+ independent domains", f"{q['multi_domain_rate']:.0%}")
    qc[3].metric("Unsupported claim rate", f"{q['unsupported_rate']:.0%}", help="Claims with fewer than 2 independent domains.")
    qc[4].metric("Topics summarised", f"{q['topics_summarised']}/{q['topics_planned']}")
    qc[5].metric("Cited sentences that state the claim", f"{q['citation_support_rate']:.0%}",
                 help=f"Share of checkable citations whose sentence matches the claim's figures, subject and period. "
                      f"{q['claims_with_conflicts']} claim(s) have possibly conflicting evidence.")

    node_rows = views.node_rows(m)
    if node_rows:
        df = pd.DataFrame(node_rows)
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Time per agent (seconds)**")
            show_chart(lambda: charts.metric_bar(node_rows, "node", "seconds", "Seconds"),
                       lambda: st.bar_chart(df.set_index("node")["seconds"]))
        with c2:
            st.markdown("**Output tokens per agent**")
            show_chart(lambda: charts.metric_bar(node_rows, "node", "output_tokens", "Output tokens"),
                       lambda: st.bar_chart(df.set_index("node")["output_tokens"]))

    with st.expander("Technical details (errors and activity log)"):
        for node, kind, msg in snap.get("llm_errors") or snap.get("errors") or []:
            st.write(f"LLM error in `{node}` ({kind}): {msg}")
        for e in snap.get("state_errors") or []:
            st.write(f"Agent error: {e}")
        for e in snap.get("events") or []:
            st.write(f"`{e['t']:>6.1f}s`  **{e['label']}** — {e['text']}")


# ------------------------------------------------------------------ tab: history
def show_comparison(c: dict) -> None:
    if not c["same_question"]:
        st.caption("These runs answer different questions, so claim-level differences are expected.")
    st.dataframe(pd.DataFrame(c["table"]).rename(columns={"metric": "Metric", "run_a": "Run A", "run_b": "Run B", "change": "Change"}),
                 hide_index=True)
    k = st.columns(5)
    k[0].metric("New claims", len(c["new_claims"]))
    k[1].metric("Dropped claims", len(c["dropped_claims"]))
    k[2].metric("Verdict changed", len(c["status_changed"]))
    k[3].metric("Gained evidence", len(c["gained_evidence"]))
    k[4].metric("Lost evidence", len(c["lost_evidence"]))
    for title, items, fmt in (
            ("New in run B", c["new_claims"], lambda r: f"{r['claim']} ({r.get('status')})"),
            ("Only in run A", c["dropped_claims"], lambda r: f"{r['claim']} ({r.get('status')})"),
            ("Verdict changed", c["status_changed"], lambda r: f"{r['claim']}: {r['from']} → {r['to']}"),
            ("Gained independent domains", c["gained_evidence"], lambda r: f"{r['claim']}: {r['from']} → {r['to']}"),
            ("Lost independent domains", c["lost_evidence"], lambda r: f"{r['claim']}: {r['from']} → {r['to']}")):
        if items:
            with st.expander(f"{title} ({len(items)})"):
                for r in items:
                    st.markdown(f"- {fmt(r)}")
    if c["new_sources"]:
        with st.expander(f"New sources in run B ({len(c['new_sources'])})"):
            for u in c["new_sources"]:
                st.markdown(f"- {u}")


def tab_history() -> None:
    snaps = history.list_snapshots()
    if not snaps:
        st.info("No saved runs yet. Every completed run is saved here automatically.")
        return
    names = [str(s["path"]) for s in snaps]
    label = lambda p: next(f"{s['created_at'][:16].replace('T', ' ')} · {s['state'].capitalize()} · {s['confirmed']}/{s['claims']} confirmed · "
                           f"{views.short_topic(s['query'], 50)}" for s in snaps if str(s["path"]) == p)
    st.dataframe(pd.DataFrame([{"Date": s["created_at"][:16].replace("T", " "), "Question": views.short_topic(s["query"], 80),
                                "Claims": s["claims"], "Confirmed": s["confirmed"], "Status": s["state"].capitalize()}
                               for s in snaps]), hide_index=True)

    st.markdown("**Open, download or delete a run**")
    pick = st.selectbox("Run", names, format_func=label, key="hist_pick")
    b1, b2, b3 = st.columns(3)
    b1.button("Open this run", on_click=open_run, args=(pick,))
    try:
        picked = history.load_snapshot(pick)
        b2.download_button("⬇ Download this report (.md)", report.to_markdown(report.build_report(picked)),
                           file_name="research_report.md", mime="text/markdown", key="hist_dl")
    except (OSError, ValueError):
        b2.caption("This run could not be read.")
    if b3.checkbox("Confirm delete", key=f"confirm_{pick}"):
        b3.button("🗑 Delete this run", on_click=delete_run, args=(pick,))

    st.markdown("**Compare two runs**")
    if len(names) < 2:
        st.caption("Run research at least twice to compare.")
        return
    ca, cb = st.columns(2)
    run_a = ca.selectbox("Run A (older)", names, index=min(1, len(names) - 1), format_func=label, key="cmp_a")
    run_b = cb.selectbox("Run B (newer)", names, index=0, format_func=label, key="cmp_b")
    if st.button("Compare runs"):
        if run_a == run_b:
            st.warning("Pick two different runs.")
        else:
            try:
                show_comparison(compare.compare_runs(history.load_snapshot(run_a), history.load_snapshot(run_b)))
            except (OSError, ValueError):
                st.error("One of the runs could not be read.")


# ------------------------------------------------------------------ retry
def do_retry(snap: dict) -> None:
    from agents.fact_checking_agent import fact_check_facts
    from agents.report_generator_agent import generate_report
    from agents.summarization_agent import summarize_sub_topic
    from tools import verification
    from tools.llm_clients import search_candidates

    os.environ["VERIFY_BY_SEARCH"] = "true"
    with st.spinner("Retrying only the parts that failed…"):
        out = retry.retry_failures(
            snap, summarize_fn=summarize_sub_topic, fact_check_fn=fact_check_facts, report_fn=generate_report,
            verify_fn=lambda row: verification.verify_claim(row, search_candidates))
        new_snap = out["snapshot"]
    if new_snap.get("final_report") != snap.get("final_report"):     # report was regenerated: validate the new raw text
        new_snap.pop("final_report_raw", None)
        for s_ in new_snap.get("summaries") or []:
            s_.pop("summary_raw", None)
    new_snap = output_validator.finalize_snapshot(new_snap, live_fetch=live_url_check())
    out["snapshot"] = new_snap
    path = st.session_state.get("snapshot_path")
    if path:
        history.overwrite_snapshot(path, new_snap)
    st.session_state["snapshot"] = new_snap
    st.session_state["retry_log"] = out
    rerun()


# ------------------------------------------------------------------ page body
snapshot = st.session_state.get("snapshot")
if snapshot:
    results = snapshot.get("fact_check_results") or []
    counts = views.status_counts(results)
    health = runstats.run_health(snapshot)
    st.divider()
    st.markdown(f"#### Results for: *{snapshot.get('query', '')}*")
    if health["state"] == "complete":
        st.success("Run completed: every stage finished.")
    else:
        msg = "Run finished with problems: " if health["state"] == "partial" else "Run failed: "
        st.warning(msg + " ".join(health["issues"]))
        n_fail = retry.count_failures(snapshot)
        if n_fail and st.button(f"🔁 Retry {n_fail} failed part(s)", help="Re-runs only what failed and rebuilds the report."):
            do_retry(snapshot)
    last = st.session_state.get("retry_log")
    if last:
        st.info(f"Retry: fixed {last['fixed']}, {last['remaining']} still failing. " + " ".join(last["log"]))

    k = st.columns(4)
    k[0].metric("Claims checked", len(results))
    k[1].metric("Confirmed", counts["CONFIRMED"], f"{views.confirmation_rate(results):.0%} of claims", delta_color="off")
    k[2].metric("Verification upgrades", (snapshot.get("evidence") or {}).get("claims_verified_by_search", 0),
                help="Claims that gained a source through the targeted verification search.")
    k[3].metric("Runtime", f"{runstats.run_stats(snapshot)['wall_seconds']:.0f}s")
else:
    st.info("Pick an example or type a question, then press **Run research**. Past runs are in the Research History tab.")

t_report, t_evidence, t_analytics, t_history = st.tabs(["📄 Report", "🔍 Evidence Explorer", "📊 Analytics", "🕘 Research History"])
with t_report:
    tab_report(snapshot) if snapshot else st.caption("The full report and its downloads appear here after a run.")
with t_evidence:
    tab_evidence(snapshot) if snapshot else st.caption("Every claim, its sources and the sentence that backs it appear here after a run.")
with t_analytics:
    tab_analytics(snapshot) if snapshot else st.caption("Tokens, latency, cache and quality signals appear here after a run.")
with t_history:
    tab_history()

with st.expander("How this system works"):
    st.markdown("""A Supervisor splits the question into sub-questions and routes work by data dependency. Searches run in
parallel, then extraction, fact-checking, summarisation and report generation run in order. Every stage records its
outcome, so a failure degrades one result instead of silently dropping it, and failed parts can be retried alone.

A claim is **confirmed** only when sources from at least two independent domains back it (social platforms excluded).
Duplicate claims are merged first. Under-sourced claims get a targeted search for a second domain, and a new source is
accepted only if one of its sentences states the same figures. A code-enforced gate downgrades any confirmed verdict
that lacks two domains, whatever the model said.""")
    st.graphviz_chart(views.GRAPH_DOT)

"""Pure data-shaping helpers for the UI (Day 26) - no Streamlit import, fully unit-tested."""
import html as _html
from typing import Dict, List, Optional

STATUS_ORDER = ["CONFIRMED", "DISPUTED", "UNVERIFIED"]
STATUS_BADGE = {"CONFIRMED": "🟢", "DISPUTED": "🔴", "UNVERIFIED": "🟡"}

GRAPH_DOT = """
digraph G {
  rankdir=LR; bgcolor="transparent";
  node [shape=box, style="rounded,filled", fillcolor="#eef2ff", color="#6366f1", fontname="Helvetica"];
  edge [color="#6b7280", fontname="Helvetica", fontsize=10];
  start [label="Question", shape=oval, fillcolor="#f3f4f6", color="#9ca3af"];
  supervisor [label="Supervisor\\n(plan + route)", fillcolor="#e0e7ff"];
  search [label="Web search\\n(parallel fan-out)"];
  extract [label="Data extraction"];
  verify [label="Fact-checking\\n(merge + verify + gate)"];
  summarize [label="Summarization"];
  report [label="Report generator"];
  end [label="Report + evidence", shape=oval, fillcolor="#dcfce7", color="#22c55e"];
  start -> supervisor;
  supervisor -> search [label="Send ×N"];
  supervisor -> extract; supervisor -> verify; supervisor -> summarize; supervisor -> report;
  search -> supervisor [style=dashed]; extract -> supervisor [style=dashed];
  verify -> supervisor [style=dashed]; summarize -> supervisor [style=dashed];
  report -> end;
}
"""


def short_topic(question: str, n: int = 52) -> str:
    text = " ".join((question or "").split())
    if len(text) <= n:
        return text
    return (text[:n].rsplit(" ", 1)[0] or text[:n]).rstrip(",;:- ") + "…"


def topic_options(results: List[Dict]) -> List[str]:
    """Distinct sub-questions in order of first appearance."""
    seen, out = set(), []
    for r in results:
        q = r.get("sub_question")
        if q and q not in seen:
            seen.add(q)
            out.append(q)
    return out


def claim_rows(results: List[Dict]) -> List[Dict]:
    rows = []
    for r in results:
        rows.append({
            "claim": r.get("claim", ""),
            "status": r.get("status", "UNVERIFIED"),
            "topic": r.get("sub_question", ""),
            "domains": r.get("independent_sources", 0),
            "sources": list(r.get("sources") or []),
            "verified": bool(r.get("verified_by_search")),
            "merged": r.get("merged_count", 1),
            "reasoning": r.get("reasoning", ""),
            "downgraded": bool(r.get("downgraded")),
        })
    return rows


def filter_rows(rows: List[Dict], statuses: Optional[List[str]] = None, topic: Optional[str] = None,
                verified_only: bool = False, text: str = "") -> List[Dict]:
    text = (text or "").strip().lower()
    out = []
    for r in rows:
        if statuses and r["status"] not in statuses:
            continue
        if topic and r["topic"] != topic:
            continue
        if verified_only and not r["verified"]:
            continue
        if text and text not in r["claim"].lower():
            continue
        out.append(r)
    return out


def status_counts(results: List[Dict]) -> Dict[str, int]:
    counts = {s: 0 for s in STATUS_ORDER}
    for r in results:
        counts[r.get("status", "UNVERIFIED")] = counts.get(r.get("status", "UNVERIFIED"), 0) + 1
    return counts


def confirmation_rate(results: List[Dict]) -> float:
    return (status_counts(results)["CONFIRMED"] / len(results)) if results else 0.0


def status_by_topic(results: List[Dict]) -> List[Dict]:
    """One row per sub-topic: {"topic": "Q1: …", "CONFIRMED": n, "DISPUTED": n, "UNVERIFIED": n}."""
    rows = []
    for i, q in enumerate(topic_options(results), start=1):
        row = {"topic": f"Q{i}: {short_topic(q, 36)}", **{s: 0 for s in STATUS_ORDER}}
        for r in results:
            if r.get("sub_question") == q:
                st_ = r.get("status", "UNVERIFIED")
                row[st_] = row.get(st_, 0) + 1
        rows.append(row)
    return rows


def node_rows(metrics_summary: Dict) -> List[Dict]:
    rows = []
    for node, n in (metrics_summary.get("nodes") or {}).items():
        seconds = n.get("node_seconds") or n.get("llm_seconds") or 0.0
        if seconds or n.get("calls"):
            rows.append({"node": node, "seconds": round(seconds, 1), "output_tokens": n.get("output", 0),
                         "llm_calls": n.get("calls", 0), "rate_limited": n.get("rate_limited", 0)})
    return rows


def source_html(source: Dict) -> str:
    """HTML for one source line. Titles/URLs/snippets come from the open web, so everything is
    escaped and only http(s) links are allowed."""
    raw_url = source.get("url", "") or ""
    url = _html.escape(raw_url, quote=True) if raw_url.startswith(("http://", "https://")) else "#"
    title = _html.escape(source.get("title") or raw_url or "source")
    tag = " <em>(verification)</em>" if source.get("via") == "verification" else ""
    out = f'<div class="src">• <a href="{url}" target="_blank" rel="noopener noreferrer">{title}</a>{tag}</div>'
    if source.get("evidence"):
        out += f'<div class="quote">“{_html.escape(source["evidence"])}”</div>'
    return out


MIN_QUESTION_CHARS, MAX_QUESTION_CHARS = 10, 500
_ORIGIN_LABEL = {"search": "original search", "verification": "targeted verification"}
_MATCH_LABEL = {"exact": "Sentence that states the claim", "contradicts": "Sentence that contradicts the claim",
                "closest": "Closest sentence in the snippet (does not fully state the claim)",
                "none": "No relevant sentence in the stored snippet", "unavailable": "Snippet text was not saved for this run"}


def validate_question(raw: str):
    """(clean_question, error). Strips control characters and extra whitespace; bounds the length so an empty
    or huge input never reaches the paid APIs."""
    import re
    clean = " ".join(re.sub(r"[\x00-\x1f\x7f]", " ", raw or "").split())
    if len(clean) < MIN_QUESTION_CHARS:
        return clean, f"Please enter a research question of at least {MIN_QUESTION_CHARS} characters."
    if len(clean) > MAX_QUESTION_CHARS:
        return clean, f"Please keep the question under {MAX_QUESTION_CHARS} characters."
    return clean, None


def source_card_html(src: Dict) -> str:
    """One traced source (from evidence_trace.trace_source): clickable title, domain, origin, evidence sentence.
    Everything from the web is escaped; only http(s) links are linkable."""
    raw = src.get("url", "") or ""
    url = _html.escape(raw, quote=True) if raw.startswith(("http://", "https://")) else "#"
    title = _html.escape(src.get("title") or raw or "source")
    domain = _html.escape(src.get("domain") or "")
    origin = _ORIGIN_LABEL.get(src.get("origin"), "original search")
    flag = "" if src.get("independent", True) else " · <em>not counted as independent</em>"
    out = (f'<div class="src"><a href="{url}" target="_blank" rel="noopener noreferrer">{title}</a> '
           f'<span class="tag">{domain}</span> <span class="tag">{origin}</span>{flag}</div>')
    if src.get("evidence"):
        why = ""
        if src.get("reasons"):
            why = f'<br><span class="qlabel">Checks: {_html.escape("; ".join(src["reasons"])[:200])}</span>'
        out += (f'<div class="quote"><span class="qlabel">{_MATCH_LABEL.get(src.get("match"), "")}</span><br>'
                f'“{_html.escape(src["evidence"])}”{why}</div>')
    else:
        out += f'<div class="quote qlabel">{_MATCH_LABEL.get(src.get("match"), "")}</div>'
    return out

"""Structured research report + exports (Day 27). Built deterministically from a saved run snapshot so every
number in it is real; the LLM-written narrative is included as its own section. Pure except `to_pdf`, which
needs reportlab (pip install reportlab)."""
import csv
import io
import re
from datetime import datetime
from typing import Dict, List
from xml.sax.saxutils import escape

from tools.evidence_trace import CATEGORY_LABEL, category_counts, claim_funnel, trace_all
from ui import views
from ui.runstats import quality_metrics, run_health, run_stats


def _ref_map(traced: List[Dict]) -> Dict[str, int]:
    refs: Dict[str, int] = {}
    for t in traced:
        for s in t["sources"]:
            if s["url"] and s["url"] not in refs:
                refs[s["url"]] = len(refs) + 1
    return refs


def build_report(snap: Dict) -> Dict:
    results = snap.get("fact_check_results") or []
    traced = trace_all(results, snap.get("search_results") or [], snap.get("query", ""))
    refs = _ref_map(traced)
    counts = views.status_counts(results)
    cats = category_counts(results)
    stats, quality, health = run_stats(snap), quality_metrics(snap), run_health(snap)
    ev = snap.get("evidence") or {}
    funnel = claim_funnel(ev.get("extracted_claims", 0), results, snap.get("usage_log") or [])
    topics = views.topic_options(results) or [s.get("sub_question", "") for s in snap.get("summaries") or []]
    n = len(results)

    info = [("Research question", snap.get("query", "")),
            ("Generated", (snap.get("created_at") or "").replace("T", " ")),
            ("Sub-topics researched", str(len(topics))), ("Claims checked", str(n)),
                        ("Sources cited", str(len(refs))), ("Run status", health["state"].capitalize())]
    validation = snap.get("validation")
    if validation:
        from tools.output_validator import summary_line
        info.append(("Output validation", summary_line(validation)))

    S: List[Dict] = []
    # 1 executive summary
    top = sorted((t for t in traced if t["category"] == "confirmed"),
                 key=lambda t: (-t["independent_domains"], not t["verified_by_search"]))[:5]
    p = (f"This report addresses the question above using {len(topics)} sub-topics. {n} claims were fact-checked: "
         f"{counts['CONFIRMED']} confirmed by at least two independent domains, {counts['DISPUTED']} disputed and "
         f"{counts['UNVERIFIED']} unverified.")
    blocks = [("p", p)]
    if top:
        blocks += [("p", "Key confirmed findings:"), ("ul", [t["claim"] for t in top])]
    else:
        blocks.append(("p", "No claim reached two independent domains, so every finding below should be treated as "
                            "an unconfirmed lead rather than an established fact."))
    S.append({"title": "Executive summary", "blocks": blocks})

    # 2 methodology
    meth = [f"{len(snap.get('search_results') or []) or len(topics)} sub-question searches (search chain: Gemini, then Tavily, then DuckDuckGo).",
            f"{ev.get('extracted_claims', 'n/a')} claims extracted; duplicates were merged, leaving {funnel['distinct']} distinct claims "
            f"shown as {funnel['rows']} rows (a merged claim appears once per sub-topic that raised it).",
            "A claim counts as CONFIRMED only with sources from at least two independent domains "
            "(social platforms excluded); a code-enforced gate downgrades any verdict that lacks them."]
    if funnel["verification_recorded"]:
        meth.append(f"Targeted verification: {funnel['initially_under_supported']} distinct claims started with fewer than two "
                    f"independent domains; {funnel['verification_tried']} were searched for a second source, "
                    f"{funnel['verification_upgraded']} reached two domains ({funnel['verification_sources_added']} sources added).")
    else:
        meth.append("Targeted verification search was not recorded for this run.")
    meth.append(f"{stats['llm_calls']} LLM calls, {stats['input_tokens']:,} input and {stats['output_tokens']:,} output tokens.")
    S.append({"title": "Research methodology", "blocks": [("ul", meth)]})

    # 3 findings by sub-topic
    summaries = {s.get("sub_question"): s for s in snap.get("summaries") or []}
    blocks = []
    for i, q in enumerate(topics, start=1):
        rows = [t for t in traced if t["topic"] == q]
        s = summaries.get(q, {})
        text = s.get("summary") if s.get("status") in ("ok", "no_facts") else "Summary unavailable: generation failed for this sub-topic."
        blocks += [("h3", f"Q{i}. {q}"), ("p", text or "No summary was produced for this sub-topic.")]
        conf = [t for t in rows if t["category"] == "confirmed"]
        if conf:
            blocks.append(("ul", [f"{t['claim']} " + " ".join(f"[{refs[x['url']]}]" for x in t["sources"] if x["url"] in refs)
                                  for t in conf]))
    S.append({"title": "Findings by sub-topic", "blocks": blocks})

    # 4 comparison table
    table = []
    for i, q in enumerate(topics, start=1):
        rows = [t for t in traced if t["topic"] == q]
        table.append([f"Q{i}", views.short_topic(q, 60), str(len(rows)),
                      str(sum(t["category"] == "confirmed" for t in rows)),
                      str(sum(t["category"] == "disputed" for t in rows)),
                      str(sum(t["status"] == "UNVERIFIED" for t in rows)),
                      str(len({x["url"] for t in rows for x in t["sources"]}))])
    S.append({"title": "Comparison of sub-topics",
              "blocks": [("table", ["#", "Sub-topic", "Claims", "Confirmed", "Disputed", "Unverified", "Sources"], table)]})

    # 5 evidence and citations
    src_rows = []
    seen = {}
    for t in traced:
        for s in t["sources"]:
            seen.setdefault(s["url"], s)
    for url, i in refs.items():
        s = seen[url]
        src_rows.append([str(i), s["title"][:90], s["domain"], "targeted verification" if s["origin"] == "verification" else "original search", url])
    conf_rows = [[t["claim"][:140], ", ".join(f"[{refs[x['url']]}]" for x in t["sources"] if x["url"] in refs),
                  str(t["independent_domains"])] for t in traced if t["category"] == "confirmed"]
    blocks = []
    if conf_rows:
        blocks += [("p", "Confirmed claims and the sources that back them:"), ("table", ["Claim", "Sources", "Independent domains"], conf_rows)]
    blocks += [("p", "Source list:"), ("table", ["Ref", "Title", "Domain", "Found by", "URL"], src_rows)] if src_rows else [("p", "No sources were recorded.")]
    S.append({"title": "Evidence and citations", "blocks": blocks})

    # 6 limitations
    lim = []
    if n:
        lim.append(f"{counts['UNVERIFIED']} of {n} claims ({counts['UNVERIFIED'] / n:.0%}) are unverified. Breakdown: "
                   f"too few independent sources {cats['insufficient_sources']}, inconclusive evidence {cats['inconclusive']}, "
                   f"failed research or verification step {cats['failed']}.")
    lim.append("Independent domains are not always independent origins: two sites can repeat the same upstream figure.")
    if any(t["shared_origin_warning"] for t in traced):
        lim.append(f"{sum(t['shared_origin_warning'] for t in traced)} claim(s) have near-identical evidence sentences on different domains and may share one origin.")
    if quality["claims_with_conflicts"]:
        lim.append(f"{quality['claims_with_conflicts']} claim(s) have retrieved sentences that state different figures or the opposite; these are rule-based flags to review.")
    lim.append(f"{quality['citation_support_rate']:.0%} of checkable citations have a sentence that states the claim's figures, subject and period.")
    lim.append("Contradictions are only detected between claims checked in the same batch, so zero disputed claims does not prove none exist.")
    lim += health["issues"]
    if validation and validation.get("issues"):
        vs = validation.get("stats") or {}
        lim.append(f"Automated validation of the generated text found {vs.get('errors', 0)} problem(s) and {vs.get('warnings', 0)} warning(s) "
                   "(unverifiable links removed; figures or confirm/dispute wording that disagree with the evidence are listed at the end of the narrative).")
    elif validation:
        lim.append("Automated validation found no unverifiable links, unsupported figures or overstated claims in the generated text. This is a rule-based check, not proof that every sentence is true.")
    lim.append(f"Evidence-sentence coverage is {quality['evidence_sentence_coverage']:.0%}: the share of claims with a stored sentence that states them. This is a proxy for citation correctness, not proof of it.")
    S.append({"title": "Limitations and research gaps", "blocks": [("ul", lim)]})

    # 7 conclusion
    concl = (f"{counts['CONFIRMED']} of {n} claims meet the two-independent-domain standard. " if n else "No claims were available. ")
    concl += ("The remaining findings are leads that need more primary-source corroboration." if n and counts["CONFIRMED"] < n else "")
    S.append({"title": "Conclusion", "blocks": [("p", concl.strip())]})

    # 8 narrative
    if snap.get("final_report"):
        S.append({"title": "Generated narrative report", "blocks": [("md", snap["final_report"])]})
    return {"title": "Research report", "info": info, "sections": S}


# --------------------------------------------------------------------------- markdown
def _md_table(headers, rows) -> str:
    cell = lambda v: str(v).replace("|", "\\|").replace("\n", " ")
    return "\n".join(["| " + " | ".join(headers) + " |", "|" + "---|" * len(headers)]
                     + ["| " + " | ".join(cell(c) for c in r) + " |" for r in rows])


def section_markdown(sec: Dict, level: int = 2) -> str:
    out = [f"{'#' * level} {sec['title']}", ""]
    for b in sec["blocks"]:
        kind = b[0]
        if kind == "p":
            out += [b[1], ""]
        elif kind == "h3":
            out += [f"{'#' * (level + 1)} {b[1]}", ""]
        elif kind == "ul":
            out += [f"- {x}" for x in b[1]] + [""]
        elif kind == "table":
            out += [_md_table(b[1], b[2]), ""]
        elif kind == "md":
            out += [re.sub(r"^(#{1,2}) ", r"##\1 ", b[1], flags=re.M), ""]   # demote narrative headings
    return "\n".join(out)


def to_markdown(doc: Dict) -> str:
    parts = [f"# {doc['title']}", "", _md_table(["Field", "Value"], doc["info"]), ""]
    parts += [section_markdown(sec) for sec in doc["sections"]]
    return "\n".join(parts).rstrip() + "\n"


# --------------------------------------------------------------------------- pdf
_REPLACE = {"\u2018": "'", "\u2019": "'", "\u201c": '"', "\u201d": '"', "\u2013": "-", "\u2014": "-", "\u2192": "->",
            "\u2265": ">=", "\u2264": "<=", "\u2022": "-", "\u2026": "...", "\u00a0": " ", "\u2082": "2", "\u2083": "3",
            "\u2705": "[ok]", "\u26a0": "[!]", "\u2753": "[?]", "\u274c": "[x]"}


def pdf_text(s: str) -> str:
    """The built-in PDF fonts are Latin-1 only: map common punctuation, drop what cannot be shown, then XML-escape."""
    s = re.sub("[️‍]", "", str(s))          # emoji variation selector / joiner would print as a stray "?"
    for k, v in _REPLACE.items():
        s = s.replace(k, v)
    return escape(s.encode("latin-1", "replace").decode("latin-1"))

def _inline(s: str) -> str:
    return re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", pdf_text(s)).replace("`", "")


def to_pdf(doc: Dict) -> bytes:
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    from reportlab.lib.units import mm
    from reportlab.platypus import ListFlowable, ListItem, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

    ss = getSampleStyleSheet()
    body = ParagraphStyle("b", parent=ss["BodyText"], fontSize=9.5, leading=13)
    cellst = ParagraphStyle("c", parent=body, fontSize=8, leading=10)
    h1 = ParagraphStyle("h1", parent=ss["Title"], fontSize=20, spaceAfter=8)
    h2 = ParagraphStyle("h2", parent=ss["Heading2"], spaceBefore=12, spaceAfter=4, textColor=colors.HexColor("#3730a3"))
    h3 = ParagraphStyle("h3", parent=ss["Heading3"], spaceBefore=8, spaceAfter=2)
    buf = io.BytesIO()
    page = SimpleDocTemplate(buf, pagesize=A4, leftMargin=18 * mm, rightMargin=18 * mm, topMargin=16 * mm,
                             bottomMargin=16 * mm, title=pdf_text(doc["title"]))
    width = A4[0] - 36 * mm

    def table(headers, rows, widths=None):
        data = [[Paragraph(f"<b>{pdf_text(h)}</b>", cellst) for h in headers]] + \
               [[Paragraph(pdf_text(c), cellst) for c in r] for r in rows]
        t = Table(data, colWidths=widths, repeatRows=1)
        t.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e0e7ff")),
                               ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#9ca3af")),
                               ("VALIGN", (0, 0), (-1, -1), "TOP")]))
        return t

    def bullets(items):
        return ListFlowable([ListItem(Paragraph(_inline(x), body)) for x in items], bulletType="bullet", leftIndent=12)

    story = [Paragraph(pdf_text(doc["title"]), h1), table(["Field", "Value"], doc["info"], [45 * mm, width - 45 * mm])]
    for sec in doc["sections"]:
        story.append(Paragraph(pdf_text(sec["title"]), h2))
        for b in sec["blocks"]:
            if b[0] == "p":
                story.append(Paragraph(_inline(b[1]), body))
            elif b[0] == "h3":
                story.append(Paragraph(pdf_text(b[1]), h3))
            elif b[0] == "ul" and b[1]:
                story.append(bullets(b[1]))
            elif b[0] == "table":
                n = len(b[1])
                widths = None
                if b[1][-1] == "URL":
                    widths = [10 * mm, 50 * mm, 28 * mm, 26 * mm, width - 114 * mm]
                elif n == 7:
                    widths = [9 * mm, width - 9 * mm - 5 * 21 * mm] + [21 * mm] * 5
                story.append(table(b[1], b[2], widths))
            elif b[0] == "md":
                for line in b[1].splitlines():
                    line = line.rstrip()
                    if not line or line.startswith("|") or set(line) <= set("-| "):
                        continue
                    m = re.match(r"^(#{1,6})\s+(.*)", line)
                    if m:
                        story.append(Paragraph(_inline(m.group(2)), h3))
                    elif re.match(r"^\s*([-*]|\d+\.)\s+", line):
                        story.append(Paragraph("&bull; " + _inline(re.sub(r"^\s*([-*]|\d+\.)\s+", "", line)), body))
                    else:
                        story.append(Paragraph(_inline(line), body))
            story.append(Spacer(1, 4))
    page.build(story)
    return buf.getvalue()


# --------------------------------------------------------------------------- structured exports
def claims_export(snap: Dict) -> Dict:
    traced = trace_all(snap.get("fact_check_results") or [], snap.get("search_results") or [], snap.get("query", ""))
    return {"query": snap.get("query", ""), "generated": snap.get("created_at", ""),
            "run_status": run_health(snap)["state"],
            "claims": [{"claim": t["claim"], "status": t["status"], "verdict_category": t["category"],
                        "reason": t["explanation"], "sub_topic": t["topic"],
                        "independent_domains": t["independent_domains"], "verified_by_search": t["verified_by_search"],
                        "verification_outcome": t["verification"], "possible_shared_origin": t["shared_origin_warning"],
                        "sources": [{k: s[k] for k in ("title", "url", "domain", "origin", "evidence", "match")}
                                    for s in t["sources"]]} for t in traced]}


def _safe_cell(v) -> str:
    """Spreadsheet formula-injection guard: web text must not be able to start a formula."""
    v = "" if v is None else str(v)
    return "'" + v if v[:1] in ("=", "+", "-", "@", "\t", "\r") else v


EVIDENCE_COLUMNS = ["claim", "status", "verdict_category", "sub_topic", "independent_domains", "source_title",
                    "source_url", "domain", "found_by", "evidence_sentence", "evidence_match"]


def evidence_csv(snap: Dict) -> str:
    out = io.StringIO()
    w = csv.writer(out)
    w.writerow(EVIDENCE_COLUMNS)
    for t in trace_all(snap.get("fact_check_results") or [], snap.get("search_results") or [], snap.get("query", "")):
        base = [t["claim"], t["status"], t["category"], t["topic"], t["independent_domains"]]
        for s in t["sources"] or [None]:
            tail = ["", "", "", "", "", ""] if s is None else [s["title"], s["url"], s["domain"], s["origin"], s["evidence"], s["match"]]
            w.writerow([_safe_cell(c) for c in base + tail])
    return out.getvalue()

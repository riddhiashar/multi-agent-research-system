"""Validation layer for the FINAL generated text (Day 28). Pure and offline by default; unit-tested.

The LLM writes the narrative report and the sub-topic summaries. Before they are shown or downloaded this layer
checks three things against the run's own evidence (claims, stored evidence sentences, retrieved snippets, URLs):

1. URLs    - well-formed, safe, and actually retrieved during this run (an unseen domain is likely invented).
2. Numbers - every figure (with its unit) and year in the text exists in the evidence, units and scale respected.
3. Claims  - a sentence must not contradict a checked claim, and must not call something "confirmed" when the
             matching claim is disputed or unverified.

It finds problems and removes unverifiable links; it does not rewrite prose. Rule-based: it cannot prove a
sentence is true, only that its URLs, figures and confirm/dispute wording agree with what the run found."""
import ipaddress
import re
from datetime import datetime
from typing import Callable, Dict, Iterable, List, Optional
from urllib.parse import urlparse

from tools import claim_verifier, guardrails
from tools.evidence import _registrable
from tools.evidence_check import snippet_records

URL_RE = re.compile(r"(?:https?://|\b(?:javascript|vbscript|data|file):)[^\s<>\"'\]\)]*", re.I)
_TRAIL = ".,;:!?*_`"
_HEDGE = re.compile(r"\b(disput\w*|conflict\w*|contradict\w*|however|whereas|while another|according to|reported|reportedly|"
                    r"claims?|claimed|unverified|unconfirmed|single source|not clear|unclear|may|might|could|suggest\w*)\b", re.I)
_CONFIRM = re.compile(r"\b(confirmed|verified|established|proven|corroborated)\b", re.I)
_NEGATED_CONFIRM = re.compile(r"\b(not|no|never|cannot|can't|without|lack\w*|un)\b\W+(?:\w+\W+){0,2}(confirmed|verified|established|proven|corroborated)\b|"
                              r"\b(unconfirmed|unverified)\b", re.I)
SKIP_PLAIN_INTEGERS_UP_TO = 12            # list numbers and "3 sub-topics" are not factual figures
LIVE_URL_LIMIT = 10


# -------------------------------------------------------------------------------------------- context
def build_context(snap: Dict) -> Dict:
    """Everything the text may legitimately rely on, taken from the saved run."""
    results = snap.get("fact_check_results") or []
    search = snap.get("search_results") or []
    records = snippet_records(search)
    urls = {s.get("url") for r in results for s in r.get("sources") or [] if s.get("url")}
    urls |= {s.get("url") for r in search if isinstance(r, dict) for s in r.get("sources") or [] if s.get("url")}
    urls.discard(None)
    evidence = [r.get("claim", "") for r in results]
    evidence += [s.get("evidence", "") for r in results for s in r.get("sources") or []]
    evidence += [x["text"] for x in records]
    evidence.append(snap.get("query", ""))                       # figures the user typed are legitimate
    q = claim_verifier.extract_quantities("\n".join(e for e in evidence if e))
    created = (snap.get("created_at") or "")[:4]
    years = set(q["years"]) | {datetime.now().year}
    if created.isdigit():
        years.add(int(created))
    n_rows = len(results)
    counts = {st: sum(1 for r in results if r.get("status") == st) for st in ("CONFIRMED", "DISPUTED", "UNVERIFIED")}
    meta = {n_rows, len({r.get("claim") for r in results}), len(urls), len(snap.get("summaries") or []), *counts.values(),
            (snap.get("evidence") or {}).get("extracted_claims", -1)}
    return {"claims": [{"claim": r.get("claim", ""), "status": r.get("status", "UNVERIFIED")} for r in results if r.get("claim")],
            "urls": urls, "norm_urls": {norm_url(u) for u in urls}, "domains": {_registrable(urlparse(u).hostname or "") for u in urls},
            "quantities": q["quantities"], "years": years, "meta_numbers": {float(m) for m in meta if m is not None and m >= 0},
            "entities": claim_verifier.query_entities(snap.get("query", ""))}


# -------------------------------------------------------------------------------------------- urls
def norm_url(url: str) -> str:
    p = urlparse(url.strip())
    host = (p.hostname or "").lower()
    host = host[4:] if host.startswith("www.") else host
    return host + ((p.path or "").rstrip("/"))


def _url_problem(url: str) -> Optional[str]:
    if re.match(r"(?i)(javascript|vbscript|data|file):", url):
        return "unsafe link scheme"
    p = urlparse(url)
    host = (p.hostname or "").lower()
    if p.scheme not in ("http", "https") or not host:
        return "malformed link"
    if "@" in (p.netloc or ""):
        return "link contains embedded credentials"
    if host in ("localhost",) or host.endswith((".local", ".internal")):
        return "link points to a local address"
    try:
        ip = ipaddress.ip_address(host)
        if ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_reserved:
            return "link points to a private network address"
    except ValueError:
        if "." not in host:
            return "link has no valid domain"
    return None


def extract_urls(text: str) -> List[str]:
    return [m.group(0).rstrip(_TRAIL) for m in URL_RE.finditer(text or "")]


def check_url(url: str, ctx: Dict) -> Optional[Dict]:
    bad = _url_problem(url)
    if bad:
        return {"type": "url_invalid", "severity": "error", "text": url[:160], "detail": bad}
    n = norm_url(url)
    if any(n == k or n.startswith(k + "/") or k.startswith(n + "/") for k in ctx["norm_urls"]):
        return None
    if _registrable(urlparse(url).hostname or "") in ctx["domains"]:
        return {"type": "url_not_retrieved", "severity": "warning", "text": url[:160],
                "detail": "the site was used in this run but this exact page was not retrieved"}
    return {"type": "url_unknown", "severity": "error", "text": url[:160],
            "detail": "this site never appeared in the run's sources, so the link may be invented"}


# -------------------------------------------------------------------------------------------- text helpers
def _plain(text: str) -> str:
    t = URL_RE.sub(" ", text or "")
    t = re.sub(r"\[\d+(?:[,\s]+\d+)*\]", " ", t)                 # citation markers [2] [3]
    t = re.sub(r"\[([^\]]*)\]\(\s*\)", r"\1", t)
    t = re.sub(r"[#*_`|>]+", " ", t)
    t = re.sub(r"^\s*\d+[.)]\s+", "", t, flags=re.M)              # list numbering
    t = re.sub(r"\bQ\d+\b[:.]?", " ", t)
    return t


def _sentences(text: str) -> List[str]:
    return [s for s in claim_verifier.split_sentences(_plain(text)) if len(s) > 12]


def _is_plain_small(q: Dict) -> bool:
    return not q["family"] and float(q["hi"]).is_integer() and q["hi"] <= SKIP_PLAIN_INTEGERS_UP_TO and q["lo"] == q["hi"]


# -------------------------------------------------------------------------------------------- numbers
def check_numbers(sentence: str, ctx: Dict) -> List[Dict]:
    out = []
    q = claim_verifier.extract_quantities(sentence)
    for item in q["quantities"]:
        if _is_plain_small(item) or float(item["lo"]) in ctx["meta_numbers"] and not item["family"]:
            continue
        res = claim_verifier.compare_quantities([item], ctx["quantities"])
        if res["status"] in ("match", "partial"):
            continue
        out.append({"type": "number_unsupported", "severity": "error" if item["family"] else "warning",
                    "text": sentence[:200], "detail": f"{item['raw']} does not appear in any claim or retrieved snippet"})
    for y in sorted(q["years"] - ctx["years"]):
        out.append({"type": "year_unsupported", "severity": "warning", "text": sentence[:200],
                    "detail": f"the year {y} does not appear in the evidence"})
    return out


# -------------------------------------------------------------------------------------------- claims
def _best_claim(sentence: str, ctx: Dict) -> Optional[Dict]:
    best = None
    for c in ctx["claims"]:
        a = claim_verifier.assess(c["claim"], sentence, entities=ctx["entities"])
        if a["relation"] != "unrelated" and (best is None or a["score"] > best["score"]):
            best = {**a, "claim": c["claim"], "status": c["status"]}
    return best


def check_claim_statement(sentence: str, ctx: Dict) -> List[Dict]:
    confirms = bool(_CONFIRM.search(sentence)) and not _NEGATED_CONFIRM.search(sentence)
    q = claim_verifier.extract_quantities(sentence)
    if not (confirms or q["quantities"] or q["years"]):
        return []
    best = _best_claim(sentence, ctx)
    if not best or best["score"] < 0.45:
        return []
    hedged = bool(_HEDGE.search(sentence))
    if best["relation"] == "contradicts" and not hedged:
        return [{"type": "claim_contradicts", "severity": "error", "text": sentence[:200],
                 "detail": f"conflicts with the checked claim \"{best['claim'][:110]}\" ({'; '.join(best['reasons'])[:120]})"}]
    if confirms and best["status"] != "CONFIRMED" and best["relation"] in ("supports", "uncertain") and not hedged:
        return [{"type": "claim_overstated", "severity": "error", "text": sentence[:200],
                 "detail": f"presented as confirmed, but the matching claim \"{best['claim'][:110]}\" is {best['status'].lower()}"}]
    return []


# -------------------------------------------------------------------------------------------- orchestration
def validate_text(where: str, text: str, ctx: Dict) -> Dict:
    issues, stats = [], {"urls": 0, "sentences": 0}
    urls = list(dict.fromkeys(extract_urls(text)))
    stats["urls"] = len(urls)
    for u in urls:
        i = check_url(u, ctx)
        if i:
            issues.append(i)
    for s in _sentences(text):
        stats["sentences"] += 1
        issues += check_numbers(s, ctx) + check_claim_statement(s, ctx)
    for i in issues:
        i["where"] = where
    return {"issues": issues, "stats": stats}


def _texts(snap: Dict, report: Optional[str] = None) -> List[tuple]:
    out = [("Narrative report", report if report is not None else (snap.get("final_report") or ""))]
    for i, s in enumerate(snap.get("summaries") or [], start=1):
        if s.get("status") in ("ok", "no_facts") and s.get("summary"):
            out.append((f"Sub-topic Q{i} summary", s["summary"]))
    return out


def validate_snapshot(snap: Dict, report: Optional[str] = None, live_fetch: Optional[Callable] = None) -> Dict:
    ctx = build_context(snap)
    issues: List[Dict] = []
    stats = {"urls": 0, "sentences": 0}
    seen_urls: List[str] = []
    for where, text in _texts(snap, report):
        r = validate_text(where, text, ctx)
        issues += r["issues"]
        stats["urls"] += r["stats"]["urls"]
        stats["sentences"] += r["stats"]["sentences"]
        seen_urls += [u for u in extract_urls(text) if not _url_problem(u)]
    if live_fetch:
        for u in list(dict.fromkeys(seen_urls))[:LIVE_URL_LIMIT]:
            code = live_fetch(u)
            if code is None or code >= 400 and code not in (401, 403, 429):
                issues.append({"type": "url_unreachable", "severity": "warning", "where": "Links", "text": u[:160],
                               "detail": "no response" if code is None else f"HTTP {code}"})
        stats["live_checked"] = min(len(set(seen_urls)), LIVE_URL_LIMIT)
    errors = sum(1 for i in issues if i["severity"] == "error")
    stats.update({"errors": errors, "warnings": len(issues) - errors, "claims_known": len(ctx["claims"])})
    return {"ok": errors == 0, "issues": issues, "stats": stats, "checked_at": datetime.now().isoformat(timespec="seconds")}


def summary_line(v: Dict) -> str:
    s = v.get("stats") or {}
    if not v.get("issues"):
        return "Passed"
    return f"{s.get('errors', 0)} problem(s), {s.get('warnings', 0)} warning(s)"


# -------------------------------------------------------------------------------------------- repair
def remove_bad_links(text: str, issues: Iterable[Dict]) -> str:
    """Unverifiable links are removed from the text (the words stay). Warnings keep their link."""
    bad = {i["text"] for i in issues if i["type"] in ("url_unknown", "url_invalid")}
    for u in sorted(bad, key=len, reverse=True):
        text = re.sub(r"\[([^\]]+)\]\(\s*" + re.escape(u) + r"[^)]*\)", r"\1 (link removed: not a source from this run)", text)
        text = text.replace(u, "[link removed]")
    return text


def _footer(issues: List[Dict]) -> str:
    if not issues:
        return ""
    lines = ["", "---", "**Automated validation notes** (rule-based checks of the text above against this run's evidence):"]
    for i in issues[:12]:
        lines.append(f"- {i['type'].replace('_', ' ')}: {i['detail']} - \"{i['text'][:110]}\"")
    if len(issues) > 12:
        lines.append(f"- ...and {len(issues) - 12} more")
    return "\n".join(lines) + "\n"


def finalize_snapshot(snap: Dict, live_fetch: Optional[Callable] = None) -> Dict:
    """Sanitise, validate and repair the generated text; returns a new snapshot. Raw text is kept in final_report_raw
    (and summary_raw). Call again after the report is regenerated, after removing final_report_raw."""
    snap = dict(snap)
    raw = snap.get("final_report_raw") or snap.get("final_report") or ""
    san = guardrails.sanitize_output(raw)
    actions = list(san["actions"])
    summaries = []
    for s in snap.get("summaries") or []:
        s = dict(s)
        if s.get("summary"):
            cleaned = guardrails.sanitize_output(s.get("summary_raw") or s["summary"])
            s["summary_raw"] = s.get("summary_raw") or s["summary"]
            s["summary"] = cleaned["text"]
            actions += cleaned["actions"]
        summaries.append(s)
    snap["summaries"] = summaries
    v = validate_snapshot({**snap, "final_report": san["text"]}, report=san["text"], live_fetch=live_fetch)
    all_issues = v["issues"]
    snap["summaries"] = [{**s, "summary": remove_bad_links(s["summary"], [i for i in all_issues if i["where"].startswith("Sub-topic")])}
                         if s.get("summary") else s for s in summaries]
    narrative = [i for i in all_issues if i["where"] == "Narrative report"]
    body = remove_bad_links(san["text"], narrative)
    snap["final_report_raw"] = raw
    snap["final_report"] = (body.rstrip() + "\n" + _footer([i for i in narrative if i["severity"] == "error" or i["type"].startswith("url")])) if raw else raw
    merged: Dict[str, int] = {}
    for a in actions:
        merged[a["type"]] = merged.get(a["type"], 0) + a["count"]
    v["guardrail_actions"] = [{"type": k, "count": n} for k, n in merged.items()]
    snap["validation"] = v
    return snap


# -------------------------------------------------------------------------------------------- optional live check
def http_head(url: str, timeout: float = 5.0) -> Optional[int]:
    """HTTP status of a public URL, or None on network failure. Only used when VALIDATE_URLS_LIVE=true."""
    import urllib.error
    import urllib.request
    if _url_problem(url):
        return None
    for method in ("HEAD", "GET"):
        try:
            req = urllib.request.Request(url, method=method, headers={"User-Agent": "research-system-link-check"})
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return r.status
        except urllib.error.HTTPError as e:
            if method == "HEAD" and e.code in (400, 403, 405):
                continue
            return e.code
        except Exception:
            return None
    return None

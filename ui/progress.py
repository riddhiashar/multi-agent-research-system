"""Progress tracking for the UI (Day 26). Pure Python (no Streamlit import) so it is unit-tested.
Turns LangGraph 'updates' chunks into readable log lines and a 5-stage pipeline status."""
import html
import time
from collections import Counter
from typing import Callable, Dict, List, Optional, Tuple

STAGE_CARDS = [
    ("web_search_single", "Search", "🔎"),
    ("data_extraction", "Extract", "🧾"),
    ("fact_checking", "Verify", "✅"),
    ("summarization", "Summarize", "📝"),
    ("report_generator", "Report", "📄"),
]
STAGES = [node for node, _, _ in STAGE_CARDS]
STAGE_LABELS = {
    "supervisor": "Supervisor",
    "web_search_single": "Web search",
    "data_extraction": "Data extraction",
    "fact_checking": "Fact-checking",
    "summarization": "Summarization",
    "report_generator": "Report generator",
}


def clip(text: str, n: int) -> str:
    """Shorten at a word boundary with an ellipsis."""
    text = " ".join((text or "").split())
    if len(text) <= n:
        return text
    cut = text[:n].rsplit(" ", 1)[0] or text[:n]
    return cut.rstrip(",;:- ") + "…"


def stage_progress(node: str) -> Optional[float]:
    """0..1 progress once `node` has reported (None for nodes outside the pipeline)."""
    if node not in STAGES:
        return None
    return (STAGES.index(node) + 1) / len(STAGES)


def describe_update(node: str, update) -> str:
    """One short line describing what a node just did."""
    update = update or {}
    errors = update.get("errors") or []
    suffix = f"  ⚠ {len(errors)} error(s)" if errors else ""

    if node == "supervisor":
        plan = update.get("plan")
        if plan:
            return f"Planned {len(plan)} sub-questions" + suffix
        nxt = update.get("next_agent")
        return (f"Next step: {STAGE_LABELS.get(nxt, nxt)}" if nxt else "Supervisor decided the next step") + suffix

    if node == "web_search_single":
        lines = []
        for r in update.get("search_results") or []:
            q = clip(r.get("sub_question") or "", 60)
            if r.get("status") == "failed" or not r.get("sources"):
                lines.append(f"Search found nothing for “{q}”")
            else:
                lines.append(f"Searched “{q}” via {r.get('provider', '?')} ({len(r['sources'])} sources)")
        return "; ".join(lines) + suffix if lines else "Search finished" + suffix

    if node == "data_extraction":
        return f"Extracted {len(update.get('extracted_facts') or [])} claims" + suffix

    if node == "fact_checking":
        results = update.get("fact_check_results") or []
        c = Counter(r.get("status", "UNVERIFIED") for r in results)
        return (f"Checked {len(results)} claims: {c.get('CONFIRMED', 0)} confirmed, "
                f"{c.get('DISPUTED', 0)} disputed, {c.get('UNVERIFIED', 0)} unverified") + suffix

    if node == "summarization":
        return f"Summarised {len(update.get('summaries') or [])} sub-topic(s)" + suffix

    if node == "report_generator":
        report = update.get("final_report") or ""
        return f"Report written ({len(report)} characters)" + suffix

    return f"{node} finished" + suffix


def iter_updates(chunk) -> List[Tuple[str, Dict]]:
    """Normalises one 'updates' chunk -> [(node, update)], tolerating None / list updates."""
    out = []
    if not isinstance(chunk, dict):
        return out
    for node, update in chunk.items():
        if isinstance(update, list):
            out.extend((node, u) for u in update)
        else:
            out.append((node, update))
    return out


class PipelineTracker:
    """Tracks which of the 5 pipeline stages is pending / running / done, with timings."""

    def __init__(self, clock: Callable[[], float] = time.time):
        self._clock = clock
        self._t0: Optional[float] = None
        self.stages = {n: {"status": "pending", "detail": "", "seconds": None, "started": None}
                       for n in STAGES}
        self.expected_searches: Optional[int] = None
        self._searches_done = 0
        self.events: List[Dict] = []

    def start(self) -> None:
        self._t0 = self._clock()
        self._begin(STAGES[0])

    def _begin(self, node: str) -> None:
        s = self.stages[node]
        if s["status"] == "pending":
            s["status"], s["started"] = "running", self._clock()

    def _finish(self, node: str, detail: str) -> None:
        s = self.stages[node]
        if s["status"] == "done":
            return
        if s["started"] is None:
            s["started"] = self._clock()
        s["status"], s["detail"] = "done", detail
        s["seconds"] = self._clock() - s["started"]
        idx = STAGES.index(node)
        if idx + 1 < len(STAGES):
            self._begin(STAGES[idx + 1])

    def on_update(self, node: str, update) -> str:
        update = update or {}
        text = describe_update(node, update)
        elapsed = (self._clock() - self._t0) if self._t0 is not None else 0.0
        self.events.append({"t": round(elapsed, 1), "node": node,
                            "label": STAGE_LABELS.get(node, node), "text": text})
        if node == "supervisor" and update.get("plan"):
            self.expected_searches = len(update["plan"])
        if node == "web_search_single":
            self._searches_done += 1
            target = self.expected_searches or 1
            if self._searches_done >= target:
                self._finish(node, f"{self._searches_done} queries")
            else:
                self.stages[node]["detail"] = f"{self._searches_done}/{target} queries"
        elif node in self.stages:
            self._finish(node, text)
        return text

    def fraction_done(self) -> float:
        return sum(1 for s in self.stages.values() if s["status"] == "done") / len(STAGES)

    def snapshot(self) -> List[Dict]:
        return [{"node": n, "label": label, "icon": icon, **self.stages[n]}
                for n, label, icon in STAGE_CARDS]


_PIPELINE_CSS = """<style>
.pl-wrap{display:flex;gap:10px;flex-wrap:wrap;margin:6px 0 14px 0}
.pl-card{flex:1;min-width:140px;border:1px solid rgba(128,128,128,.35);border-radius:12px;padding:12px 14px}
.pl-title{font-weight:600;font-size:1rem}
.pl-status{font-size:.82rem;margin-top:2px}
.pl-detail{font-size:.75rem;opacity:.7;margin-top:4px;line-height:1.25}
.pl-pending{opacity:.5}
.pl-running{border-color:#3b82f6;box-shadow:0 0 0 2px rgba(59,130,246,.25);animation:pl-pulse 1.4s infinite}
.pl-done{border-color:#22c55e;background:rgba(34,197,94,.08)}
@keyframes pl-pulse{0%{box-shadow:0 0 0 0 rgba(59,130,246,.45)}70%{box-shadow:0 0 0 8px rgba(59,130,246,0)}100%{box-shadow:0 0 0 0 rgba(59,130,246,0)}}
</style>"""


def render_pipeline_html(stages: List[Dict]) -> str:
    """HTML for the stage cards (use with st.markdown(..., unsafe_allow_html=True))."""
    cards = []
    for s in stages:
        status = s["status"]
        if status == "done":
            line = f"Done · {s['seconds']:.1f}s" if s.get("seconds") is not None else "Done"
        elif status == "running":
            line = "Running…"
        else:
            line = "Waiting"
        detail = html.escape(clip(s.get("detail") or "", 70))
        cards.append(
            f'<div class="pl-card pl-{status}"><div class="pl-title">{s["icon"]} {html.escape(s["label"])}</div>'
            f'<div class="pl-status">{line}</div><div class="pl-detail">{detail}</div></div>')
    return _PIPELINE_CSS + '<div class="pl-wrap">' + "".join(cards) + "</div>"

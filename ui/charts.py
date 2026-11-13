"""Chart builders (Day 27). Horizontal bars, full-width readable labels, exact numbers drawn on the bars and in
tooltips. Altair ships with Streamlit; if it is missing or a spec fails, callers fall back to st.bar_chart,
so a chart problem can never break the page."""
from typing import Dict, List

VERDICT_COLORS = {"Confirmed": "#16a34a", "Disputed": "#dc2626", "Unverified": "#d97706"}


def verdict_rows(counts: Dict[str, int]) -> List[Dict]:
    """Always three rows, zero included: a hidden zero would look like missing data."""
    return [{"Verdict": "Confirmed", "Claims": counts.get("CONFIRMED", 0)},
            {"Verdict": "Disputed", "Claims": counts.get("DISPUTED", 0)},
            {"Verdict": "Unverified", "Claims": counts.get("UNVERIFIED", 0)}]


def topic_rows(by_topic: List[Dict]) -> List[Dict]:
    """views.status_by_topic -> long format for a stacked horizontal bar."""
    out = []
    for r in by_topic:
        for status, label in (("CONFIRMED", "Confirmed"), ("DISPUTED", "Disputed"), ("UNVERIFIED", "Unverified")):
            out.append({"Sub-topic": r["topic"], "Verdict": label, "Claims": r.get(status, 0)})
    return out


def _alt():
    import altair as alt
    return alt

def _color(alt, legend=True):
    return alt.Color("Verdict:N", scale=alt.Scale(domain=list(VERDICT_COLORS), range=list(VERDICT_COLORS.values())),
                     legend=alt.Legend(orient="bottom", title=None) if legend else None)


def verdict_chart(counts: Dict[str, int]):
    import pandas as pd
    alt = _alt()
    df = pd.DataFrame(verdict_rows(counts))
    base = alt.Chart(df).encode(y=alt.Y("Verdict:N", sort=["Confirmed", "Disputed", "Unverified"], title=None,
                                        axis=alt.Axis(labelLimit=200)),
                                x=alt.X("Claims:Q", title="Claims", axis=alt.Axis(tickMinStep=1)))
    bars = base.mark_bar().encode(color=_color(alt, legend=False), tooltip=["Verdict", "Claims"])
    labels = base.mark_text(align="left", dx=4).encode(text="Claims:Q")
    return (bars + labels).properties(height=alt.Step(38))


def topic_chart(by_topic: List[Dict]):
    import pandas as pd
    alt = _alt()
    df = pd.DataFrame(topic_rows(by_topic))
    order = [r["topic"] for r in by_topic]
    bars = alt.Chart(df).mark_bar().encode(
        y=alt.Y("Sub-topic:N", sort=order, title=None, axis=alt.Axis(labelLimit=420)),
        x=alt.X("Claims:Q", title="Claims", axis=alt.Axis(tickMinStep=1)),
        color=_color(alt), tooltip=["Sub-topic", "Verdict", "Claims"])
    return bars.properties(height=alt.Step(34))


def metric_bar(rows: List[Dict], label: str, value: str, title: str):
    """Generic horizontal bar with the number drawn at the end of each bar."""
    import pandas as pd
    alt = _alt()
    df = pd.DataFrame(rows)
    base = alt.Chart(df).encode(y=alt.Y(f"{label}:N", sort="-x", title=None, axis=alt.Axis(labelLimit=260)),
                                x=alt.X(f"{value}:Q", title=title))
    return (base.mark_bar(color="#6366f1").encode(tooltip=[label, value])
            + base.mark_text(align="left", dx=4).encode(text=f"{value}:Q")).properties(height=alt.Step(30))

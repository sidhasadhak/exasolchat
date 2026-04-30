"""Auto-charting for ExasolChat query results.

LLM suggests a chart config; this module renders it with Plotly or Altair.
Returns ("plotly", fig), ("altair", chart), or None.
"""

from __future__ import annotations

from typing import Optional

import pandas as pd


def auto_chart(
    df: pd.DataFrame,
    chart_config: Optional[dict],
    library: str = "auto",
) -> Optional[tuple[str, object]]:
    """Render a chart from a DataFrame + LLM chart config.

    Returns ("plotly", fig) | ("altair", chart) | None.
    library: "plotly" | "altair" | "auto" (tries plotly first).
    """
    if not chart_config or df is None or df.empty:
        return None

    chart_type = chart_config.get("chart_type", "table_only")
    if chart_type == "table_only":
        return None

    x = chart_config.get("x")
    y = chart_config.get("y")
    color = chart_config.get("color")
    title = chart_config.get("title", "")

    # Validate columns exist
    cols = set(df.columns)
    if x and x not in cols:
        x = df.columns[0] if len(df.columns) > 0 else None
    if isinstance(y, list):
        y = [c for c in y if c in cols] or None
        if y and len(y) == 1:
            y = y[0]
    elif y and y not in cols:
        numeric_cols = df.select_dtypes("number").columns
        y = numeric_cols[0] if len(numeric_cols) > 0 else None
    if color and color not in cols:
        color = None

    if not x or not y:
        return None

    use_lib = _pick_library(library)

    try:
        if use_lib == "plotly":
            return ("plotly", _plotly_chart(df, chart_type, x, y, color, title))
        else:
            return ("altair", _altair_chart(df, chart_type, x, y, color, title))
    except Exception:
        # If chosen library fails, try the other one
        try:
            if use_lib == "plotly":
                return ("altair", _altair_chart(df, chart_type, x, y, color, title))
            else:
                return ("plotly", _plotly_chart(df, chart_type, x, y, color, title))
        except Exception:
            return None


def _pick_library(preference: str) -> str:
    if preference in ("plotly", "altair"):
        return preference
    # "auto": prefer plotly
    try:
        import plotly  # noqa: F401
        return "plotly"
    except ImportError:
        return "altair"


def _plotly_chart(
    df: pd.DataFrame,
    chart_type: str,
    x: str,
    y,
    color: Optional[str],
    title: str,
):
    import plotly.express as px

    kwargs = dict(x=x, color=color, title=title)

    if chart_type == "bar":
        return px.bar(df, y=y, **kwargs)
    if chart_type == "line":
        return px.line(df, y=y, **kwargs)
    if chart_type == "area":
        return px.area(df, y=y, **kwargs)
    if chart_type == "scatter":
        return px.scatter(df, y=y, **kwargs)
    if chart_type == "pie":
        return px.pie(df, names=x, values=y if isinstance(y, str) else y[0], title=title)
    if chart_type == "heatmap":
        y_col = y if isinstance(y, str) else y[0]
        pivot = df.pivot_table(index=x, columns=color or x, values=y_col, aggfunc="sum") if color else df.set_index(x)
        import plotly.graph_objects as go
        return go.Figure(go.Heatmap(z=pivot.values, x=list(pivot.columns), y=list(pivot.index)))

    # Fallback: bar
    return px.bar(df, y=y, **kwargs)


def _altair_chart(
    df: pd.DataFrame,
    chart_type: str,
    x: str,
    y,
    color: Optional[str],
    title: str,
):
    import altair as alt

    y_col = y if isinstance(y, str) else y[0]
    enc = dict(
        x=alt.X(x, type="ordinal" if df[x].dtype == object else "quantitative"),
        y=alt.Y(y_col, type="quantitative"),
    )
    if color:
        enc["color"] = alt.Color(color)

    base = alt.Chart(df).properties(title=title)

    if chart_type == "bar":
        return base.mark_bar().encode(**enc)
    if chart_type == "line":
        return base.mark_line().encode(**enc)
    if chart_type == "area":
        return base.mark_area().encode(**enc)
    if chart_type == "scatter":
        enc["x"] = alt.X(x, type="quantitative")
        return base.mark_point().encode(**enc)
    if chart_type == "pie":
        return (
            base.mark_arc()
            .encode(theta=alt.Theta(y_col, type="quantitative"), color=alt.Color(x))
        )

    # Fallback: bar
    return base.mark_bar().encode(**enc)

"""Streamlit app for talonsight."""

from __future__ import annotations

# ── sys.path fix ──────────────────────────────────────────────────────────────
# Streamlit inserts the script's own directory (site-packages/talonsight/) at
# sys.path[0], so "import talonsight" resolves to talonsight/talonsight/ — not
# found.  Fix: ensure site-packages/ (the parent) is on sys.path first.
import sys as _sys, pathlib as _pl
_here = _pl.Path(__file__).resolve().parent   # …/site-packages/talonsight
_site = str(_here.parent)                      # …/site-packages
if _site not in _sys.path:
    _sys.path.insert(0, _site)
del _sys, _pl, _here, _site
# ─────────────────────────────────────────────────────────────────────────────

import os
import sys
import platform
import time
import streamlit as st
import pandas as pd
from dotenv import load_dotenv

from talonsight.app_builder import render_builder, render_metrics_tab
from talonsight.core import TalonSight, QueryResult
from talonsight.connection import ConnectionConfig
from talonsight.llm import OllamaBackend, OpenAICompatibleBackend, MLXBackend
from talonsight.setup_wizard import load_config as _load_wizard_config
from talonsight.safety import RiskLevel

load_dotenv()

_DEFAULT_DUCKDB_PATH    = os.environ.get("EXACHAT_DUCKDB_PATH", "")
_DEFAULT_OLLAMA_URL     = os.environ.get("EXACHAT_OLLAMA_URL", "http://localhost:11434")
_DEFAULT_OLLAMA_MODEL   = os.environ.get("EXACHAT_OLLAMA_MODEL", "qwen2.5-coder:7b")
_DEFAULT_KB_PATH        = os.environ.get("EXACHAT_KB_PATH", "")
_DEFAULT_METRICS_PATH   = os.environ.get("EXACHAT_METRICS_PATH", "")


# ── Page config ──────────────────────────────────────────────────────
st.set_page_config(
    page_title="⚡ talonsight",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom styles — Claude-inspired palette ───────────────────────────
st.markdown("""
<style>
    /* ── Cloudscape Design System inspired theme ────────────────────────── */
    @import url('https://fonts.googleapis.com/css2?family=Open+Sans:wght@300;400;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

    /* Design tokens */
    :root {
        --cs-bg-page:        #161d26;
        --cs-bg-container:   #1b232d;
        --cs-bg-sidebar:     #0f141a;
        --cs-bg-hover:       #333843;
        --cs-bg-input:       #232b37;
        --cs-border:         #424650;
        --cs-border-subtle:  #232b37;
        --cs-text-primary:   #ebebf0;
        --cs-text-secondary: #a4a4ad;
        --cs-text-muted:     #656871;
        --cs-blue:           #42b4ff;
        --cs-blue-hover:     #75cfff;
        --cs-green:          #29ad7f;
        --cs-yellow:         #f0a800;
        --cs-red:            #d13212;
        --cs-radius-sm:      4px;
        --cs-radius:         8px;
        --cs-radius-lg:      12px;
    }

    /* ── Base ────────────────────────────────────────────────────────────── */
    html, body, [class*="css"] {
        font-family: 'Open Sans', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
        font-size: 14px;
        line-height: 20px;
    }
    .stApp { background-color: var(--cs-bg-page) !important; }
    .block-container { padding-top: 1.25rem; max-width: 1100px; }
    code, pre, .stCode, [data-testid="stCode"] {
        font-family: 'JetBrains Mono', 'Monaco', 'Menlo', 'Consolas', monospace !important;
        font-size: 12px !important;
    }

    /* ── App header ──────────────────────────────────────────────────────── */
    header[data-testid="stHeader"], .stAppHeader {
        height: 2.75rem !important;
        min-height: 2.75rem !important;
        background-color: var(--cs-bg-sidebar) !important;
        border-bottom: 1px solid var(--cs-border) !important;
    }

    /* ── Sidebar ─────────────────────────────────────────────────────────── */
    [data-testid="stSidebar"] {
        background-color: var(--cs-bg-sidebar) !important;
        border-right: 1px solid var(--cs-border) !important;
    }
    [data-testid="stSidebar"] * { color: var(--cs-text-primary) !important; }
    [data-testid="stSidebar"] .stSelectbox label,
    [data-testid="stSidebar"] .stTextInput label,
    [data-testid="stSidebar"] .stTextArea label,
    [data-testid="stSidebar"] .stNumberInput label,
    [data-testid="stSidebar"] .stMultiSelect label {
        color: var(--cs-text-secondary) !important;
        font-size: 12px !important;
        font-weight: 600 !important;
        text-transform: uppercase !important;
        letter-spacing: 0.05em !important;
    }
    [data-testid="stSidebar"] hr {
        border-color: var(--cs-border) !important;
        margin: 12px 0 !important;
    }

    /* ── Form inputs ─────────────────────────────────────────────────────── */
    .stTextInput > div > div > input,
    .stTextArea > div > div > textarea,
    .stNumberInput > div > div > input {
        background-color: var(--cs-bg-input) !important;
        border: 1px solid var(--cs-border) !important;
        border-radius: var(--cs-radius-sm) !important;
        color: var(--cs-text-primary) !important;
        font-size: 14px !important;
        transition: border-color 0.15s ease !important;
    }
    .stTextInput > div > div > input:focus,
    .stTextArea > div > div > textarea:focus {
        border-color: var(--cs-blue) !important;
        box-shadow: 0 0 0 2px rgba(66,180,255,0.2) !important;
    }
    .stSelectbox > div > div,
    .stMultiSelect > div > div {
        background-color: var(--cs-bg-input) !important;
        border: 1px solid var(--cs-border) !important;
        border-radius: var(--cs-radius-sm) !important;
    }

    /* ── Buttons ─────────────────────────────────────────────────────────── */
    .stButton > button[kind="primary"],
    .stFormSubmitButton > button {
        background-color: var(--cs-blue) !important;
        color: #0f141a !important;
        border: none !important;
        border-radius: var(--cs-radius-sm) !important;
        font-weight: 700 !important;
        font-size: 14px !important;
        letter-spacing: 0.01em !important;
        padding: 6px 20px !important;
        transition: background-color 0.15s ease, box-shadow 0.15s ease !important;
    }
    .stButton > button[kind="primary"]:hover,
    .stFormSubmitButton > button:hover {
        background-color: var(--cs-blue-hover) !important;
        box-shadow: 0 0 0 3px rgba(66,180,255,0.25) !important;
    }
    .stButton > button[kind="secondary"],
    .stDownloadButton > button {
        background-color: transparent !important;
        color: var(--cs-text-primary) !important;
        border: 1px solid var(--cs-border) !important;
        border-radius: var(--cs-radius-sm) !important;
        font-weight: 600 !important;
        font-size: 14px !important;
        padding: 5px 20px !important;
        transition: background-color 0.15s ease, border-color 0.15s ease !important;
    }
    .stButton > button[kind="secondary"]:hover,
    .stDownloadButton > button:hover {
        background-color: var(--cs-bg-hover) !important;
        border-color: var(--cs-blue) !important;
    }

    /* ── Chat messages ───────────────────────────────────────────────────── */
    [data-testid="stChatMessage"] {
        background-color: var(--cs-bg-container) !important;
        border: 1px solid var(--cs-border-subtle) !important;
        border-radius: var(--cs-radius) !important;
        padding: 12px 16px !important;
        margin-bottom: 8px !important;
        box-shadow: none !important;
    }
    [data-testid="stChatInputContainer"],
    [data-testid="stBottom"] {
        background-color: var(--cs-bg-page) !important;
        border-top: 1px solid var(--cs-border) !important;
        padding-top: 8px !important;
    }
    [data-testid="stChatInputContainer"] textarea,
    [data-testid="stChatInput"] textarea {
        background-color: var(--cs-bg-input) !important;
        border: 1px solid var(--cs-border) !important;
        border-radius: var(--cs-radius) !important;
        color: var(--cs-text-primary) !important;
    }
    [data-testid="stChatInputContainer"] textarea:focus,
    [data-testid="stChatInput"] textarea:focus {
        border-color: var(--cs-blue) !important;
        box-shadow: 0 0 0 2px rgba(66,180,255,0.2) !important;
    }

    /* ── Expanders ───────────────────────────────────────────────────────── */
    [data-testid="stExpander"] {
        background-color: var(--cs-bg-container) !important;
        border: 1px solid var(--cs-border) !important;
        border-radius: var(--cs-radius) !important;
        margin-bottom: 8px !important;
    }
    [data-testid="stExpander"] summary {
        padding: 10px 16px !important;
        font-weight: 600 !important;
        font-size: 13px !important;
        color: var(--cs-text-secondary) !important;
        border-radius: var(--cs-radius) !important;
    }
    [data-testid="stExpander"] summary:hover {
        color: var(--cs-text-primary) !important;
        background-color: var(--cs-bg-hover) !important;
    }

    /* ── Tabs ────────────────────────────────────────────────────────────── */
    button[data-testid="stTab"] p {
        color: var(--cs-text-secondary) !important;
        font-size: 14px !important;
        font-weight: 600 !important;
        margin: 0 !important;
        letter-spacing: 0.01em !important;
    }
    button[data-testid="stTab"]:hover p { color: var(--cs-text-primary) !important; }
    button[data-testid="stTab"][aria-selected="true"] p {
        color: var(--cs-blue) !important;
        font-weight: 700 !important;
    }
    [data-baseweb="tab-highlight"] {
        background-color: var(--cs-blue) !important;
        height: 2px !important;
    }
    [data-baseweb="tab-border"] { background-color: var(--cs-border) !important; }

    /* Sticky tab bar */
    [data-testid="stTabs"] > div:first-child,
    div[data-baseweb="tabs"] > div:first-child,
    div[role="tablist"] {
        position: -webkit-sticky !important;
        position: sticky !important;
        top: 2.75rem !important;
        z-index: 100 !important;
        background-color: var(--cs-bg-page) !important;
        padding-bottom: 2px !important;
    }

    /* ── Dataframe / table ───────────────────────────────────────────────── */
    .stDataFrame, [data-testid="stDataFrame"] {
        border: 1px solid var(--cs-border) !important;
        border-radius: var(--cs-radius) !important;
        overflow: hidden !important;
    }
    [data-testid="stDataFrame"] thead th {
        background-color: var(--cs-bg-input) !important;
        color: var(--cs-text-secondary) !important;
        font-size: 12px !important;
        font-weight: 700 !important;
        text-transform: uppercase !important;
        letter-spacing: 0.05em !important;
        border-bottom: 1px solid var(--cs-border) !important;
    }

    /* ── Metrics ─────────────────────────────────────────────────────────── */
    [data-testid="stMetric"] {
        background-color: var(--cs-bg-container) !important;
        border: 1px solid var(--cs-border) !important;
        border-radius: var(--cs-radius) !important;
        padding: 16px 20px !important;
    }
    [data-testid="stMetricValue"] {
        color: var(--cs-text-primary) !important;
        font-size: 28px !important;
        font-weight: 300 !important;
    }
    [data-testid="stMetricLabel"] {
        color: var(--cs-text-secondary) !important;
        font-size: 12px !important;
        font-weight: 700 !important;
        text-transform: uppercase !important;
        letter-spacing: 0.05em !important;
    }

    /* ── Alerts ──────────────────────────────────────────────────────────── */
    .stAlert { border-radius: var(--cs-radius) !important; }
    [data-testid="stAlert"] {
        background-color: rgba(66,180,255,0.08) !important;
        border-color: rgba(66,180,255,0.3) !important;
    }

    /* ── App-specific components ─────────────────────────────────────────── */
    .badge-safe {
        display: inline-flex; align-items: center; gap: 4px;
        padding: 2px 10px; border-radius: 12px;
        background: rgba(41,173,127,0.12); color: #29ad7f;
        border: 1px solid rgba(41,173,127,0.3);
        font-size: 12px; font-weight: 600; letter-spacing: 0.02em;
    }
    .badge-warn {
        display: inline-flex; align-items: center; gap: 4px;
        padding: 2px 10px; border-radius: 12px;
        background: rgba(240,168,0,0.12); color: #f0a800;
        border: 1px solid rgba(240,168,0,0.3);
        font-size: 12px; font-weight: 600; letter-spacing: 0.02em;
    }
    .badge-blocked {
        display: inline-flex; align-items: center; gap: 4px;
        padding: 2px 10px; border-radius: 12px;
        background: rgba(209,50,18,0.12); color: #d13212;
        border: 1px solid rgba(209,50,18,0.3);
        font-size: 12px; font-weight: 600; letter-spacing: 0.02em;
    }
    .kb-indicator { font-size: 12px; color: var(--cs-text-muted); margin-top: 6px; }
    .col-warning {
        background: rgba(240,168,0,0.08);
        border: 1px solid rgba(240,168,0,0.25);
        border-left: 3px solid #f0a800;
        padding: 8px 12px; border-radius: var(--cs-radius-sm);
        font-size: 13px; color: #f0a800; margin-bottom: 8px;
    }
    .schema-table-name { font-weight: 700; font-size: 13px; color: var(--cs-blue); }
    .schema-row-count  { font-size: 12px; color: var(--cs-text-muted); }

    /* Follow-up suggestion pills */
    div[data-testid="stHorizontalBlock"] button[kind="secondary"] {
        background: rgba(66,180,255,0.08) !important;
        border: 1px solid rgba(66,180,255,0.3) !important;
        color: var(--cs-blue) !important;
        border-radius: 16px !important;
        font-size: 13px !important;
        padding: 4px 16px !important;
        font-weight: 400 !important;
        transition: background 0.15s ease !important;
    }
    div[data-testid="stHorizontalBlock"] button[kind="secondary"]:hover {
        background: rgba(66,180,255,0.16) !important;
        border-color: var(--cs-blue) !important;
    }

    /* ── Responsive ──────────────────────────────────────────────────────── */
    @media (min-width: 1400px) { .block-container { max-width: 1300px !important; } }
    @media (min-width: 1800px) {
        .block-container { max-width: 1600px !important; padding-left: 3rem; padding-right: 3rem; }
    }
    @media (max-width: 1280px) {
        .block-container { max-width: 100% !important; padding-left: 1rem !important; padding-right: 1rem !important; }
    }
    @media (max-width: 768px) {
        .block-container { padding: 0.5rem !important; }
        html, body, [class*="css"] { font-size: 13px; }
        [data-testid="stSidebar"] .stSelectbox label,
        [data-testid="stSidebar"] .stTextInput label { font-size: 11px !important; }
        button[data-testid="stTab"] p { font-size: 13px !important; }
        div[data-testid="stHorizontalBlock"] { flex-wrap: wrap !important; }
    }
    @media (min-width: 1600px) { [data-testid="stSidebar"] { min-width: 320px !important; } }
</style>
""", unsafe_allow_html=True)


# ── Session state ────────────────────────────────────────────────────
if "chat" not in st.session_state:
    st.session_state.chat = None
if "messages" not in st.session_state:
    st.session_state.messages = []
if "connected" not in st.session_state:
    st.session_state.connected = False
if "explore_questions" not in st.session_state:
    st.session_state.explore_questions = []
if "pending_question" not in st.session_state:
    st.session_state.pending_question = None
if "open_in_builder" not in st.session_state:
    st.session_state.open_in_builder = None
if "db_preview" not in st.session_state:
    st.session_state.db_preview = None

_VIZ_KEYWORDS = {"chart", "graph", "plot", "visuali", "diagram", "bar", "line", "pie", "scatter", "trend"}


def _wants_viz(question: str) -> bool:
    q = question.lower()
    return any(k in q for k in _VIZ_KEYWORDS)


# ── DB preview helper ─────────────────────────────────────────────────

def _fetch_db_preview(conn_type: str, **kwargs) -> dict:
    """Temporarily connect to fetch available schemas and tables.

    Returns {"schemas": [...], "tables_by_schema": {"schema": ["table", ...]}}
    """
    schemas: list[str] = []
    tables_by_schema: dict[str, list[str]] = {}

    if conn_type == "DuckDB":
        import duckdb
        path = kwargs.get("path", ":memory:")
        conn = duckdb.connect(database=path, read_only=(path != ":memory:"))
        try:
            df = conn.execute(
                "SELECT DISTINCT table_schema, table_name "
                "FROM information_schema.tables "
                "WHERE table_schema NOT IN ('information_schema', 'pg_catalog') "
                "ORDER BY table_schema, table_name"
            ).fetchdf()
            schemas = df["table_schema"].unique().tolist()
            for _, row in df.iterrows():
                tables_by_schema.setdefault(row["table_schema"], []).append(row["table_name"])
        finally:
            conn.close()

    elif conn_type == "Exasol (pyexasol)":
        import pyexasol
        conn = pyexasol.connect(
            dsn=kwargs["dsn"], user=kwargs["user"],
            password=kwargs["password"], compression=True,
        )
        try:
            stmt = conn.execute(
                "SELECT DISTINCT TABLE_SCHEMA, TABLE_NAME "
                "FROM EXA_ALL_TABLES ORDER BY 1, 2"
            )
            rows = stmt.fetchall()
            schemas = list(dict.fromkeys(r[0] for r in rows))
            for r in rows:
                tables_by_schema.setdefault(r[0], []).append(r[1])
        finally:
            conn.close()

    else:  # SQLAlchemy
        from sqlalchemy import create_engine, inspect
        engine = create_engine(kwargs["url"])
        try:
            insp = inspect(engine)
            schemas = insp.get_schema_names()
            for s in schemas:
                try:
                    tables_by_schema[s] = insp.get_table_names(schema=s)
                except Exception:
                    tables_by_schema[s] = []
        finally:
            engine.dispose()

    return {"schemas": schemas, "tables_by_schema": tables_by_schema}


# ── Helper functions ─────────────────────────────────────────────────

import re as _re

def _clean_summary(text: str) -> str:
    """Strip markdown bold/italic markers that cause rendering artifacts.

    LLM summaries sometimes wrap numbers or words in *asterisks* which
    Streamlit renders as italic/bold, breaking readability.
    """
    # Remove bold (**text**) and italic (*text* / _text_) markers
    text = _re.sub(r"\*\*([^*]+)\*\*", r"\1", text)
    text = _re.sub(r"\*([^*]+)\*", r"\1", text)
    text = _re.sub(r"_([^_]+)_", r"\1", text)
    return text


def _fmt_labels(series: pd.Series) -> list:
    """Format numeric values as compact human-readable strings for chart labels."""
    vals = series.dropna()
    if vals.empty:
        return [""] * len(series)
    max_abs = float(vals.abs().max())

    def _f(v):
        if pd.isna(v):
            return ""
        v = float(v)
        if max_abs >= 1e9:
            return f"{v/1e9:.1f}B"
        if max_abs >= 1e6:
            return f"{v/1e6:.1f}M"
        if max_abs >= 1e3:
            return f"{v/1e3:.1f}K"
        if max_abs >= 1:
            try:
                return f"{v:,.0f}" if v == int(v) else f"{v:.3g}"
            except (ValueError, OverflowError):
                return f"{v:.3g}"
        return f"{v:.3g}"

    return [_f(v) for v in series]


_CHART_PALETTE = ["#486de8", "#f59e0b", "#8b5cf6", "#06b6d4", "#84cc16", "#ec4899"]
_C_POS_NEG     = "#018977"  # positive bars when series contains negatives
_C_NEG         = "#962249"  # negative bars (any series)
_C_ONLY_POS    = "#486de8"  # bars that are entirely non-negative (single series)


def _dual_axis_zero_ranges(y1_cols, y2_cols, df):
    """Return aligned (y1_range, y2_range) so both Y axes share a single zero level.

    When one or both axes cross zero, the zero tick must be at the same fractional
    height on both axes to avoid two separate baseline bands.
    """
    try:
        def _span(cols):
            vals = pd.concat(
                [df[c] for c in cols if c in df.columns], ignore_index=True
            ).dropna()
            if vals.empty:
                return 0.0, 1.0
            return float(vals.min()), float(vals.max())

        y1_mn, y1_mx = _span(y1_cols)
        y2_mn, y2_mx = _span(y2_cols)

        # Clamp both to include zero
        y1_mn, y1_mx = min(0.0, y1_mn), max(0.0, y1_mx)
        y2_mn, y2_mx = min(0.0, y2_mn), max(0.0, y2_mx)

        # Fraction of the range that sits below zero
        def _neg_frac(mn, mx):
            r = mx - mn
            return abs(mn) / r if r > 0 else 0.0

        # Adopt the larger fraction so neither axis clips its data
        frac = max(_neg_frac(y1_mn, y1_mx), _neg_frac(y2_mn, y2_mx))
        if frac == 0.0:
            return None, None  # no negatives anywhere — no alignment needed

        def _new_range(mn, mx, f, pad=0.08):
            abs_mn = abs(mn)
            # Positive span required so zero lands at fraction f
            pos_needed = abs_mn * (1 - f) / f if f > 0 else mx
            # Negative span required so zero lands at fraction f
            neg_needed = mx * f / (1 - f) if (1 - f) > 0 else abs_mn
            pos_span = max(mx, pos_needed)
            neg_span = max(abs_mn, neg_needed)
            total = neg_span + pos_span
            return [-neg_span - total * pad, pos_span + total * pad]

        return _new_range(y1_mn, y1_mx, frac), _new_range(y2_mn, y2_mx, frac)
    except Exception:
        return None, None


def _build_chart_figure(
    df: pd.DataFrame,
    x_col: str,
    y1_cols: list,
    y2_cols: list,
    chart_type: str,
) -> object:
    """Build a Plotly figure with dual-axis, brand color coding, and data labels.

    Color rules:
      - Entirely non-negative series → #486de8 (or palette for multi-series)
      - Series with negatives        → positive bars #018977, negative bars #e07f9d
    Dual-axis zeros are aligned so there is only one baseline band.
    Labels appear for small datasets; uniformtext hides overlapping labels gracefully.
    """
    import plotly.graph_objects as go
    import plotly.express as _px

    n_rows  = len(df)
    all_cols = y1_cols + y2_cols

    # ── Pie: special-case (no dual axis) ──────────────────────────
    if chart_type == "pie" and y1_cols:
        fig = _px.pie(df, names=x_col, values=y1_cols[0])
        fig.update_traces(textinfo="label+percent", textposition="auto")
        fig.update_layout(margin=dict(t=30, b=0), legend_title_text="")
        return fig

    fig = go.Figure()

    bar_labels  = chart_type == "bar"            and n_rows <= 30
    line_labels = chart_type in ("line", "area") and n_rows <= 20
    scat_labels = chart_type == "scatter"         and n_rows <= 15

    for axis_n, (cols, yaxis) in enumerate([(y1_cols, "y"), (y2_cols, "y2")]):
        for i, col in enumerate(cols):
            if col not in df.columns:
                continue
            pal_idx  = (len(y1_cols) if axis_n else 0) + i
            vals     = df[col]
            col_has_neg = pd.api.types.is_numeric_dtype(vals) and (vals < 0).any()

            # Resolve base color for this series
            if col_has_neg:
                base = _C_POS_NEG  # positive split-trace for mixed series
            else:
                base = _CHART_PALETTE[pal_idx % len(_CHART_PALETTE)]

            if chart_type == "bar":
                if col_has_neg:
                    # Split into positive (shown in legend) + negative (hidden from
                    # legend, colored red). offsetgroup keeps both at the same x slot.
                    pos = vals.where(vals >= 0)
                    neg = vals.where(vals < 0)
                    common = dict(yaxis=yaxis, offsetgroup=col, cliponaxis=False)
                    fig.add_trace(go.Bar(
                        name=col, x=df[x_col], y=pos,
                        marker_color=_C_POS_NEG, showlegend=True,
                        text=_fmt_labels(pos) if bar_labels else None,
                        textposition="outside", **common,
                    ))
                    fig.add_trace(go.Bar(
                        name=col, x=df[x_col], y=neg,
                        marker_color=_C_NEG, showlegend=False,
                        text=_fmt_labels(neg) if bar_labels else None,
                        textposition="outside", **common,
                    ))
                else:
                    fig.add_trace(go.Bar(
                        name=col, x=df[x_col], y=vals,
                        marker_color=base, yaxis=yaxis,
                        offsetgroup=col, cliponaxis=False,
                        text=_fmt_labels(vals) if bar_labels else None,
                        textposition="outside",
                    ))

            elif chart_type == "line":
                mode = "lines+markers" + ("+text" if line_labels else "")
                fig.add_trace(go.Scatter(
                    name=col, x=df[x_col], y=vals,
                    mode=mode, line_color=base, yaxis=yaxis,
                    text=_fmt_labels(vals) if line_labels else None,
                    textposition="top center", textfont=dict(size=10),
                ))
                if col_has_neg:
                    fig.add_hline(y=0, line_width=1, line_dash="dot",
                                  line_color="rgba(255,255,255,0.25)")

            elif chart_type == "area":
                mode = "lines" + ("+text" if line_labels else "")
                fig.add_trace(go.Scatter(
                    name=col, x=df[x_col], y=vals,
                    mode=mode, fill="tozeroy", line_color=base, yaxis=yaxis,
                    text=_fmt_labels(vals) if line_labels else None,
                    textposition="top center", textfont=dict(size=10),
                ))
                if col_has_neg:
                    fig.add_hline(y=0, line_width=1, line_dash="dot",
                                  line_color="rgba(255,255,255,0.25)")

            elif chart_type == "scatter":
                mode = "markers" + ("+text" if scat_labels else "")
                fig.add_trace(go.Scatter(
                    name=col, x=df[x_col], y=vals,
                    mode=mode, marker_color=base, yaxis=yaxis,
                    text=_fmt_labels(vals) if scat_labels else None,
                    textposition="top center", textfont=dict(size=10),
                ))

    # ── Layout ────────────────────────────────────────────────────
    barmode = "group" if len(all_cols) > 1 else "relative"
    layout: dict = dict(
        barmode=barmode,
        margin=dict(t=44, b=40, l=60),
        legend_title_text="",
        legend=dict(
            orientation="h",
            yanchor="bottom", y=1.02,
            xanchor="left", x=0.0,
        ),
        uniformtext_minsize=8,
        uniformtext_mode="hide",
    )

    if y2_cols:
        layout["yaxis"]  = dict(title=", ".join(y1_cols) if y1_cols else "")
        layout["yaxis2"] = dict(
            title=", ".join(y2_cols),
            overlaying="y", side="right", showgrid=False,
        )
        # Align zeros so there is exactly one baseline across both axes
        r1, r2 = _dual_axis_zero_ranges(y1_cols, y2_cols, df)
        if r1 is not None:
            layout["yaxis"]["range"]  = r1
            layout["yaxis2"]["range"] = r2
    fig.update_layout(**layout)
    return fig


def _render_chart_dynamic(
    r: QueryResult,
    df: pd.DataFrame,
    y1_cols: list,
    y2_cols: list,
    dim_cols: list,
    chart_type_override: str | None = None,
    x_override: str | None = None,
) -> bool:
    """Render a chart. Returns True if a chart was drawn."""
    if not y1_cols and not y2_cols or df.empty:
        return False

    cfg = r.chart_config or {}
    chart_type = chart_type_override or cfg.get("chart_type", "bar")
    if chart_type == "table_only":
        return False

    if x_override and x_override in df.columns:
        x_col = x_override
    else:
        x_col = cfg.get("x")
        if not x_col or x_col not in df.columns:
            x_col = dim_cols[0] if dim_cols else None
    if not x_col:
        return False

    try:
        fig = _build_chart_figure(df, x_col, y1_cols, y2_cols, chart_type)
        st.plotly_chart(fig, use_container_width=True)
        return True
    except Exception:
        return False


def _render_result(r: QueryResult, elapsed: float | None = None):
    key = hash(r.question)

    if r.error:
        # Plain-English lead — strip the raw exception prefix for cleaner display
        _err_display = r.error
        for _prefix in ("Query execution failed: ", "Query blocked: ", "LLM generation failed: "):
            if _err_display.startswith(_prefix):
                _err_display = _err_display[len(_prefix):]
                break
        st.error(f"**Query failed** — {_err_display}")
        if r.original_error and r.original_error != _err_display:
            with st.expander("📋 Error detail", expanded=False):
                st.code(r.original_error, language="text")
        if r.sql:
            with st.expander("🔍 Generated SQL", expanded=False):
                st.code(r.sql, language="sql")
        return

    # ── Agent mode result layout ──────────────────────────────────────
    if r.agent_steps is not None:
        # Plan (if captured)
        if r.agent_plan:
            with st.expander("🗺 Investigation plan", expanded=False):
                for i, step in enumerate(r.agent_plan, 1):
                    st.markdown(f"{i}. {step}")

        # Narrative — shown prominently above everything else
        if r.summary:
            st.markdown(
                f'<div style="background:rgba(99,102,241,0.08);border-left:3px solid #6366f1;'
                f'padding:12px 16px;border-radius:6px;margin-bottom:12px;">'
                f'<div style="font-size:0.75rem;font-weight:600;color:#6366f1;'
                f'text-transform:uppercase;letter-spacing:0.05em;margin-bottom:6px;">📝 Finding</div>'
                f'<div style="font-size:0.95rem;line-height:1.6;">{_clean_summary(r.summary)}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )

        if elapsed is not None:
            st.markdown(
                f'<div style="font-size:0.82rem;color:#a4a4ad;margin:4px 0 12px;">'
                f'🧠 Agent · {len(r.agent_steps)} steps · ⏱ {elapsed:.1f}s</div>',
                unsafe_allow_html=True,
            )

        # Agent trace (collapsed by default)
        if r.agent_steps:
            total_ms = sum(s.elapsed_ms for s in r.agent_steps)
            with st.expander(
                f"🔍 Agent trace — {len(r.agent_steps)} steps · {total_ms/1000:.1f}s",
                expanded=False,
            ):
                for step in r.agent_steps:
                    icon = "✓" if not step.error else "✗"
                    elapsed_fmt = (
                        f"{step.elapsed_ms/1000:.1f}s"
                        if step.elapsed_ms >= 1000
                        else f"{step.elapsed_ms}ms"
                    )
                    st.markdown(
                        f"**Step {step.step_num} · `{step.tool_name}`** · {elapsed_fmt} {icon}"
                    )
                    if step.tool_name == "run_sql" and step.tool_input.get("sql"):
                        st.code(step.tool_input["sql"], language="sql")
                    elif step.tool_name == "create_plan" and step.tool_input.get("steps"):
                        for i, s in enumerate(step.tool_input["steps"], 1):
                            st.caption(f"{i}. {s}")
                    elif step.tool_name not in ("final_answer",) and step.tool_output:
                        st.text(step.tool_output[:400])
                    if step.error:
                        st.error(step.error)

        # SQL used (agent mode)
        if r.sql:
            with st.expander("🔍 SQL used", expanded=False):
                st.code(r.sql, language="sql")

    # ── Classic mode decorations (skipped in agent mode) ─────────────
    if r.agent_steps is None and r.auto_corrected:
        with st.expander("✅ Query had an issue that was automatically corrected", expanded=False):
            if r.correction_explanation:
                st.caption(r.correction_explanation)
            if r.original_sql:
                st.markdown("**Original SQL (failed):**")
                st.code(r.original_sql, language="sql")
            if r.original_error:
                st.markdown("**Error:**")
                st.code(r.original_error, language="text")
            st.markdown("**Corrected SQL (used):**")
            st.code(r.sql, language="sql")

    if r.agent_steps is None and r.column_warnings:
        for w in r.column_warnings:
            st.markdown(f'<div class="col-warning">{w}</div>', unsafe_allow_html=True)

    if r.agent_steps is None and r.summary:
        st.markdown(_clean_summary(r.summary))

    # ── Proactive insights — relevant past findings surfaced automatically ──
    # Pull from the session's BusinessModel; score against this question.
    try:
        _bm_ref = st.session_state.get("chat") and st.session_state.chat.business_model
        if _bm_ref and _bm_ref.finding_count() > 0:
            from talonsight.memory import _relevance_score
            _all_findings = _bm_ref.get_recent_findings(50)
            _current_q    = r.question
            _current_narr = r.summary or ""
            _scored = [
                (f, _relevance_score(_current_q, f["question"] + " " + f["narrative"]))
                for f in _all_findings
                # Exclude this exact question to avoid self-reference
                if f["question"].strip().lower() != _current_q.strip().lower()
            ]
            _relevant = [f for f, s in sorted(_scored, key=lambda x: x[1], reverse=True)
                         if s > 0.15][:3]
            if _relevant:
                with st.expander("💡 Related insights from past investigations", expanded=False):
                    for _pf in _relevant:
                        _pts = _pf.get("timestamp","")[:10]
                        st.markdown(
                            f'<div style="border-left:3px solid #6366f1;'
                            f'padding:8px 12px;margin-bottom:8px;'
                            f'border-radius:0 6px 6px 0;background:#0f1929">'
                            f'<div style="color:#818cf8;font-size:.72rem;margin-bottom:3px">'
                            f'🕐 {_pts}</div>'
                            f'<div style="color:#94a3b8;font-size:.78rem;font-weight:500">'
                            f'{_pf["question"]}</div>'
                            f'<div style="color:#64748b;font-size:.76rem;margin-top:4px;'
                            f'line-height:1.5">{_pf["narrative"][:180]}'
                            + ("…" if len(_pf["narrative"]) > 180 else "")
                            + f'</div></div>',
                            unsafe_allow_html=True,
                        )
    except Exception:
        pass  # proactive insights must never crash the render

    if r.agent_steps is None:
        with st.expander("🔍 Generated SQL", expanded=False):
            if r.safety.level == RiskLevel.SAFE:
                badge = '<span class="badge-safe">✓ Safe</span>'
            elif r.safety.level == RiskLevel.SUSPICIOUS:
                badge = f'<span class="badge-warn">⚠ {r.safety.reason}</span>'
            else:
                badge = f'<span class="badge-blocked">✕ Blocked: {r.safety.reason}</span>'
            st.markdown(badge, unsafe_allow_html=True)
            st.code(r.sql, language="sql")
            if r.explanation:
                st.caption(r.explanation)
            if r.kb_patterns_used > 0:
                st.markdown(
                    f'<div class="kb-indicator">📖 {r.kb_patterns_used} KB pattern{"s" if r.kb_patterns_used != 1 else ""} guided this query</div>',
                    unsafe_allow_html=True,
                )

        if elapsed is not None:
            st.markdown(
                f'<div style="font-size:0.82rem;color:#a4a4ad;margin:4px 0 12px;font-weight:500;letter-spacing:0.01em;">⏱ {elapsed:.1f}s</div>',
                unsafe_allow_html=True,
            )

    if r.data is not None and len(r.data) > 0:
        df = r.data
        num_cols = df.select_dtypes(include="number").columns.tolist()
        dim_cols = df.select_dtypes(exclude="number").columns.tolist()
        has_chart_data = len(num_cols) >= 1 and len(df.columns) >= 2

        # ── Viz control bar: chart-type | x-axis | Y1 measures | Y2 (right axis) ──
        ctl1, ctl2, ctl3, ctl4 = st.columns([2, 2, 3, 3])
        with ctl1:
            viz_choice = st.selectbox(
                "Chart type",
                ["auto", "bar", "line", "area", "scatter", "pie", "table only"],
                key=f"viz_{key}",
                help="Override the AI-suggested chart type",
            )

        # Resolve effective chart behaviour
        if viz_choice == "table only":
            chart_type_override = "table_only"
            show_chart = False
        elif viz_choice == "auto":
            chart_type_override = None
            show_chart = (
                has_chart_data
                and r.chart_obj is not None
                and (r.chart_config or {}).get("chart_type") != "table_only"
            )
        else:
            chart_type_override = viz_choice
            show_chart = has_chart_data

        # X-axis / dimension selector
        x_override: str | None = None
        with ctl2:
            if show_chart and dim_cols:
                _all_xax = dim_cols + [c for c in num_cols if c not in dim_cols]
                x_override = st.selectbox(
                    "Dimension (X axis)",
                    _all_xax,
                    key=f"xax_{key}",
                    help="Column to use as the X axis / dimension",
                )

        # Y1 (left axis) measure multiselect
        y1_measures = num_cols
        with ctl3:
            if show_chart and num_cols:
                y1_measures = st.multiselect(
                    "Measures (left Y)",
                    options=num_cols,
                    default=num_cols,
                    key=f"ms_{key}",
                    help="Measures on the left Y axis",
                    placeholder="Select measures",
                )

        # Y2 (right axis) — subset of y1_measures; only shown when 2+ numerics available
        y2_measures: list = []
        with ctl4:
            if show_chart and len(num_cols) > 1 and y1_measures:
                y2_measures = st.multiselect(
                    "Second axis (right Y)",
                    options=y1_measures,
                    default=[],
                    key=f"y2_{key}",
                    help="Move these measures to the right Y axis (dual axis)",
                    placeholder="Select for dual axis",
                )

        # Effective Y1 = all selected minus those promoted to Y2
        eff_y1 = [c for c in y1_measures if c not in y2_measures]

        # ── Chart rendering ────────────────────────────────────────
        chart_rendered = False
        if show_chart and (eff_y1 or y2_measures):
            chart_rendered = _render_chart_dynamic(
                r, df, eff_y1, y2_measures, dim_cols,
                chart_type_override, x_override,
            )

        # ── Data table (always shown so sorted data is visible) ───
        st.dataframe(df, use_container_width=True, height=min(400, 35 * len(df) + 50))

        col_dl, col_nav = st.columns([1, 5])
        with col_dl:
            st.download_button(
                "📥 CSV", df.to_csv(index=False), "query_result.csv", "text/csv",
                use_container_width=True, key=f"dl_{key}",
            )
        with col_nav:
            _stc.html("""
            <style>
                html, body { margin:0; padding:0; height:38px; overflow:hidden; }
                body { display:flex; align-items:center; justify-content:flex-end; }
                .tab-nav { display:flex; align-items:center; gap:6px; }
                .tab-nav-btn {
                    background: transparent;
                    border: 1px solid #424650;
                    color: #a4a4ad;
                    border-radius: 6px;
                    padding: 0 12px;
                    height: 32px;
                    font-size: 12px;
                    font-family: 'Open Sans', sans-serif;
                    cursor: pointer;
                    white-space: nowrap;
                    transition: border-color 0.15s, color 0.15s;
                }
                .tab-nav-btn:hover { border-color: #42b4ff; color: #42b4ff; }
            </style>
            <div class="tab-nav">
                <button class="tab-nav-btn" onclick="goTab('Build')">📊 Build</button>
                <button class="tab-nav-btn" onclick="goTab('Metrics')">📐 Metrics</button>
                <button class="tab-nav-btn" onclick="goTab('Schema')">🗺️ Schema</button>
            </div>
            <script>
            function goTab(name) {
                var tabs = window.parent.document.querySelectorAll('button[role="tab"]');
                for (var i = 0; i < tabs.length; i++) {
                    if (tabs[i].innerText.trim().indexOf(name) !== -1) {
                        tabs[i].click(); break;
                    }
                }
            }
            </script>
            """, height=38)

    # Follow-up suggestions
    if r.followups:
        st.markdown(
            '<div style="font-size:0.75rem;color:#a4a4ad;margin-top:12px;margin-bottom:4px;font-weight:600;text-transform:uppercase;letter-spacing:0.05em;">Suggested follow-ups</div>',
            unsafe_allow_html=True,
        )
        cols = st.columns(len(r.followups))
        for i, (col, q) in enumerate(zip(cols, r.followups)):
            with col:
                if st.button(q, key=f"fu_{key}_{i}", use_container_width=True):
                    st.session_state.pending_question = q
                    st.rerun()


# ── Sidebar ──────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚡ talonsight")

    st.divider()

    # ── 1. DATABASE ───────────────────────────────────────────────────
    st.markdown('<p style="font-size:11px;font-weight:700;text-transform:uppercase;letter-spacing:0.08em;color:#a4a4ad;margin:0 0 8px;">Database</p>', unsafe_allow_html=True)
    _conn_options = ["DuckDB", "PostgreSQL", "Exasol (pyexasol)", "SQLAlchemy URL"]
    conn_type = st.selectbox("Connection type", _conn_options)

    if st.session_state.get("_prev_conn_type") != conn_type:
        st.session_state.db_preview = None
        st.session_state["_prev_conn_type"] = conn_type

    if conn_type == "Exasol (pyexasol)":
        exa_host = st.text_input("Host:Port", placeholder="host:8563")
        exa_user = st.text_input("Username")
        exa_pass = st.text_input("Password", type="password")
    elif conn_type == "DuckDB":
        duck_path = st.text_input(
            "Database path",
            value=_DEFAULT_DUCKDB_PATH,
            placeholder=":memory: or /path/to/file.duckdb",
        )
    elif conn_type == "PostgreSQL":
        pg_host = st.text_input("Host", value="localhost")
        pg_port = st.number_input("Port", value=5432, min_value=1, max_value=65535)
        pg_db   = st.text_input("Database", placeholder="mydb")
        pg_user = st.text_input("Username", placeholder="myuser")
        pg_pass = st.text_input("Password", type="password")
    else:
        db_url = st.text_input("Connection URL", placeholder="sqlite:///data.db")

    if st.button("🔍 Load schemas & tables", key="b_db_preview", use_container_width=True):
        try:
            with st.spinner("Connecting…"):
                _kw: dict = {}
                if conn_type == "DuckDB":
                    _kw["path"] = duck_path or ":memory:"
                elif conn_type == "Exasol (pyexasol)":
                    _kw.update(dsn=exa_host, user=exa_user, password=exa_pass)
                elif conn_type == "PostgreSQL":
                    _kw["url"] = (
                        f"postgresql+psycopg://{pg_user}:{pg_pass}"
                        f"@{pg_host}:{int(pg_port)}/{pg_db}"
                    )
                else:
                    _kw["url"] = db_url
                st.session_state.db_preview = _fetch_db_preview(conn_type, **_kw)
        except Exception as _pe:
            st.error(f"Preview failed: {_pe}")

    _preview = st.session_state.db_preview

    st.divider()

    # ── 2. ACCESS CONTROL (includes schema selector) ──────────────────
    st.markdown('<p style="font-size:11px;font-weight:700;text-transform:uppercase;letter-spacing:0.08em;color:#a4a4ad;margin:8px 0 8px;">Access Control</p>', unsafe_allow_html=True)

    # Schema selector
    if conn_type == "Exasol (pyexasol)":
        if _preview:
            _exa_opts = ["(none)"] + _preview["schemas"]
            _exa_sel = st.selectbox("Schema", _exa_opts, key="exa_schema_sel")
            exa_schema = "" if _exa_sel == "(none)" else _exa_sel
        else:
            exa_schema = st.text_input("Schema", placeholder="MY_SCHEMA")
    elif conn_type == "DuckDB":
        if _preview:
            _duck_schemas = _preview["schemas"]
            _duck_idx = _duck_schemas.index("main") if "main" in _duck_schemas else 0
            duck_schema = st.selectbox("Schema", _duck_schemas, index=_duck_idx, key="duck_schema_sel")
        else:
            duck_schema = st.text_input("Schema", value="main")
    elif conn_type == "PostgreSQL":
        if _preview:
            _pg_schema_opts = [s for s in _preview["schemas"] if s not in ("information_schema", "pg_catalog", "pg_toast")]
            _pg_idx = _pg_schema_opts.index("public") if "public" in _pg_schema_opts else 0
            pg_schema = st.selectbox("Schema", _pg_schema_opts, index=_pg_idx, key="pg_schema_sel")
        else:
            pg_schema = st.text_input("Schema", value="public")

    # Table restriction
    if _preview:
        _ac_schema: str | None = None
        if conn_type == "Exasol (pyexasol)":
            _ac_schema = exa_schema or None
        elif conn_type == "DuckDB":
            _ac_schema = duck_schema or None
        elif conn_type == "PostgreSQL":
            _ac_schema = pg_schema or None

        if _ac_schema and _ac_schema in _preview["tables_by_schema"]:
            _avail_tables = _preview["tables_by_schema"][_ac_schema]
        else:
            _avail_tables = sorted({t for ts in _preview["tables_by_schema"].values() for t in ts})

        _n_total = sum(len(v) for v in _preview["tables_by_schema"].values())
        st.caption(f"{_n_total} tables · {len(_preview['schemas'])} schemas")

        _ac_tables_sel = st.multiselect(
            "Restrict tables (optional)",
            _avail_tables,
            key="ac_tables_sel",
            help="Leave empty to allow all tables in the selected schema",
        )
        allowed_schemas_str = _ac_schema or ""
        allowed_tables_str  = ",".join(_ac_tables_sel)
    else:
        allowed_schemas_str = st.text_input(
            "Allowed schemas",
            placeholder="SALES, ANALYTICS",
        )
        allowed_tables_str = st.text_input(
            "Allowed tables",
            placeholder="CUSTOMERS, ORDERS",
        )

    st.divider()

    # ── 3. CONNECT / SESSION BUTTONS ──────────────────────────────────
    # Pre-fill MLX fields only on Apple Silicon — meaningless elsewhere.
    _IS_APPLE_SILICON = sys.platform == "darwin" and platform.machine() == "arm64"

    # Config written by the first-time setup wizard (overrides built-in defaults)
    _wiz = _load_wizard_config()

    # Seed session_state defaults once (avoids value= + key= conflict in widgets below).
    _sb_defaults = {
        "_sb_llm_type":      _wiz.get("llm_backend",
                                 "MLX (Apple Silicon)" if _IS_APPLE_SILICON else "Ollama"),
        "_sb_ollama_url":    _wiz.get("ollama_url",
                                 _DEFAULT_OLLAMA_URL or "http://localhost:11434"),
        "_sb_ollama_model":  _wiz.get("ollama_model",
                                 _DEFAULT_OLLAMA_MODEL or "qwen3:8b"),
        "_sb_api_url":       _wiz.get("api_url",   "http://localhost:1234/v1"),
        "_sb_api_model":     _wiz.get("api_model", "local-model"),
        "_sb_api_key":       "not-needed",
        "_sb_mlx_url":       _wiz.get("mlx_url",
                                 "http://localhost:8080/v1" if _IS_APPLE_SILICON else ""),
        "_sb_mlx_model":     _wiz.get("mlx_model",
                                 "mlx-community/Qwen3-8B-4bit" if _IS_APPLE_SILICON else ""),
        "_sb_embed_backend": "Bag of words (offline)",
        "_sb_embed_url":     "",
        "_sb_embed_model":   "nomic-embed-text",
        "_sb_extra_context": "",
        "_sb_max_rows":      5000,
        "_sb_chart_lib":     "auto",
        "_sb_kb_path":       _DEFAULT_KB_PATH      or "",
        "_sb_metrics_path":  _DEFAULT_METRICS_PATH or "",
    }
    for _k, _v in _sb_defaults.items():
        if _k not in st.session_state:
            st.session_state[_k] = _v

    # Now read them for use in the Connect handler (widgets below keep them in sync).
    _llm_type     = st.session_state["_sb_llm_type"]
    _ollama_url   = st.session_state["_sb_ollama_url"]
    _ollama_model = st.session_state["_sb_ollama_model"]
    _api_url      = st.session_state["_sb_api_url"]
    _api_model    = st.session_state["_sb_api_model"]
    _api_key      = st.session_state["_sb_api_key"]
    _mlx_url      = st.session_state["_sb_mlx_url"]
    _mlx_model    = st.session_state["_sb_mlx_model"]
    _embed_backend = st.session_state["_sb_embed_backend"]
    _embed_url     = st.session_state["_sb_embed_url"]
    _embed_model   = st.session_state["_sb_embed_model"]
    _extra_ctx    = st.session_state["_sb_extra_context"]
    _max_rows     = int(st.session_state["_sb_max_rows"])
    _chart_lib    = st.session_state["_sb_chart_lib"]
    _kb_path      = st.session_state["_sb_kb_path"]
    _metrics_path = st.session_state["_sb_metrics_path"]

    _btn_label = "✅ Connected — reconnect" if st.session_state.connected else "⚡ Connect"
    if st.button(_btn_label, use_container_width=True, type="primary"):
        try:
            with st.spinner("Connecting & reading schema..."):
                if conn_type == "Exasol (pyexasol)":
                    if not exa_host or not exa_user or not exa_pass:
                        st.error("Fill in host, user, and password.")
                        st.stop()
                    config = ConnectionConfig.exasol(
                        dsn=exa_host, user=exa_user, password=exa_pass,
                        schema=exa_schema or None,
                    )
                elif conn_type == "DuckDB":
                    if not duck_path:
                        st.error("Enter a database path.")
                        st.stop()
                    config = ConnectionConfig.duckdb(path=duck_path)
                elif conn_type == "PostgreSQL":
                    if not pg_db or not pg_user:
                        st.error("Fill in database name and username.")
                        st.stop()
                    _pg_url = (
                        f"postgresql+psycopg://{pg_user}:{pg_pass}"
                        f"@{pg_host}:{int(pg_port)}/{pg_db}"
                    )
                    config = ConnectionConfig.from_url(_pg_url)
                else:
                    if not db_url:
                        st.error("Enter a database URL.")
                        st.stop()
                    config = ConnectionConfig.from_url(db_url)

                if _llm_type == "Ollama":
                    llm = OllamaBackend(model=_ollama_model, base_url=_ollama_url)
                elif _llm_type == "MLX (Apple Silicon)":
                    llm = MLXBackend(base_url=_mlx_url, model=_mlx_model)
                else:
                    llm = OpenAICompatibleBackend(
                        base_url=_api_url, model=_api_model, api_key=_api_key,
                    )

                allowed_schemas = None
                if allowed_schemas_str.strip():
                    allowed_schemas = [s.strip() for s in allowed_schemas_str.split(",") if s.strip()]
                allowed_tables = None
                if allowed_tables_str.strip():
                    allowed_tables = [t.strip() for t in allowed_tables_str.split(",") if t.strip()]

                if conn_type == "Exasol (pyexasol)":
                    schema_param = exa_schema or None
                elif conn_type == "DuckDB":
                    schema_param = duck_schema if duck_schema and duck_schema != "main" else None
                elif conn_type == "PostgreSQL":
                    schema_param = pg_schema if pg_schema and pg_schema != "public" else None
                else:
                    schema_param = None

                # Map sidebar label → backend key used by build_embedding_fn
                _embed_backend_key = {
                    "Bag of words (offline)": "bow",
                    "Ollama":                 "ollama",
                }.get(_embed_backend, "bow")

                chat = TalonSight(
                    connection=config,
                    llm=llm,
                    schema=schema_param,
                    allowed_schemas=allowed_schemas,
                    allowed_tables=allowed_tables,
                    extra_context=_extra_ctx,
                    max_rows=_max_rows,
                    kb_path=_kb_path.strip() or None,
                    chart_library=_chart_lib,
                    metrics_path=_metrics_path.strip() or None,
                    embedding_backend=_embed_backend_key,
                    embedding_url=_embed_url.strip(),
                    embedding_model=_embed_model.strip() or "nomic-embed-text",
                )

            st.session_state.chat = chat
            st.session_state.connected = True
            st.session_state.messages = []
            st.session_state.pending_question = None
            st.session_state.explore_questions = None  # None = pending generation

            st.success(
                f"Connected! {len(chat.schema_context.tables)} tables "
                f"({chat.schema_context.dialect})"
            )
        except Exception as e:
            st.error(f"Connection failed: {e}")

    if st.session_state.connected and st.session_state.chat:
        col_nc, col_dc = st.columns(2)
        with col_nc:
            if st.button("🗨️ New Chat", use_container_width=True,
                         help="Keep connection, clear conversation"):
                st.session_state.chat.clear_history()
                st.session_state.messages = []
                st.session_state.pending_question = None
                st.rerun()
        with col_dc:
            if st.button("🔌 Disconnect", use_container_width=True,
                         help="Close connection and start over"):
                try:
                    st.session_state.chat.close()
                except Exception:
                    pass
                st.session_state.chat = None
                st.session_state.connected = False
                st.session_state.messages = []
                st.session_state.db_preview = None
                st.rerun()

    st.divider()

    # ── 4. LLM BACKEND ────────────────────────────────────────────────
    with st.expander("🤖 LLM Backend", expanded=not st.session_state.connected):
        llm_type = st.selectbox(
            "Backend",
            ["Ollama", "MLX (Apple Silicon)", "OpenAI-compatible API"],
            key="_sb_llm_type",
        )
        if llm_type == "Ollama":
            st.text_input("URL",   key="_sb_ollama_url")
            st.text_input("Model", key="_sb_ollama_model")
        elif llm_type == "MLX (Apple Silicon)":
            st.text_input("Server URL", key="_sb_mlx_url")
            st.text_input(
                "Model",
                key="_sb_mlx_model",
                help="Any mlx-community model, e.g. mlx-community/Qwen3-8B-4bit",
            )
            if _IS_APPLE_SILICON:
                st.caption(
                    "Install once: `pip install talonsight[mlx]`  \n"
                    "The server starts automatically on first query — no terminal needed."
                )
            else:
                st.caption(
                    "⚠️ MLX runs only on Apple Silicon (M-series Mac).  \n"
                    "Enter the URL and model manually if connecting to a remote MLX server."
                )
        else:
            st.text_input("API URL", key="_sb_api_url")
            st.text_input("Model",   key="_sb_api_model")
            st.text_input("API Key", key="_sb_api_key", type="password")

        # ── Live server status ping ───────────────────────────────────
        # Clear stale result when backend type changes
        if st.session_state.get("_ping_last_type") != llm_type:
            st.session_state.pop("_ping_result", None)
            st.session_state["_ping_last_type"] = llm_type

        _ping_col, _ping_btn_col = st.columns([3, 1])
        with _ping_btn_col:
            _do_ping = st.button("🔄", help="Check server connectivity", key="_btn_ping")
        with _ping_col:
            if _do_ping or st.session_state.get("_ping_result"):
                if _do_ping:
                    # Build a temporary backend just for the ping
                    try:
                        if llm_type == "Ollama":
                            from talonsight.llm import OllamaBackend
                            _tmp = OllamaBackend(
                                model=st.session_state.get("_sb_ollama_model", ""),
                                base_url=st.session_state.get("_sb_ollama_url", ""),
                            )
                        elif llm_type == "MLX (Apple Silicon)":
                            _tmp = MLXBackend(
                                base_url=st.session_state.get("_sb_mlx_url", ""),
                                model=st.session_state.get("_sb_mlx_model", ""),
                            )
                        else:
                            from talonsight.llm import OpenAICompatibleBackend
                            _tmp = OpenAICompatibleBackend(
                                base_url=st.session_state.get("_sb_api_url", ""),
                                model=st.session_state.get("_sb_api_model", ""),
                            )
                        _ok, _msg = _tmp.ping()
                        st.session_state["_ping_result"] = (_ok, _msg)
                    except Exception as _pe:
                        st.session_state["_ping_result"] = (False, str(_pe))

                _ok, _msg = st.session_state.get("_ping_result", (None, ""))
                if _ok is True:
                    st.success(_msg, icon="✅")
                elif _ok is False:
                    st.error(_msg, icon="🔴")

    # ── 5. EMBEDDINGS ─────────────────────────────────────────────────
    with st.expander("🔢 Embeddings", expanded=False):
        embed_backend = st.selectbox(
            "Backend",
            ["Bag of words (offline)", "Ollama"],
            key="_sb_embed_backend",
            help=(
                "Bag of words is the default — instant, no download, works offline. "
                "Switch to Ollama for 768-dim semantic embeddings using nomic-embed-text "
                "(run: ollama pull nomic-embed-text)."
            ),
        )
        if embed_backend == "Ollama":
            # Auto-load URL from the Ollama LLM backend setting
            _auto_ollama_url = st.session_state.get("_sb_ollama_url", "").strip() or "http://localhost:11434"
            st.session_state["_sb_embed_url"] = _auto_ollama_url
            st.text_input(
                "Model",
                key="_sb_embed_model",
                help="Run: ollama pull nomic-embed-text",
            )
            st.caption(
                f"URL auto-loaded from Ollama LLM setting: `{_auto_ollama_url}`  \n"
                "Run `ollama pull nomic-embed-text` once to enable semantic search."
            )
        else:
            # Bag of words — no fields needed
            st.session_state["_sb_embed_url"] = ""
            st.caption(
                "Offline keyword matching — no model download, no server required.  \n"
                "Switch to **Ollama** for semantic embeddings via `nomic-embed-text`."
            )

    # ── 6. AGENT MODE ────────────────────────────────────────────────
    with st.expander("🧠 Agent Mode", expanded=False):
        st.toggle(
            "Enable Agent Mode",
            key="_sb_agent_mode",
            value=False,
            help=(
                "Agent mode runs a multi-step autonomous investigation instead of "
                "a single SQL query. The agent plans, explores the schema, executes "
                "iterative queries, and synthesises a narrative answer.\n\n"
                "⚡ Recommended models: qwen3.5:9b, hermes3, llama3.1\n"
                "📥 Install: ollama pull qwen3.5:9b"
            ),
        )
        if st.session_state.get("_sb_agent_mode"):
            st.caption(
                "🧠 **Agent mode active** — each question triggers a full investigation "
                "(5–12 tool calls, ~30–60s on 8B models).  \n"
                "Best models: `qwen3.5:9b` · `hermes3` · `llama3.1:8b`"
            )
            st.slider(
                "Max investigation steps",
                min_value=6,
                max_value=20,
                value=12,
                step=1,
                key="_sb_agent_max_steps",
                help="Hard cap on tool calls per investigation. Higher = more thorough, slower.",
            )
        else:
            st.caption("Off — uses classic one-shot SQL generation.")

    # ── 7. KNOWLEDGE BASE ─────────────────────────────────────────────
    with st.expander("📖 Knowledge Base", expanded=False):
        st.text_input(
            "Extra KB directory",
            key="_sb_kb_path",
            placeholder="/path/to/kb/",
            help="Built-in patterns are always loaded. Set EXACHAT_KB_PATH in .env to pre-fill.",
        )
        if st.session_state.connected and st.session_state.chat:
            st.caption(f"{st.session_state.chat.kb.count} SQL patterns loaded.")

    # ── 7. METRICS CATALOG ────────────────────────────────────────────
    with st.expander("📐 Metrics Catalog", expanded=False):
        st.text_input(
            "Metrics directory",
            key="_sb_metrics_path",
            placeholder="~/.talonsight/metrics/",
            help="Set EXACHAT_METRICS_PATH in .env to pre-fill.",
        )

    # ── 8. OPTIONS ────────────────────────────────────────────────────
    with st.expander("⚙️ Options", expanded=False):
        st.text_area(
            "Extra context / DDL",
            key="_sb_extra_context",
            placeholder="Business rules, column descriptions...",
            height=80,
        )
        st.number_input("Max rows", key="_sb_max_rows", min_value=100, max_value=50000)
        st.selectbox("Chart library", ["auto", "plotly", "altair"], key="_sb_chart_lib")

    # ── 8. SCHEMA EXPLORER (when connected) ───────────────────────────
    if st.session_state.connected and st.session_state.chat:
        st.divider()
        chat_ref = st.session_state.chat
        with st.expander(f"📋 Schema ({len(chat_ref.schema_context.tables)} tables)", expanded=False):
            for table in chat_ref.schema_context.tables:
                label = f'<div class="schema-table-name">{table.name}</div>'
                if table.row_count is not None:
                    label += f'<div class="schema-row-count">{table.row_count:,} rows</div>'
                st.markdown(label, unsafe_allow_html=True)
                cols_text = " · ".join(f"`{c.name}` {c.type}" for c in table.columns[:8])
                if len(table.columns) > 8:
                    cols_text += f" · ... +{len(table.columns) - 8} more"
                st.caption(cols_text)


# ── Main area ────────────────────────────────────────────────────────
if not st.session_state.connected:
    st.markdown("## ⚡ talonsight")
    st.markdown(
        "Connect to your database in the sidebar, then ask questions in plain English. "
        "Powered by local LLMs and a built-in SQL pattern knowledge base."
    )
    st.info(
        "**Prerequisites:** Ollama running locally (or any OpenAI-compatible API) "
        "and a database to connect to."
    )

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("#### Python API")
        st.code(
            'from talonsight import TalonSight\n\n'
            'chat = TalonSight(\n'
            '    "duckdb:///data.duckdb"\n'
            ')\n'
            'result = chat.ask("Top 10 customers by revenue")\n'
            'print(result.data)',
            language="python",
        )
    with col2:
        st.markdown("#### DuckDB")
        st.code(
            'from talonsight import TalonSight\n\n'
            'chat = TalonSight("./analytics.duckdb")\n\n'
            'result = chat.ask("Monthly trends")\n'
            'print(result.summary)',
            language="python",
        )
    with col3:
        st.markdown("#### CLI")
        st.code(
            '# Install\n'
            'pip install talonsight\n\n'
            '# Launch\n'
            'talonsight',
            language="bash",
        )
    st.stop()


# ── Connected — inject top-right status badge ─────────────────────────
_chat_ref = st.session_state.chat
_dial = (_chat_ref.schema_context.dialect or "DB").upper()
_n_tables = len(_chat_ref.schema_context.tables)
st.markdown(
    f'<div style="position:fixed;top:0.45rem;right:1.2rem;z-index:9999;'
    f'background:#0d1f0d;border:1px solid rgba(34,197,94,0.55);border-radius:6px;'
    f'padding:3px 12px;font-size:0.72rem;color:#29ad7f;font-weight:600;'
    f'letter-spacing:0.03em;box-shadow:0 1px 4px rgba(0,0,0,0.4);">'
    f'✅ {_dial} · {_n_tables} tables</div>',
    unsafe_allow_html=True,
)

# ── Tabbed interface ─────────────────────────────────────────────────
chat_engine: TalonSight = st.session_state.chat

# JS reinforcement for sticky tabs (CSS alone unreliable inside Streamlit's scroll container)
import streamlit.components.v1 as _stc
_stc.html(
    """<script>
    (function stickyTabs() {
        const doc = window.parent.document;
        function apply() {
            // Try multiple selectors in priority order
            const candidates = [
                doc.querySelector('[data-testid="stTabs"] > div:first-child'),
                doc.querySelector('div[data-baseweb="tabs"] > [role="tablist"]'),
                doc.querySelector('div[role="tablist"]'),
            ];
            for (const el of candidates) {
                if (el) {
                    el.style.position  = 'sticky';
                    el.style.top       = '2.75rem';
                    el.style.zIndex    = '100';
                    el.style.background = '#161d26';
                    el.style.paddingBottom = '2px';
                    break;
                }
            }
        }
        apply();
        // Re-apply after Streamlit re-renders
        setTimeout(apply, 400);
        setTimeout(apply, 1200);
    })();
    </script>""",
    height=0,
)

# ── Scroll utilities: floating ↓ button + per-tab scroll memory ───────
# Injected once; guard on window.parent._exachatScrollReady survives
# Streamlit soft-reruns so the MutationObserver is never duplicated.
# Scroll technique (from gist.github.com/dtmilano/41a8c45d9e17c663bb970ab318cec96c):
#   append a real sentinel div to parent body → scrollIntoView propagates
#   through all scroll ancestors automatically → remove sentinel.
_stc.html("""<script>
(function () {
    var par = window.parent;
    var pdoc = par.document;

    /* ── already initialised in this page session ── */
    if (par._exachatScrollReady) return;
    par._exachatScrollReady = true;

    /* ── helper: scroll parent page to absolute bottom ── */
    function scrollToBottom() {
        var s = pdoc.createElement('div');
        pdoc.body.appendChild(s);
        s.scrollIntoView({ behavior: 'smooth' });
        setTimeout(function () { if (s.parentNode) s.parentNode.removeChild(s); }, 800);
    }

    /* ── floating ↓ button ── */
    var btn = pdoc.createElement('button');
    btn.id = 'talonsight-scroll-btn';
    btn.title = 'Scroll to bottom';
    btn.innerHTML = '&#8595;';
    Object.assign(btn.style, {
        position:       'fixed',
        bottom:         '1.5rem',
        left:           '50%',
        transform:      'translateX(-50%)',
        zIndex:         '99999',
        width:          '2.2rem',
        height:         '2.2rem',
        borderRadius:   '50%',
        background:     '#42b4ff',
        color:          '#161d26',
        border:         'none',
        fontSize:       '1.15rem',
        fontWeight:     '700',
        cursor:         'pointer',
        boxShadow:      '0 2px 10px rgba(0,0,0,0.45)',
        display:        'flex',
        alignItems:     'center',
        justifyContent: 'center',
        lineHeight:     '1',
        opacity:        '0.7',
        transition:     'opacity 0.15s',
    });
    btn.onmouseenter = function () { btn.style.opacity = '1'; };
    btn.onmouseleave = function () { btn.style.opacity = '0.7'; };
    btn.onclick      = scrollToBottom;
    pdoc.body.appendChild(btn);

    /* ── per-tab scroll memory ── */
    var KEY = '_esc_';          // sessionStorage prefix
    var saveTimer;

    function activeTabKey() {
        var t = pdoc.querySelector('button[role="tab"][aria-selected="true"]');
        return t ? KEY + t.innerText.trim() : KEY + 'default';
    }

    /* save current scroll Y, debounced */
    par.addEventListener('scroll', function () {
        clearTimeout(saveTimer);
        saveTimer = setTimeout(function () {
            sessionStorage.setItem(activeTabKey(), String(Math.round(par.scrollY)));
        }, 120);
    }, { passive: true });

    /* watch for tab selection changes */
    var prevKey = activeTabKey();
    var obs = new MutationObserver(function () {
        var cur = activeTabKey();
        if (cur === prevKey) return;
        prevKey = cur;
        /* wait for tab content to paint, then restore saved position */
        setTimeout(function () {
            var saved = sessionStorage.getItem(cur);
            par.scrollTo({ top: saved ? parseInt(saved, 10) : 0, behavior: 'instant' });
        }, 180);
    });
    /* observe the tabs container; fall back to body if not found yet */
    var root = pdoc.querySelector('[data-testid="stTabs"]') || pdoc.body;
    obs.observe(root, { subtree: true, attributes: true, attributeFilter: ['aria-selected'] });
})();
</script>""", height=0)

tab_ask, tab_intelligence, tab_build, tab_metrics, tab_dq, tab_schema = st.tabs(
    ["💬 Ask", "🧠 Intelligence", "📊 Build", "📐 Metrics", "🔍 Data Quality", "🗺️ Schema"]
)

# ── ASK tab ──────────────────────────────────────────────────────────
with tab_ask:
    # ── Intelligence summary card — shown once, before any messages ────
    if not st.session_state.messages and st.session_state.get("connected") and st.session_state.get("chat"):
        _ce = st.session_state.chat
        _sg = _ce.schema_graph.to_dict()
        _sbm = _ce.business_model

        _dom     = _sg.get("domain", "unknown")
        _dom_col = {
            "e-commerce": "#f97316", "saas": "#6366f1", "finance": "#10b981",
            "hr": "#ec4899", "healthcare": "#06b6d4", "logistics": "#84cc16",
            "analytics": "#a855f7",
        }.get(_dom, "#64748b")
        _dom_lbl = _dom.replace("-", " ").title()

        # Table type counts
        _type_counts: dict[str, int] = {}
        for _tn in _sg.get("nodes", []):
            _type_counts[_tn["type"]] = _type_counts.get(_tn["type"], 0) + 1

        _tt_bg  = {"fact":"#f9731622","dimension":"#6366f122","bridge":"#10b98122","lookup":"#64748b22"}
        _tt_fg  = {"fact":"#f97316","dimension":"#818cf8","bridge":"#34d399","lookup":"#94a3b8"}
        _tt_bd  = {"fact":"#f9731644","dimension":"#6366f144","bridge":"#10b98144","lookup":"#64748b44"}
        _table_pills = " ".join(
            f'<span style="background:{_tt_bg.get(t,"#33415522")};'
            f'color:{_tt_fg.get(t,"#94a3b8")};'
            f'border:1px solid {_tt_bd.get(t,"#33415544")};'
            f'padding:2px 9px;border-radius:12px;font-size:.72rem;font-weight:600">'
            f'{t} {n}</span>'
            for t, n in sorted(_type_counts.items())
        )

        _rels_count = len(_sg.get("relationships", []))
        _bi_parts   = []
        if _sbm.finding_count(): _bi_parts.append(f"{_sbm.finding_count()} findings")
        if _sbm.kpi_count():     _bi_parts.append(f"{_sbm.kpi_count()} KPIs")
        if _sbm.fact_count():    _bi_parts.append(f"{_sbm.fact_count()} facts")
        _bi_str = " · ".join(_bi_parts) if _bi_parts else "No prior knowledge — start asking!"

        st.markdown(
            f"""
            <div style="background:#111827;border:1px solid #1e2d40;
                        border-radius:12px;padding:18px 22px;margin-bottom:20px">
              <div style="display:flex;align-items:center;gap:10px;margin-bottom:10px">
                <span style="background:{_dom_col}22;color:{_dom_col};
                    border:1px solid {_dom_col}55;padding:3px 12px;
                    border-radius:20px;font-size:.76rem;font-weight:700;
                    letter-spacing:.05em">{_dom_lbl}</span>
                <span style="color:#475569;font-size:.76rem">
                  {len(_sg.get("nodes",[]))} tables · {_rels_count} join path{"s" if _rels_count != 1 else ""}
                </span>
              </div>
              <div style="margin-bottom:10px">{_table_pills}</div>
              <div style="color:#64748b;font-size:.76rem;border-top:1px solid #1e2d40;
                          padding-top:8px;margin-top:4px">
                🧠 <span style="color:#475569">Business knowledge:</span>
                <span style="color:#94a3b8">{_bi_str}</span>
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # Explore question grid — shown only before the first message
    if not st.session_state.messages:
        # Lazy-load: generate on first render after connect, not during connect
        if st.session_state.explore_questions is None and st.session_state.connected and st.session_state.chat:
            with st.spinner("Generating starter questions…"):
                st.session_state.explore_questions = st.session_state.chat.generate_explore_questions() or []
            st.rerun()

        if st.session_state.explore_questions:
            st.markdown("#### Where do you want to start?")
            eq = st.session_state.explore_questions
            row1, row2 = eq[:3], eq[3:]
            for row in [row1, row2]:
                cols = st.columns(len(row))
                for col, q in zip(cols, row):
                    with col:
                        if st.button(q, use_container_width=True, key=f"eq_{hash(q)}"):
                            st.session_state.pending_question = q
                            st.rerun()
            st.divider()

    # Render conversation history — always drives the display order
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            if msg["role"] == "user":
                st.markdown(msg["content"])
            elif "result" in msg:
                _render_result(msg["result"], elapsed=msg.get("elapsed"))
            else:
                st.markdown(msg.get("content", ""))

    # Collect next question (typed or from a button)
    pending = st.session_state.get("pending_question")
    if pending:
        st.session_state.pending_question = None

    typed = st.chat_input("Ask a question about your data...")
    question = typed or pending

    if question:
        # Store the user turn, process, store the assistant turn, then rerun.
        # This keeps the chat_input anchored at the bottom and ensures each
        # follow-up appears as a new entry below the previous one.
        st.session_state.messages.append({"role": "user", "content": question})

        _status = st.empty()
        _agent_mode = st.session_state.get("_sb_agent_mode", False)

        if _agent_mode:
            # ── Agent mode — multi-step investigation ─────────────────
            _status.markdown(
                '<div style="color:#a4a4ad;font-size:0.85rem;">🧠 Planning investigation…</div>',
                unsafe_allow_html=True,
            )
            _trace_placeholder = st.empty()
            _step_lines: list[str] = []

            def _on_agent_step(step) -> None:
                icon = "✓" if not step.error else "✗"
                elapsed_s = f"{step.elapsed_ms/1000:.1f}s" if step.elapsed_ms >= 1000 else f"{step.elapsed_ms}ms"
                _step_lines.append(
                    f"Step {step.step_num} &nbsp;·&nbsp; `{step.tool_name}` &nbsp;·&nbsp; "
                    f"{elapsed_s} &nbsp; {icon}"
                )
                _trace_placeholder.markdown(
                    '<div style="color:#a4a4ad;font-size:0.82rem;line-height:1.7">'
                    + "<br>".join(_step_lines)
                    + "</div>",
                    unsafe_allow_html=True,
                )

            _t0 = time.perf_counter()
            _max_steps = st.session_state.get("_sb_agent_max_steps", 12)
            result = chat_engine.ask_agent(
                question,
                on_step=_on_agent_step,
                max_steps=_max_steps,
            )
            _elapsed = time.perf_counter() - _t0
            _status.empty()
            _trace_placeholder.empty()

        else:
            # ── Classic mode — one-shot SQL generation ─────────────────
            _status.markdown(
                '<div style="color:#a4a4ad;font-size:0.85rem;">⏳ Thinking…</div>',
                unsafe_allow_html=True,
            )

            def _on_attempt(attempt: int, total: int) -> None:
                _status.markdown(
                    f'<div style="color:#f0a800;font-size:0.85rem;">'
                    f'⚠️ Attempt {attempt}/{total} — refining query…</div>',
                    unsafe_allow_html=True,
                )

            _t0 = time.perf_counter()
            result = chat_engine.ask(question, on_attempt=_on_attempt)
            _elapsed = time.perf_counter() - _t0
            _status.empty()

        st.session_state.messages.append({
            "role": "assistant",
            "result": result,
            "elapsed": _elapsed,
        })
        st.rerun()

# ── BUILD tab ────────────────────────────────────────────────────────
# ── INTELLIGENCE tab ─────────────────────────────────────────────────
with tab_intelligence:
    if not st.session_state.get("connected") or not st.session_state.get("chat"):
        st.info("Connect to a database to see accumulated business intelligence.")
    else:
        _i_chat = st.session_state.chat
        _i_bm   = _i_chat.business_model
        _i_sg   = _i_chat.schema_graph.to_dict()

        # ── Header ────────────────────────────────────────────────────
        _i_dom     = _i_sg.get("domain", "unknown")
        _i_dom_col = {
            "e-commerce":"#f97316","saas":"#6366f1","finance":"#10b981",
            "hr":"#ec4899","healthcare":"#06b6d4","logistics":"#84cc16",
            "analytics":"#a855f7",
        }.get(_i_dom, "#64748b")
        _i_dom_lbl = _i_dom.replace("-"," ").title()
        _i_conf    = int(_i_sg.get("domain_confidence", 0) * 100)

        st.markdown(
            f'<div style="display:flex;align-items:center;gap:12px;margin-bottom:6px">'
            f'<span style="background:{_i_dom_col}22;color:{_i_dom_col};'
            f'border:1px solid {_i_dom_col}55;padding:4px 14px;border-radius:20px;'
            f'font-size:.82rem;font-weight:700">{_i_dom_lbl}</span>'
            f'<span style="color:#475569;font-size:.8rem">'
            f'{_i_conf}% domain confidence · {_i_sg.get("summary","")}</span>'
            f'</div>',
            unsafe_allow_html=True,
        )
        st.caption(
            "Everything talonsight has learned about your data. "
            "Grows automatically with every Agent Mode investigation."
        )
        st.markdown("---")

        _i_facts    = _i_bm.get_domain_facts()
        _i_kpis     = _i_bm.get_kpis()
        _i_findings = _i_bm.get_recent_findings(50)

        if not _i_facts and not _i_kpis and not _i_findings:
            st.markdown(
                '<div style="text-align:center;padding:60px 0;color:#475569">'
                '<div style="font-size:2rem;margin-bottom:12px">🧠</div>'
                '<div style="font-size:1rem;font-weight:600;color:#64748b">No knowledge yet</div>'
                '<div style="font-size:.85rem;margin-top:6px">Ask questions in Agent Mode.<br>'
                'Every confirmed answer enriches this knowledge base.</div>'
                '</div>',
                unsafe_allow_html=True,
            )
        else:
            _ic1, _ic2 = st.columns([1, 1])

            with _ic1:
                # ── Domain Facts ──────────────────────────────────────
                st.markdown("### 📌 Data Facts")
                if _i_facts:
                    _cat_order = ["date_range", "segment", "quality", "general"]
                    _cat_col   = {"date_range":"#6366f1","segment":"#f97316",
                                  "quality":"#ef4444","general":"#64748b"}
                    for _f in reversed(_i_facts[-20:]):
                        _fc = _cat_col.get(_f.get("category","general"), "#64748b")
                        _conf_badge = (
                            '<span style="color:#f59e0b;font-size:.65rem"> ~inferred</span>'
                            if _f.get("confidence") == "inferred" else ""
                        )
                        st.markdown(
                            f'<div style="border-left:3px solid {_fc};padding:6px 10px;'
                            f'margin-bottom:6px;border-radius:0 6px 6px 0;'
                            f'background:#111827">'
                            f'<span style="color:#e2e8f0;font-size:.82rem">{_f["fact"]}</span>'
                            f'{_conf_badge}'
                            f'</div>',
                            unsafe_allow_html=True,
                        )
                else:
                    st.caption("No data facts confirmed yet.")

                st.markdown("---")

                # ── KPI Definitions ───────────────────────────────────
                st.markdown("### 📊 Discovered KPIs")
                if _i_kpis:
                    for _k in reversed(_i_kpis[-20:]):
                        _kval = (
                            f' &nbsp;<span style="color:#34d399;font-size:.72rem">'
                            f'last: {_k["last_value"]:,.2f}</span>'
                            if _k.get("last_value") is not None else ""
                        )
                        _kunit = f' [{_k["unit"]}]' if _k.get("unit") else ""
                        st.markdown(
                            f'<div style="background:#0f1929;border:1px solid #1e2d40;'
                            f'border-radius:8px;padding:8px 12px;margin-bottom:6px">'
                            f'<div style="color:#6ee7b7;font-size:.82rem;font-weight:600">'
                            f'{_k["name"]}{_kunit}{_kval}</div>'
                            f'<code style="color:#94a3b8;font-size:.74rem">'
                            f'{_k["sql_expression"]}</code>'
                            f'<div style="color:#334155;font-size:.68rem;margin-top:2px">'
                            f'from {_k["table"]}</div>'
                            f'</div>',
                            unsafe_allow_html=True,
                        )
                else:
                    st.caption("No KPIs discovered yet.")

            with _ic2:
                # ── Finding history ───────────────────────────────────
                st.markdown(f"### 🔍 Findings ({len(_i_findings)})")
                if _i_findings:
                    for _fi in _i_findings:
                        _ts  = _fi.get("timestamp","")[:10]
                        _tbls = ", ".join(_fi.get("tables_used",[])[:3])
                        st.markdown(
                            f'<div style="background:#0f1929;border:1px solid #1e2d40;'
                            f'border-radius:8px;padding:10px 14px;margin-bottom:8px">'
                            f'<div style="color:#94a3b8;font-size:.72rem;margin-bottom:4px">'
                            f'{_ts}'
                            + (f' · <span style="color:#334155">{_tbls}</span>' if _tbls else "")
                            + f'</div>'
                            f'<div style="color:#e2e8f0;font-size:.82rem;font-weight:500;'
                            f'margin-bottom:4px">{_fi["question"]}</div>'
                            f'<div style="color:#64748b;font-size:.78rem;line-height:1.5">'
                            f'{_fi["narrative"][:200]}'
                            + ("…" if len(_fi["narrative"]) > 200 else "")
                            + f'</div>'
                            f'</div>',
                            unsafe_allow_html=True,
                        )
                else:
                    st.caption("No findings yet.")

        # ── Data Profile section ───────────────────────────────────────
        st.markdown("---")
        st.markdown("### 🔬 Data Profile")
        st.caption("Column distributions and ranges captured at connect time — injected into every agent prompt.")
        _i_prof = _i_chat.data_profiler.to_dict()
        if not _i_prof:
            st.caption("No profile available.")
        else:
            for _tp_name, _tp in _i_prof.items():
                with st.expander(
                    f'**{_tp["fqn"]}** — {_tp["row_count"]:,} rows · '
                    f'{len(_tp["columns"])} profiled columns',
                    expanded=False,
                ):
                    for _cp in _tp["columns"]:
                        _cp_parts = []
                        if _cp.get("top_values"):
                            _cp_parts.append(
                                " · ".join(f'{v}({n})' for v, n in _cp["top_values"][:5])
                            )
                        elif _cp.get("min") and _cp.get("max"):
                            _rng = f'{_cp["min"]} → {_cp["max"]}'
                            if _cp.get("avg"):
                                _rng += f', avg {_cp["avg"]}'
                            _cp_parts.append(_rng)
                        if _cp.get("null_pct", 0) > 10:
                            _cp_parts.append(f'{_cp["null_pct"]:.0f}% null')
                        _cp_str = "  ·  ".join(_cp_parts)
                        if _cp_str:
                            st.markdown(
                                f'<div style="display:flex;gap:12px;padding:3px 0;'
                                f'border-bottom:1px solid #1e2d40">'
                                f'<span style="color:#6366f1;font-size:.76rem;'
                                f'min-width:180px;font-weight:500">{_cp["name"]}</span>'
                                f'<span style="color:#64748b;font-size:.74rem">{_cp_str}</span>'
                                f'</div>',
                                unsafe_allow_html=True,
                            )

# ── BUILD tab ─────────────────────────────────────────────────────────
with tab_build:
    # If "Open in Builder" was clicked in the Ask tab, seed the builder
    open_sql = st.session_state.pop("open_in_builder", None)
    if open_sql and chat_engine.builder:
        seeded = chat_engine.builder.seed_from_sql(open_sql)
        # Only seed if the table was found and it exists in the schema
        if seeded.get("table") and seeded["table"] in chat_engine.builder.table_names():
            st.session_state.builder = seeded
            st.session_state.pop("builder_result", None)

    render_builder(chat_engine, chat_engine.builder, chat_engine.metrics_catalog)

# ── METRICS tab ──────────────────────────────────────────────────────
with tab_metrics:
    render_metrics_tab(chat_engine.metrics_catalog)

# ── DATA QUALITY tab ─────────────────────────────────────────────────
with tab_dq:
    st.markdown("### 🔍 Data Quality Scan")
    st.caption(
        "Run automated data-quality checks on any connected table — "
        "null rates, blank strings, date format validation, rare categorical values, "
        "duplicate rows, and LLM-detected logical constraint violations."
    )

    _dq_tables = [t.name for t in chat_engine.schema_context.tables]

    if not _dq_tables:
        st.info("Connect to a database first.")
    else:
        _col_sel, _col_btn = st.columns([3, 1])
        with _col_sel:
            _dq_table = st.selectbox(
                "Table to scan",
                _dq_tables,
                key="_dq_selected_table",
            )
        with _col_btn:
            st.write("")   # vertical spacer
            _dq_run = st.button("▶ Run Scan", use_container_width=True, type="primary")

        if _dq_run:
            st.session_state["_dq_results"] = None
            st.session_state["_dq_scanned_table"] = _dq_table

            _dq_progress_text = st.empty()
            _dq_bar = st.progress(0)

            def _dq_on_progress(current, total, rule, col):
                pct = int(current / total * 100)
                _dq_bar.progress(pct)
                _dq_progress_text.caption(
                    f"Running check {current}/{total}: `{rule}` on `{col}`"
                )

            try:
                _dq_raw = chat_engine.run_dq_scan(
                    _dq_table, on_progress=_dq_on_progress
                )
                st.session_state["_dq_results"] = _dq_raw
            except Exception as _dq_err:
                st.error(f"Scan failed: {_dq_err}")
            finally:
                _dq_bar.empty()
                _dq_progress_text.empty()

        # ── Display results ───────────────────────────────────────────
        _dq_results = st.session_state.get("_dq_results")
        _dq_for_table = st.session_state.get("_dq_scanned_table", "")

        if _dq_results is not None:
            if not _dq_results:
                st.success(f"✅ No issues found in `{_dq_for_table}`.")
            else:
                from talonsight.dq import SEVERITY_COLOR
                import pandas as _pd_dq

                _sev_counts = {}
                for r in _dq_results:
                    _sev_counts[r.severity] = _sev_counts.get(r.severity, 0) + 1

                # Summary pills
                _pill_cols = st.columns(len(_sev_counts))
                for _pi, (_sev, _cnt) in enumerate(
                    sorted(_sev_counts.items(), key=lambda x: {"critical":0,"high":1,"medium":2,"low":3}.get(x[0],9))
                ):
                    _pill_cols[_pi].metric(
                        f"{SEVERITY_COLOR.get(_sev,'⚪')} {_sev.capitalize()}",
                        _cnt
                    )

                st.divider()

                # Full results table
                _rows = []
                for r in _dq_results:
                    _rows.append({
                        "Sev": f"{r.severity_icon} {r.severity}",
                        "Rule": r.rule_name,
                        "Column": r.column_name,
                        "Failed": f"{r.failed_count:,}",
                        "Rate": r.failure_pct,
                        "Sample values": ", ".join(r.sample_failed_values[:3]) if r.sample_failed_values else "—",
                        "Error": r.error or "",
                    })

                _df_dq = _pd_dq.DataFrame(_rows)
                st.dataframe(_df_dq, use_container_width=True, hide_index=True)

                # Expandable SQL per result
                with st.expander("🔎 View generated SQL for each check", expanded=False):
                    for r in _dq_results:
                        if r.failed_count > 0 or r.error:
                            st.markdown(
                                f"**{r.severity_icon} {r.rule_name}** — `{r.column_name}`"
                            )
                            st.code(r.generated_sql, language="sql")

# ── SCHEMA MAP tab ────────────────────────────────────────────────────
with tab_schema:
    import re as _re2
    from talonsight.schema import get_join_map

    tables = chat_engine.schema_context.tables
    jmap   = get_join_map(tables)

    # ── Schema Intelligence panel ──────────────────────────────────────
    _graph = chat_engine.schema_graph
    _gd    = _graph.to_dict()

    _domain_colours = {
        "e-commerce": "#f97316", "saas": "#6366f1", "finance": "#10b981",
        "hr": "#ec4899", "healthcare": "#06b6d4", "logistics": "#84cc16",
        "analytics": "#a855f7", "general": "#64748b", "unknown": "#64748b",
    }
    _domain = _gd.get("domain", "unknown")
    _domain_col = _domain_colours.get(_domain, "#64748b")
    _domain_label = _domain.replace("-", " ").title()
    _conf_pct = int(_gd.get("domain_confidence", 0) * 100)

    st.markdown(
        f"""
        <div style="
            background: #1a2236;
            border: 1px solid #2d3a50;
            border-radius: 10px;
            padding: 16px 20px;
            margin-bottom: 18px;
        ">
          <div style="display:flex; align-items:center; gap:12px; margin-bottom:10px">
            <span style="
                background:{_domain_col}22; color:{_domain_col};
                border:1px solid {_domain_col}55;
                padding:3px 12px; border-radius:20px;
                font-size:0.8rem; font-weight:600; letter-spacing:.04em;
            ">{_domain_label}</span>
            <span style="color:#94a3b8; font-size:0.78rem">{_conf_pct}% domain confidence</span>
          </div>
          <p style="color:#cbd5e1; font-size:0.85rem; margin:0">{_gd.get("summary","")}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Table role cards
    _type_order = ["fact", "dimension", "bridge", "lookup", "unknown"]
    _type_col   = {
        "fact": "#f97316", "dimension": "#6366f1",
        "bridge": "#10b981", "lookup": "#64748b", "unknown": "#334155",
    }
    _nodes_by_type: dict[str, list] = {t: [] for t in _type_order}
    for _n in _gd.get("nodes", []):
        _nodes_by_type.setdefault(_n["type"], []).append(_n)

    _role_cols = st.columns(len([t for t in _type_order if _nodes_by_type[t]]))
    _col_idx = 0
    for _ttype in _type_order:
        _tnodes = _nodes_by_type[_ttype]
        if not _tnodes:
            continue
        _tc = _type_col[_ttype]
        with _role_cols[_col_idx]:
            _col_idx += 1
            st.markdown(
                f'<p style="color:{_tc};font-size:.75rem;font-weight:600;'
                f'letter-spacing:.06em;margin-bottom:6px">'
                f'{_tnodes[0]["emoji"]} {_ttype.upper()} ({len(_tnodes)})</p>',
                unsafe_allow_html=True,
            )
            for _n in _tnodes:
                _rc = f"{_n['row_count']:,}" if _n.get("row_count") else "?"
                _measures = ", ".join(_n.get("measures", [])[:3])
                _tip = f"measures: {_measures}" if _measures else ""
                st.markdown(
                    f'<div style="background:#1e2a3a;border-left:3px solid {_tc};'
                    f'padding:6px 10px;border-radius:0 6px 6px 0;margin-bottom:5px">'
                    f'<span style="color:#e2e8f0;font-size:.8rem;font-weight:500">'
                    f'{_n["name"]}</span>'
                    f'<span style="color:#64748b;font-size:.7rem;margin-left:6px">'
                    f'{_rc} rows</span>'
                    + (f'<br><span style="color:#475569;font-size:.68rem">{_tip}</span>'
                       if _tip else "")
                    + "</div>",
                    unsafe_allow_html=True,
                )

    # Join paths summary
    _rels = _gd.get("relationships", [])
    if _rels:
        with st.expander(f"🔗 Detected Join Paths ({len(_rels)})", expanded=False):
            for _r in _rels:
                _conf = int(_r["confidence"] * 100)
                _badge = (
                    '<span style="color:#22c55e;font-size:.7rem">✓ inferred</span>'
                    if _r["source"] == "name_match" else
                    '<span style="color:#6366f1;font-size:.7rem">✓ declared</span>'
                )
                st.markdown(
                    f'`{_r["from_table"]}.{_r["from_column"]}` '
                    f'**→** `{_r["to_table"]}.{_r["to_column"]}` '
                    f'&nbsp;{_badge}&nbsp;'
                    f'<span style="color:#475569;font-size:.7rem">{_conf}%</span>',
                    unsafe_allow_html=True,
                )

    st.markdown("---")
    st.markdown("### 🗺️ Schema Relationship Map")
    st.caption(
        "Auto-generated ER diagram. Solid lines = exact column-name match. "
        "Dashed = similar name. Grey tables have no detected join path."
    )

    if not tables:
        st.info("No tables loaded.")
    else:
        # ── Build Mermaid erDiagram ────────────────────────────────
        def _m_name(s: str) -> str:
            """Safe Mermaid entity / attribute identifier (must start with a letter)."""
            name = _re2.sub(r"[^a-zA-Z0-9_]", "_", s).strip("_") or "col"
            if name[0].isdigit() or name[0] == "_":
                name = "n" + name
            return name

        def _m_type(col_type: str) -> str:
            """Single-word SQL type for Mermaid attribute declarations."""
            base = _re2.split(r"[\s(]", col_type.strip())[0]
            return _re2.sub(r"[^a-zA-Z0-9_]", "", base) or "VARCHAR"

        mer_lines = ["erDiagram"]

        for t in tables:
            ename = _m_name(t.name)
            mer_lines.append(f"    {ename} {{")
            for c in t.columns[:30]:
                atype = _m_type(c.type)
                aname = _m_name(c.name)
                markers = ""
                if c.primary_key:
                    markers = " PK"
                elif c.foreign_key:
                    markers = " FK"
                mer_lines.append(f"        {aname} {atype}{markers}")
            if len(t.columns) > 30:
                extra = len(t.columns) - 30
                mer_lines.append(f"        varchar more{extra}cols")
            mer_lines.append("    }")

        for j in jmap["joins"]:
            t1  = _m_name(j["t1"])
            t2  = _m_name(j["t2"])
            c1s = _m_name(j["c1"])
            c2s = _m_name(j["c2"])
            lbl = c1s if c1s == c2s else f"{c1s}_{c2s}"
            rel = "||--|{" if j["match"] == "exact" else "||..|{"
            mer_lines.append(f'    {t1} {rel} {t2} : "{lbl}"')

        mermaid_src = "\n".join(mer_lines)

        # Height: scale with content but never go tiny or huge
        _total_rows = sum(min(len(t.columns), 30) + 3 for t in tables)
        _height = max(220, min(1200, _total_rows * 28 + 80))

        # JSON-encode the source so special chars in names are safe in JS
        import json as _json
        import base64 as _b64
        _mer_js = _json.dumps(mermaid_src)

        _html_page = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<script src="https://cdn.jsdelivr.net/npm/mermaid@11/dist/mermaid.min.js"></script>
<style>
  html, body {{ margin: 0; padding: 8px; background: #161d26; overflow: auto; box-sizing: border-box; }}
  #diagram {{ display: flex; justify-content: center; align-items: flex-start; }}
  #diagram svg {{ max-width: 100% !important; height: auto !important; }}
  #errmsg {{ color: #f88; font-family: monospace; white-space: pre-wrap; padding: 12px; }}
</style>
</head>
<body>
<div id="diagram"></div>
<script>
  mermaid.initialize({{
    startOnLoad: false,
    theme: 'dark',
    er: {{
      layoutDirection: 'LR',
      diagramPadding: 20,
      entityPadding: 12,
      useMaxWidth: false
    }}
  }});
  mermaid.render('er1', {_mer_js}).then(function(r) {{
    var el = document.getElementById('diagram');
    el.innerHTML = r.svg;
    // Let parent know the natural rendered height so the iframe can shrink/grow
    var svg = el.querySelector('svg');
    if (svg) {{
      var h = svg.getBoundingClientRect().height || svg.viewBox.baseVal.height;
      if (h > 0) window.parent.postMessage({{ type: 'mermaid-height', height: Math.ceil(h) + 24 }}, '*');
    }}
  }}).catch(function(e) {{
    document.getElementById('diagram').innerHTML =
      '<pre id="errmsg">Mermaid render error:\\n' + e.message + '</pre>';
  }});
</script>
</body>
</html>"""

        _iframe_src = "data:text/html;base64," + _b64.b64encode(_html_page.encode()).decode()
        if hasattr(st, "iframe"):
            st.iframe(_iframe_src, height=_height)
        else:
            _stc.html(_html_page, height=_height, scrolling=True)

        with st.expander("✅ Detected Join Paths", expanded=False):
            if jmap["joins"]:
                for j in jmap["joins"]:
                    badge_color = "#22c55e" if j["match"] == "exact" else "#fb923c"
                    badge_label = j["match"]
                    st.markdown(
                        f'`{j["t1"]}.{j["c1"]}` **=** `{j["t2"]}.{j["c2"]}` &nbsp;'
                        f'<span style="color:{badge_color};font-size:0.74rem">{badge_label}</span>',
                        unsafe_allow_html=True,
                    )
            else:
                st.caption("No shared columns detected between any table pair.")

        with st.expander("🔍 Mermaid source", expanded=False):
            st.code(mermaid_src, language="text")

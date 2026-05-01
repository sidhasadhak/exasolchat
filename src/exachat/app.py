"""Streamlit app for exachat."""

from __future__ import annotations

import os
import streamlit as st
import pandas as pd
from dotenv import load_dotenv

from exachat.core import ExasolChat, QueryResult
from exachat.connection import ConnectionConfig
from exachat.llm import OllamaBackend, OpenAICompatibleBackend
from exachat.safety import RiskLevel

load_dotenv()

_DEFAULT_DUCKDB_PATH  = os.environ.get("EXACHAT_DUCKDB_PATH", "")
_DEFAULT_OLLAMA_URL   = os.environ.get("EXACHAT_OLLAMA_URL", "http://localhost:11434")
_DEFAULT_OLLAMA_MODEL = os.environ.get("EXACHAT_OLLAMA_MODEL", "qwen2.5-coder:7b")
_DEFAULT_KB_PATH      = os.environ.get("EXACHAT_KB_PATH", "")


# ── Page config ──────────────────────────────────────────────────────
st.set_page_config(
    page_title="⚡ exachat",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom styles — Claude-inspired palette ───────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

    html, body, [class*="css"] { font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif; }
    .block-container { padding-top: 1.5rem; max-width: 1100px; }
    code, pre, .stCode { font-family: 'JetBrains Mono', monospace !important; }

    [data-testid="stSidebar"] { background-color: #1a1a1a; }
    [data-testid="stSidebar"] * { color: #e8e8e8 !important; }
    [data-testid="stSidebar"] .stSelectbox label,
    [data-testid="stSidebar"] .stTextInput label,
    [data-testid="stSidebar"] .stTextArea label,
    [data-testid="stSidebar"] .stNumberInput label { color: #a0a0a0 !important; font-size: 0.8rem !important; }

    [data-testid="stChatMessage"] {
        border-radius: 12px; padding: 0.75rem 1rem; margin-bottom: 0.5rem;
    }

    .badge-safe {
        display: inline-block; padding: 2px 10px; border-radius: 20px;
        background: rgba(34, 197, 94, 0.12); color: #22c55e;
        font-size: 0.75rem; font-weight: 600; letter-spacing: 0.02em;
    }
    .badge-warn {
        display: inline-block; padding: 2px 10px; border-radius: 20px;
        background: rgba(251, 146, 60, 0.12); color: #fb923c;
        font-size: 0.75rem; font-weight: 600; letter-spacing: 0.02em;
    }
    .badge-blocked {
        display: inline-block; padding: 2px 10px; border-radius: 20px;
        background: rgba(239, 68, 68, 0.12); color: #ef4444;
        font-size: 0.75rem; font-weight: 600; letter-spacing: 0.02em;
    }

    .kb-indicator { font-size: 0.73rem; color: #9ca3af; margin-top: 6px; }

    .col-warning {
        background: rgba(251, 191, 36, 0.08); border-left: 3px solid #fbbf24;
        padding: 6px 10px; border-radius: 4px; font-size: 0.82rem;
        color: #d97706; margin-bottom: 6px;
    }

    .schema-table-name { font-weight: 600; font-size: 0.9rem; color: #f97316; }
    .schema-row-count  { font-size: 0.75rem; color: #6b7280; }

    .stButton > button[kind="primary"] { background: #f97316; border: none; }
    .stButton > button[kind="primary"]:hover { background: #ea6c0a; }
</style>
""", unsafe_allow_html=True)


# ── Session state ────────────────────────────────────────────────────
if "chat" not in st.session_state:
    st.session_state.chat = None
if "messages" not in st.session_state:
    st.session_state.messages = []
if "connected" not in st.session_state:
    st.session_state.connected = False

_VIZ_KEYWORDS = {"chart", "graph", "plot", "visuali", "diagram", "bar", "line", "pie", "scatter", "trend"}


def _wants_viz(question: str) -> bool:
    q = question.lower()
    return any(k in q for k in _VIZ_KEYWORDS)


# ── Helper functions ─────────────────────────────────────────────────

def _render_chart(result: QueryResult):
    if result.chart_obj is None:
        return False
    lib, chart = result.chart_obj
    if lib == "plotly":
        st.plotly_chart(chart, use_container_width=True)
    elif lib == "altair":
        st.altair_chart(chart, use_container_width=True)
    return True


def _render_result(r: QueryResult):
    key = hash(r.question)

    if r.error:
        st.error(r.error)
        if r.sql:
            with st.expander("🔍 Generated SQL"):
                st.code(r.sql, language="sql")
        return

    if r.column_warnings:
        for w in r.column_warnings:
            st.markdown(f'<div class="col-warning">{w}</div>', unsafe_allow_html=True)

    if r.summary:
        st.markdown(r.summary)

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

    if r.data is not None and len(r.data) > 0:
        chart_rendered = _render_chart(r)
        skip_table = chart_rendered and _wants_viz(r.question)

        if not skip_table:
            st.dataframe(
                r.data,
                use_container_width=True,
                height=min(400, 35 * len(r.data) + 50),
            )

        col_dl, col_gap = st.columns([1, 6])
        with col_dl:
            csv = r.data.to_csv(index=False)
            st.download_button(
                "📥 CSV", csv, "query_result.csv", "text/csv",
                use_container_width=True,
                key=f"dl_{key}",
            )


# ── Sidebar ──────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚡ exachat")
    st.caption("Text-to-SQL · Local LLMs · SQL Pattern KB")

    st.divider()

    # --- Connection ---
    st.markdown("#### Database")
    _conn_options = ["DuckDB", "Exasol (pyexasol)", "SQLAlchemy URL"]
    conn_type = st.selectbox("Connection type", _conn_options)

    if conn_type == "Exasol (pyexasol)":
        exa_host   = st.text_input("Host:Port", placeholder="exasoldb:8563")
        exa_user   = st.text_input("Username")
        exa_pass   = st.text_input("Password", type="password")
        exa_schema = st.text_input("Schema", placeholder="MY_SCHEMA")
    elif conn_type == "DuckDB":
        duck_path = st.text_input(
            "Database path",
            value=_DEFAULT_DUCKDB_PATH,
            placeholder="/path/to/data.duckdb or :memory:",
            help="Set EXACHAT_DUCKDB_PATH in a .env file to pre-fill.",
        )
        duck_schema = st.text_input("Schema", value="main")
    else:
        db_url = st.text_input(
            "Connection URL",
            placeholder="sqlite:///mydata.db",
        )

    st.divider()

    # --- LLM ---
    st.markdown("#### LLM Backend")
    llm_type = st.selectbox("Backend", ["Ollama", "OpenAI-compatible API"])

    if llm_type == "Ollama":
        ollama_url   = st.text_input("Ollama URL", value=_DEFAULT_OLLAMA_URL)
        ollama_model = st.text_input("Model", value=_DEFAULT_OLLAMA_MODEL)
    else:
        api_url   = st.text_input("API URL", value="http://localhost:1234/v1")
        api_model = st.text_input("Model", value="local-model")
        api_key   = st.text_input("API Key", value="not-needed", type="password")

    st.divider()

    # --- Knowledge base ---
    st.markdown("#### Knowledge Base")
    kb_path_input = st.text_input(
        "Extra KB directory (optional)",
        value=_DEFAULT_KB_PATH,
        placeholder="/path/to/your/kb/",
        help=(
            "Path to a folder of additional JSON pattern files.\n"
            "Built-in patterns are always loaded automatically.\n"
            "Set EXACHAT_KB_PATH in .env to pre-fill."
        ),
    )

    st.divider()

    # --- Access control ---
    st.markdown("#### Access Control")
    allowed_schemas_str = st.text_input(
        "Allowed schemas (comma-separated)",
        placeholder="SALES, ANALYTICS",
    )
    allowed_tables_str = st.text_input(
        "Allowed tables (comma-separated)",
        placeholder="CUSTOMERS, ORDERS, PRODUCTS",
    )

    st.divider()

    # --- Options ---
    st.markdown("#### Options")
    extra_context = st.text_area(
        "Extra context / DDL",
        placeholder="Business rules, column descriptions...",
        height=80,
    )
    max_rows  = st.number_input("Max rows", value=5000, min_value=100, max_value=50000)
    chart_lib = st.selectbox("Chart library", ["auto", "plotly", "altair"])

    st.divider()

    # --- Connect ---
    if st.button("⚡ Connect", use_container_width=True, type="primary"):
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
                else:
                    if not db_url:
                        st.error("Enter a database URL.")
                        st.stop()
                    config = ConnectionConfig.from_url(db_url)

                if llm_type == "Ollama":
                    llm = OllamaBackend(model=ollama_model, base_url=ollama_url)
                else:
                    llm = OpenAICompatibleBackend(
                        base_url=api_url, model=api_model, api_key=api_key,
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
                else:
                    schema_param = None

                chat = ExasolChat(
                    connection=config,
                    llm=llm,
                    schema=schema_param,
                    allowed_schemas=allowed_schemas,
                    allowed_tables=allowed_tables,
                    extra_context=extra_context,
                    max_rows=max_rows,
                    kb_path=kb_path_input.strip() or None,
                    chart_library=chart_lib,
                )

            st.session_state.chat = chat
            st.session_state.connected = True
            st.session_state.messages = []
            st.success(
                f"Connected! {len(chat.schema_context.tables)} tables "
                f"({chat.schema_context.dialect})"
            )
        except Exception as e:
            st.error(f"Connection failed: {e}")

    # --- Session controls ---
    if st.session_state.connected and st.session_state.chat:
        st.divider()
        col_nc, col_dc = st.columns(2)
        with col_nc:
            if st.button("🗨️ New Chat", use_container_width=True,
                         help="Keep connection, clear conversation"):
                st.session_state.chat.clear_history()
                st.session_state.messages = []
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
                st.rerun()

    # --- Schema + KB explorer ---
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

        kb_count = chat_ref.kb.count
        with st.expander(f"📖 Knowledge Base ({kb_count} patterns)", expanded=False):
            st.caption(
                f"{kb_count} SQL patterns loaded. These guide the LLM toward "
                "correct window functions, CTEs, joins, and other techniques."
            )


# ── Main area ────────────────────────────────────────────────────────
if not st.session_state.connected:
    st.markdown("## ⚡ exachat")
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
            'from exachat import ExasolChat\n\n'
            'chat = ExasolChat(\n'
            '    "duckdb:///data.duckdb"\n'
            ')\n'
            'result = chat.ask("Top 10 customers by revenue")\n'
            'print(result.data)',
            language="python",
        )
    with col2:
        st.markdown("#### DuckDB")
        st.code(
            'from exachat import ExasolChat\n\n'
            'chat = ExasolChat("./analytics.duckdb")\n\n'
            'result = chat.ask("Monthly trends")\n'
            'print(result.summary)',
            language="python",
        )
    with col3:
        st.markdown("#### CLI")
        st.code(
            '# Install\n'
            'pip install exachat\n\n'
            '# Launch\n'
            'exachat',
            language="bash",
        )
    st.stop()


# ── Chat interface ───────────────────────────────────────────────────
chat_engine: ExasolChat = st.session_state.chat

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        if msg["role"] == "user":
            st.markdown(msg["content"])
        elif "result" in msg:
            _render_result(msg["result"])
        else:
            st.markdown(msg.get("content", ""))

if question := st.chat_input("Ask a question about your data..."):
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Generating SQL..."):
            result = chat_engine.ask(question)
        _render_result(result)

    st.session_state.messages.append({"role": "assistant", "result": result})

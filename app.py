"""Streamlit app for ExasolChat."""

from __future__ import annotations

import streamlit as st
import pandas as pd

from exasolchat.core import ExasolChat, QueryResult
from exasolchat.connection import ConnectionConfig
from exasolchat.llm import OllamaBackend, OpenAICompatibleBackend
from exasolchat.rag import RAGMemory, NoopRAGMemory
from exasolchat.safety import RiskLevel


# ── Page config ──────────────────────────────────────────────────────
st.set_page_config(
    page_title="ExasolChat",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom styles ────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

    .block-container { padding-top: 1.5rem; max-width: 1200px; }

    /* Override Streamlit defaults */
    .stApp { font-family: 'DM Sans', sans-serif; }
    code, .stCode { font-family: 'JetBrains Mono', monospace !important; }

    /* Safety badges */
    .badge-safe {
        display: inline-block; padding: 2px 10px; border-radius: 12px;
        background: rgba(76, 175, 80, 0.15); color: #66bb6a;
        font-size: 0.78rem; font-weight: 600;
    }
    .badge-warn {
        display: inline-block; padding: 2px 10px; border-radius: 12px;
        background: rgba(255, 183, 77, 0.15); color: #ffa726;
        font-size: 0.78rem; font-weight: 600;
    }
    .badge-blocked {
        display: inline-block; padding: 2px 10px; border-radius: 12px;
        background: rgba(239, 83, 80, 0.15); color: #ef5350;
        font-size: 0.78rem; font-weight: 600;
    }

    /* RAG indicator */
    .rag-indicator {
        font-size: 0.75rem; color: #90a4ae; margin-top: 4px;
    }

    /* Schema card */
    .schema-table-name {
        font-weight: 600; font-size: 0.92rem;
        margin-bottom: 2px;
    }
    .schema-row-count {
        font-size: 0.78rem; color: #78909c;
    }
</style>
""", unsafe_allow_html=True)


# ── Session state ────────────────────────────────────────────────────
if "chat" not in st.session_state:
    st.session_state.chat = None
if "messages" not in st.session_state:
    st.session_state.messages = []
if "connected" not in st.session_state:
    st.session_state.connected = False


# ── Helper functions ─────────────────────────────────────────────────

def _render_chart(result: QueryResult):
    """Render chart from QueryResult."""
    if result.chart_obj is None:
        return
    lib, chart = result.chart_obj
    if lib == "plotly":
        st.plotly_chart(chart, use_container_width=True)
    elif lib == "altair":
        st.altair_chart(chart, use_container_width=True)


def _render_result(r: QueryResult):
    """Render a QueryResult in a chat message."""
    # Safety badge
    if r.safety.level == RiskLevel.SAFE:
        badge = '<span class="badge-safe">✓ Safe</span>'
    elif r.safety.level == RiskLevel.SUSPICIOUS:
        badge = f'<span class="badge-warn">⚠ {r.safety.reason}</span>'
    else:
        badge = f'<span class="badge-blocked">✕ Blocked: {r.safety.reason}</span>'

    # Error
    if r.error:
        st.error(r.error)
        if r.sql:
            with st.expander("Generated SQL"):
                st.code(r.sql, language="sql")
        return

    # Summary
    if r.summary:
        st.markdown(r.summary)

    # SQL expander with safety badge
    with st.expander("SQL  " + badge, expanded=False):
        st.code(r.sql, language="sql")
        if r.explanation:
            st.caption(r.explanation)
        # RAG indicator
        if r.rag_examples_used > 0:
            st.markdown(
                f'<div class="rag-indicator">📚 Used {r.rag_examples_used} '
                f'similar past queries as reference</div>',
                unsafe_allow_html=True,
            )

    # Data + chart
    if r.data is not None and len(r.data) > 0:
        # Chart first (more visually impactful)
        _render_chart(r)

        # Data table
        st.dataframe(
            r.data,
            use_container_width=True,
            height=min(400, 35 * len(r.data) + 50),
        )

        # Download
        col1, col2 = st.columns([1, 5])
        with col1:
            csv = r.data.to_csv(index=False)
            st.download_button(
                "📥 CSV", csv, "query_result.csv", "text/csv",
                use_container_width=True,
            )


# ── Sidebar ──────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚡ ExasolChat")
    st.caption("Text-to-SQL · Local LLMs · RAG Memory")

    st.divider()

    # --- Connection ---
    st.markdown("#### Database")
    conn_type = st.selectbox(
        "Connection type",
        ["Exasol (pyexasol)", "DuckDB", "SQLAlchemy URL"],
    )

    if conn_type == "Exasol (pyexasol)":
        exa_host = st.text_input("Host:Port", placeholder="exasoldb:8563")
        exa_user = st.text_input("Username")
        exa_pass = st.text_input("Password", type="password")
        exa_schema = st.text_input("Schema", placeholder="MY_SCHEMA")
    elif conn_type == "DuckDB":
        duck_path = st.text_input(
            "Database path",
            placeholder="/path/to/data.duckdb or :memory:",
            help=(
                "Path to a .duckdb file, or :memory: for in-memory.\n"
                "DuckDB can also query Parquet/CSV files — create views "
                "in your .duckdb file first."
            ),
        )
        duck_schema = st.text_input("Schema", value="main", help="DuckDB schema (default: main)")
    else:
        db_url = st.text_input(
            "Connection URL",
            placeholder="sqlite:///mydata.db",
            help=(
                "Examples:\n"
                "- sqlite:///path/to/db.sqlite\n"
                "- postgresql://user:pass@host:5432/db\n"
                "- mysql+pymysql://user:pass@host:3306/db"
            ),
        )

    st.divider()

    # --- LLM ---
    st.markdown("#### LLM Backend")
    llm_type = st.selectbox("Backend", ["Ollama", "OpenAI-compatible API"])

    if llm_type == "Ollama":
        ollama_url = st.text_input("Ollama URL", value="http://localhost:11434")
        ollama_model = st.text_input("Model", value="llama3.1:8b")
    else:
        api_url = st.text_input("API URL", value="http://localhost:1234/v1")
        api_model = st.text_input("Model", value="local-model")
        api_key = st.text_input("API Key", value="not-needed", type="password")

    st.divider()

    # --- Access control ---
    st.markdown("#### Access Control")
    allowed_schemas_str = st.text_input(
        "Allowed schemas (comma-separated)",
        placeholder="SALES, ANALYTICS",
        help="Leave blank to allow all schemas.",
    )
    allowed_tables_str = st.text_input(
        "Allowed tables (comma-separated)",
        placeholder="CUSTOMERS, ORDERS, PRODUCTS",
        help="Leave blank to allow all tables.",
    )

    st.divider()

    # --- Options ---
    st.markdown("#### Options")
    extra_context = st.text_area(
        "Extra context / DDL",
        placeholder="Business rules, column descriptions, custom DDL...",
        height=80,
    )
    max_rows = st.number_input("Max rows", value=5000, min_value=100, max_value=50000)
    rag_enabled = st.toggle("RAG memory (learn from queries)", value=True)
    chart_lib = st.selectbox("Chart library", ["auto", "plotly", "altair"])

    st.divider()

    # --- Connect ---
    if st.button("⚡ Connect", use_container_width=True, type="primary"):
        try:
            with st.spinner("Connecting & reading schema..."):
                # Build connection config
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

                # Build LLM
                if llm_type == "Ollama":
                    llm = OllamaBackend(model=ollama_model, base_url=ollama_url)
                else:
                    llm = OpenAICompatibleBackend(
                        base_url=api_url, model=api_model, api_key=api_key,
                    )

                # Parse access control
                allowed_schemas = None
                if allowed_schemas_str.strip():
                    allowed_schemas = [
                        s.strip() for s in allowed_schemas_str.split(",") if s.strip()
                    ]
                allowed_tables = None
                if allowed_tables_str.strip():
                    allowed_tables = [
                        t.strip() for t in allowed_tables_str.split(",") if t.strip()
                    ]

                # Determine schema param
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
                    rag_enabled=rag_enabled,
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

    # --- Schema explorer ---
    if st.session_state.connected and st.session_state.chat:
        st.divider()
        chat_ref = st.session_state.chat

        with st.expander(f"📋 Schema ({len(chat_ref.schema_context.tables)} tables)", expanded=False):
            for table in chat_ref.schema_context.tables:
                label = f'<div class="schema-table-name">{table.name}</div>'
                if table.row_count is not None:
                    label += f'<div class="schema-row-count">{table.row_count:,} rows</div>'
                st.markdown(label, unsafe_allow_html=True)
                cols_text = " · ".join(
                    f"`{c.name}` {c.type}" for c in table.columns[:8]
                )
                if len(table.columns) > 8:
                    cols_text += f" · ... +{len(table.columns) - 8} more"
                st.caption(cols_text)

        # RAG memory stats
        if rag_enabled:
            with st.expander(f"📚 RAG Memory ({chat_ref.rag.count} pairs)", expanded=False):
                if chat_ref.rag.count > 0:
                    pairs = chat_ref.rag.list_all()
                    for p in pairs[:20]:
                        st.markdown(f"**Q:** {p['question']}")
                        st.code(p["sql"], language="sql")
                    if st.button("🗑 Clear memory", use_container_width=True):
                        chat_ref.rag.clear()
                        st.rerun()
                else:
                    st.caption("No queries stored yet. Ask some questions!")

            # Manual training
            with st.expander("🎯 Train (add Q&A manually)", expanded=False):
                train_q = st.text_input("Question", key="train_q")
                train_sql = st.text_area("SQL", key="train_sql", height=80)
                if st.button("Add to memory", use_container_width=True):
                    if train_q and train_sql:
                        chat_ref.train(train_q, train_sql)
                        st.success("Added!")
                        st.rerun()


# ── Main area ────────────────────────────────────────────────────────
if not st.session_state.connected:
    st.markdown("## ⚡ ExasolChat")
    st.markdown(
        "Connect to your Exasol database (or any SQL database) in the sidebar, "
        "then ask questions in plain English."
    )
    st.info(
        "**Prerequisites:** Ollama running locally (or any OpenAI-compatible API) "
        "and a database to connect to."
    )

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("#### Python API")
        st.code(
            'from exasolchat import ExasolChat\n\n'
            'chat = ExasolChat(\n'
            '    "exa+pyexasol://user:pass@host:8563/SCHEMA"\n'
            ')\n'
            'result = chat.ask("Top 10 customers by revenue")\n'
            'print(result.data)',
            language="python",
        )
    with col2:
        st.markdown("#### DuckDB")
        st.code(
            'from exasolchat import ExasolChat\n\n'
            '# Local .duckdb file\n'
            'chat = ExasolChat("duckdb:///data.duckdb")\n\n'
            '# Or with bare path\n'
            'chat = ExasolChat("./analytics.duckdb")\n\n'
            'result = chat.ask("Monthly trends")',
            language="python",
        )
    with col3:
        st.markdown("#### CLI")
        st.code(
            '# Install\n'
            'pip install exasolchat\n\n'
            '# Launch\n'
            'exasolchat',
            language="bash",
        )
    st.stop()


# ── Chat interface ───────────────────────────────────────────────────
chat_engine: ExasolChat = st.session_state.chat

# Render history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        if msg["role"] == "user":
            st.markdown(msg["content"])
        elif "result" in msg:
            _render_result(msg["result"])
        else:
            st.markdown(msg.get("content", ""))

# Input
if question := st.chat_input("Ask a question about your data..."):
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Generating SQL..."):
            result = chat_engine.ask(question)
        _render_result(result)

    st.session_state.messages.append({"role": "assistant", "result": result})

# ⚡ ExasolChat

**Ask questions about your Exasol or DuckDB database in plain English. Get SQL, data, and interactive charts.**

Local LLMs only. No data leaves your machine. RAG memory learns from your queries.

---

## Install

```bash
pip install exasolchat
```

## How To: Zero to Querying in 5 Minutes

This section gets you from nothing to asking your database questions in plain English. Pick your database, follow the steps, done.

### Step 1: Get a local LLM running

You need a local LLM to generate SQL. Ollama is the fastest way:

```bash
# Install Ollama (macOS/Linux)
curl -fsSL https://ollama.com/install.sh | sh

# Pull a model (pick one)
ollama pull llama3.1:8b        # Fast, decent quality (recommended to start)
ollama pull qwen2.5-coder:14b  # Better SQL quality, slower
ollama pull codellama:13b       # Good middle ground

# Verify it's running
curl http://localhost:11434/api/tags
```

Using LM Studio or vLLM instead? That works too — just choose "OpenAI-compatible API" in the UI or pass an `OpenAICompatibleBackend` in code.

### Step 2: Connect to your database

**Option A: Use the Streamlit UI (easiest)**

```bash
exasolchat
```

A browser window opens. In the sidebar:
1. Pick your database type (Exasol, DuckDB, or SQLAlchemy URL)
2. Fill in credentials
3. Pick your LLM model
4. Click **⚡ Connect**
5. Start typing questions in the chat box

**Option B: Use the Python API (scriptable)**

Pick your database below and copy-paste:

```python
from exasolchat import ExasolChat

# ─── Exasol ──────────────────────────────────────────────
chat = ExasolChat("exa+pyexasol://user:pass@host:8563/MY_SCHEMA")

# ─── DuckDB (local file) ────────────────────────────────
chat = ExasolChat("duckdb:///path/to/analytics.duckdb")
# or just pass the file path directly:
chat = ExasolChat("./my_data.duckdb")

# ─── DuckDB (in-memory) ─────────────────────────────────
chat = ExasolChat("duckdb://:memory:")

# ─── PostgreSQL ─────────────────────────────────────────
chat = ExasolChat("postgresql://user:pass@localhost:5432/mydb")

# ─── SQLite ─────────────────────────────────────────────
chat = ExasolChat("sqlite:///local.db")

# ─── MySQL ──────────────────────────────────────────────
chat = ExasolChat("mysql+pymysql://user:pass@host:3306/db")
```

### Step 3: Ask questions

```python
result = chat.ask("What are the top 10 customers by total spend?")

print(result.summary)      # "The top customer is Acme Corp with €2.3M..."
print(result.sql)           # SELECT customer_name, SUM(total) ...
print(result.data)          # pandas DataFrame with the results
print(result.chart_config)  # {"chart_type": "bar", "x": "customer_name", ...}
```

That's it. The system auto-introspects your schema, generates SQL, validates it for safety, runs it read-only, and suggests a chart.

### Step 4: Make it smarter (optional)

The RAG memory learns from successful queries. But you can also teach it your company's patterns:

```python
# Teach it a query pattern it might not figure out on its own
chat.train(
    "quarterly revenue by region",
    """SELECT 
        region,
        DATE_TRUNC('quarter', order_date) AS quarter,
        SUM(amount) AS revenue
    FROM sales.orders
    JOIN sales.customers ON orders.customer_id = customers.id
    GROUP BY 1, 2
    ORDER BY 2, 3 DESC"""
)

# Now when you ask similar questions, it'll use this as a reference
result = chat.ask("Show me revenue by region for each quarter")
```

### Step 5: Lock it down (recommended for shared environments)

```python
chat = ExasolChat(
    "exa+pyexasol://readonly_user:pass@host:8563/PROD",
    
    # Only allow querying these schemas
    allowed_schemas=["SALES", "ANALYTICS"],
    
    # Only allow these specific tables
    allowed_tables=["CUSTOMERS", "ORDERS", "PRODUCTS", "REGIONS"],
    
    # Add business context so the LLM understands your data
    extra_context="""
        - revenue columns are in EUR
        - fiscal year starts April 1
        - customer_tier: 'gold' = annual spend > €50k
        - ORDERS.status: 'active', 'cancelled', 'refunded'
    """,
)
```

### Common recipes

**Load a DuckDB file and explore it interactively:**
```bash
exasolchat
# → Select "DuckDB" → Enter path → Connect → Ask away
```

**Script it for a report:**
```python
from exasolchat import ExasolChat

with ExasolChat("duckdb:///sales.duckdb") as chat:
    monthly = chat.ask("Monthly revenue for the last 12 months")
    top_products = chat.ask("Top 5 products by units sold this quarter")
    
    # Export to CSV
    monthly.data.to_csv("monthly_revenue.csv", index=False)
    top_products.data.to_csv("top_products.csv", index=False)
```

**Use a different LLM backend:**
```python
from exasolchat import ExasolChat
from exasolchat.llm import OpenAICompatibleBackend

# LM Studio running on port 1234
llm = OpenAICompatibleBackend(
    base_url="http://localhost:1234/v1",
    model="qwen2.5-coder-14b",
)
chat = ExasolChat("./data.duckdb", llm=llm)
```

**Check what the system knows about your schema:**
```python
chat = ExasolChat("duckdb:///data.duckdb")
print(chat.schema_prompt)  # Prints the full schema context sent to the LLM
```

## Prerequisites

- **Ollama** running locally (or any OpenAI-compatible API — LM Studio, vLLM, LocalAI)
- An **ExasolDB** instance, **DuckDB** file, or any SQLAlchemy-supported database

## Quick Start

### Option 1: Streamlit UI

```bash
exasolchat
```

Fill in your connection details in the sidebar and start asking questions.

### Option 2: Python API

```python
from exasolchat import ExasolChat

# Exasol (pyexasol — native, fast)
chat = ExasolChat("exa+pyexasol://user:pass@host:8563/MY_SCHEMA")

# Or with explicit config
from exasolchat.connection import ConnectionConfig
config = ConnectionConfig.exasol("host:8563", "user", "pass", "MY_SCHEMA")
chat = ExasolChat(config)

# Ask questions
result = chat.ask("What are the top 10 customers by revenue?")
print(result.sql)         # Generated SQL
print(result.data)        # pandas DataFrame
print(result.summary)     # Natural language summary
print(result.chart_config)  # Suggested chart

# Manual training (teach it your patterns)
chat.train(
    "monthly revenue trend",
    "SELECT DATE_TRUNC('month', order_date) AS month, SUM(total) AS revenue FROM orders GROUP BY 1 ORDER BY 1"
)
```

### Non-Exasol databases

```python
# DuckDB (native driver — fast, supports Parquet/CSV)
chat = ExasolChat("duckdb:///path/to/data.duckdb")
chat = ExasolChat("./analytics.duckdb")  # bare path auto-detected

# DuckDB in-memory
chat = ExasolChat("duckdb://:memory:")

# SQLAlchemy fallback — works with anything
chat = ExasolChat("postgresql://user:pass@localhost/mydb")
chat = ExasolChat("sqlite:///local.db")
chat = ExasolChat("mysql+pymysql://user:pass@host/db")
```

## Architecture

```
Question ──► RAG Retrieval ──► LLM Prompt ──► SQL Generation
                                                    │
                                              Safety Check ◄── Schema Allowlist
                                                    │
                                              Query Execution (read-only)
                                                    │
                                         Summary + Chart + DataFrame
```

### Modules

| Module | Purpose |
|--------|---------|
| `safety.py` | SQL validation — allowlist-only (SELECT/WITH), pattern matching for Exasol + DuckDB + general, schema access control |
| `schema.py` | Auto-introspection — pyexasol (Exasol), native duckdb, or SQLAlchemy (everything else) |
| `llm.py` | LLM backends — Ollama + OpenAI-compatible, with RAG-augmented prompts and dialect-aware hints |
| `rag.py` | ChromaDB-backed semantic memory — stores successful Q&A pairs, retrieves similar ones |
| `connection.py` | Connection management — pyexasol, duckdb native, SQLAlchemy fallback |
| `charts.py` | Auto-charting — Plotly + Altair, picks best library per data shape |
| `core.py` | Engine — ties everything into a single `.ask()` call |
| `app.py` | Streamlit UI — chat interface, schema explorer, RAG management |

## Safety Model

This is where we learned from Vanna's mistakes:

- **Allowlist-only**: Only `SELECT` and `WITH` (CTE) pass through. Everything else is blocked.
- **No `exec()` or `eval()`**: LLM output is NEVER executed as Python code. Anywhere.
- **Pattern matching**: Blocks DDL, DML, `EXEC`/`CALL`, `EXPORT`/`IMPORT`, `COPY`, `ATTACH`/`DETACH`, `INSTALL`/`LOAD`, `PRAGMA`, `read_csv`/`read_parquet`/`read_json`, `pg_sleep`, `BENCHMARK`, statement stacking, `SET` commands, and Exasol scripting.
- **Schema access control**: Configure which schemas and tables the LLM is allowed to reference.
- **Read-only enforcement**: Uses `SET TRANSACTION READ ONLY` where supported.
- **Suspicious detection**: Flags `UNION SELECT`, tautology injections, system table access — executes with a visible warning.

### Defense in depth

Use a **read-only database user** in production. The safety layer is defense-in-depth, not a replacement for proper DB permissions.

## RAG Memory

ChromaDB stores successful question→SQL pairs locally. When you ask a new question, similar past queries are retrieved and injected into the LLM prompt as few-shot examples.

```python
# RAG is on by default. Turn it off:
chat = ExasolChat("...", rag_enabled=False)

# Manual training:
chat.train("monthly revenue", "SELECT DATE_TRUNC('month', ...) ...")

# Clear memory:
chat.rag.clear()

# Check stored pairs:
print(chat.rag.count)
print(chat.rag.list_all())
```

Memory persists at `~/.exasolchat/rag/` by default.

## Configuration

```python
from exasolchat import ExasolChat
from exasolchat.llm import OllamaBackend

chat = ExasolChat(
    connection="exa+pyexasol://user:pass@host:8563/SCHEMA",
    llm=OllamaBackend(model="codellama:13b"),

    # Schema introspection
    schema="MY_SCHEMA",
    include_tables=["ORDERS", "CUSTOMERS"],
    exclude_tables=["INTERNAL_LOGS"],

    # Access control
    allowed_schemas=["SALES", "ANALYTICS"],
    allowed_tables=["CUSTOMERS", "ORDERS", "PRODUCTS"],

    # Context
    extra_context="revenue is in EUR. fiscal year starts April 1.",

    # Limits
    max_rows=10000,

    # RAG
    rag_enabled=True,

    # Charts
    chart_library="auto",  # "plotly", "altair", or "auto"
)
```

## LLM Recommendations

For Exasol's SQL dialect, models with strong SQL training work best:

| Model | Size | Quality | Speed |
|-------|------|---------|-------|
| `codellama:13b` | 13B | Good | Medium |
| `deepseek-coder-v2:16b` | 16B | Very good | Medium |
| `llama3.1:8b` | 8B | Decent | Fast |
| `llama3.1:70b` | 70B | Excellent | Slow |
| `qwen2.5-coder:14b` | 14B | Very good | Medium |

## Limitations (honest)

- **SQL accuracy = LLM quality.** Smaller models produce worse SQL. 13B+ recommended.
- **Exasol SQL dialect.** Not all models know Exasol's quirks. RAG training helps.
- **No multi-turn SQL refinement** (yet). Each `.ask()` is independent.
- **Safety layer is regex-based.** It catches known patterns but isn't a SQL parser. Use a read-only DB user.
- **Charts are LLM-suggested.** They're usually right but not always.

## License

MIT

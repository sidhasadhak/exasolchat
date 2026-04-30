# ⚡ exachat

**Ask your database anything — in plain English. Get SQL, data, and interactive charts.**

Local LLMs only. No data leaves your machine. Works with DuckDB, Exasol, PostgreSQL, MySQL, SQLite, and anything SQLAlchemy supports.

---

## Install

```bash
pip install exachat
```

## How To: Zero to Querying in 5 Minutes

### Step 1: Get a local LLM running

```bash
# Install Ollama (macOS/Linux)
curl -fsSL https://ollama.com/install.sh | sh

# Pull a model — qwen2.5-coder:7b is the recommended starting point
ollama pull qwen2.5-coder:7b   # ✅ Best quality/speed balance for SQL
ollama pull qwen2.5-coder:14b  # Better quality, slower
ollama pull deepseek-coder-v2:16b  # Excellent for complex queries

# Verify Ollama is running
curl http://localhost:11434/api/tags
```

Using LM Studio or vLLM? That works too — choose **"OpenAI-compatible API"** in the UI.

### Step 2: Connect to your database

**Option A: Streamlit UI (easiest)**

```bash
exachat
```

A browser window opens at `http://localhost:8501`. In the sidebar:
1. Pick your database type — **DuckDB is the default**
2. Fill in credentials or file path
3. Pick your LLM model
4. Click **⚡ Connect**
5. Start asking questions

**Option B: Python API**

```python
from exachat import ExasolChat

# ─── DuckDB (local file) ────────────────────────────────
chat = ExasolChat("duckdb:///path/to/analytics.duckdb")
chat = ExasolChat("./my_data.duckdb")  # bare path works too

# ─── DuckDB (in-memory) ─────────────────────────────────
chat = ExasolChat("duckdb://:memory:")

# ─── Exasol ──────────────────────────────────────────────
chat = ExasolChat("exa+pyexasol://user:pass@host:8563/MY_SCHEMA")

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

print(result.summary)      # "The top customer is Acme Corp with $2.3M..."
print(result.sql)          # SELECT customer_name, SUM(total) AS total_spend ...
print(result.data)         # pandas DataFrame
print(result.chart_config) # {"chart_type": "bar", "x": "customer_name", ...}
```

The system auto-introspects your schema, generates SQL, validates it for safety, runs it read-only, and suggests a chart.

### Step 4: Give feedback to make it smarter

In the UI, every answer has **👍 / 👎** buttons. Thumbs up saves the question→SQL pair to RAG memory — future similar questions will use it as a reference.

You can also train it manually:

```python
chat.train(
    "quarterly revenue by region",
    """SELECT
        region,
        date_trunc('quarter', order_date) AS quarter,
        SUM(amount) AS revenue
    FROM sales.orders
    JOIN sales.customers ON orders.customer_id = customers.id
    GROUP BY ALL
    ORDER BY quarter, revenue DESC"""
)
```

### Step 5: Pre-fill defaults with a .env file

Create a `.env` file in your working directory (it's gitignored):

```bash
EXACHAT_DUCKDB_PATH=/path/to/your/database.duckdb
EXACHAT_OLLAMA_URL=http://localhost:11434
EXACHAT_OLLAMA_MODEL=qwen2.5-coder:7b
```

The UI will pre-fill these values on launch.

### Step 6: Lock it down (recommended for shared environments)

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

---

## Common Recipes

**Explore a DuckDB file interactively:**
```bash
exachat
# → Select DuckDB → Enter path → Connect → Ask away
```

**Script it for a report:**
```python
from exachat import ExasolChat

with ExasolChat("duckdb:///sales.duckdb") as chat:
    monthly = chat.ask("Monthly revenue for the last 12 months")
    top_products = chat.ask("Top 5 products by units sold this quarter")

    monthly.data.to_csv("monthly_revenue.csv", index=False)
    top_products.data.to_csv("top_products.csv", index=False)
```

**Use a different LLM backend (LM Studio, vLLM, etc.):**
```python
from exachat import ExasolChat
from exachat.llm import OpenAICompatibleBackend

llm = OpenAICompatibleBackend(
    base_url="http://localhost:1234/v1",
    model="qwen2.5-coder-14b",
)
chat = ExasolChat("./data.duckdb", llm=llm)
```

**Inspect what the LLM sees:**
```python
chat = ExasolChat("duckdb:///data.duckdb")
print(chat.schema_prompt)  # Full schema context sent to the LLM, including detected date formats
```

---

## Architecture

```
Question ──► RAG Retrieval ──► LLM Prompt ──► SQL Generation
                                                    │
                                        DuckDB dialect hints
                                        Date format detection
                                        Column name mapping
                                                    │
                                              Safety Check ◄── Schema Allowlist
                                                    │
                                        Query Execution (read-only)
                                                    │
                                   Summary + Chart + DataFrame + Feedback
```

### Modules

| Module | Purpose |
|--------|---------|
| `safety.py` | SQL validation — allowlist-only (SELECT/WITH), blocks DDL, DML, DuckDB/Exasol-specific attacks, injection patterns, statement stacking |
| `schema.py` | Auto-introspection — pyexasol (Exasol), native duckdb, SQLAlchemy (everything else). Detects date formats from sample values. |
| `llm.py` | LLM backends — Ollama + OpenAI-compatible. Full DuckDB dialect hints, RAG-augmented prompts, column name mapping rules. |
| `rag.py` | Offline semantic memory — bag-of-words ChromaDB store, no model downloads required. Stores successful Q→SQL pairs, retrieves similar ones. |
| `connection.py` | Connection management — pyexasol, duckdb native (read-only), SQLAlchemy fallback |
| `charts.py` | Auto-charting — Plotly + Altair, renders bar/line/area/scatter/pie/heatmap |
| `core.py` | Engine — orchestrates the full ask() pipeline. Column disambiguation warnings. |
| `app.py` | Streamlit UI — chat interface, 👍/👎 feedback, schema explorer, RAG memory browser |

---

## Safety Model

Built to avoid the mistakes common in text-to-SQL tools:

- **Allowlist-only**: Only `SELECT` and `WITH` (CTE) pass through. Everything else is blocked.
- **No `exec()` or `eval()`**: LLM output is **never** executed as Python code. Anywhere.
- **Pattern matching**: Blocks DDL, DML, `EXEC`/`CALL`, `EXPORT`/`IMPORT`, `COPY`, `ATTACH`/`DETACH`, `INSTALL`/`LOAD`, `PRAGMA`, `read_csv`/`read_parquet`/`read_json`, `pg_sleep`, `BENCHMARK`, statement stacking, and `SET` commands.
- **Schema access control**: Configure which schemas and tables the LLM may reference.
- **Read-only enforcement**: DuckDB files always opened `read_only=True`. SQLAlchemy uses `SET TRANSACTION READ ONLY` where supported.
- **Suspicious detection**: Flags `UNION SELECT`, tautology injections, system table access — shows a visible warning but still executes.
- **Column disambiguation**: Warns when the LLM uses a column name that fuzzy-matches but doesn't exactly match the schema (e.g. `order_date` vs `"Order Date"`).

> Use a **read-only database user** in production. The safety layer is defence-in-depth, not a replacement for proper DB permissions.

---

## RAG Memory

Successful question→SQL pairs are stored locally in ChromaDB and retrieved for similar future questions — injected as few-shot examples into the LLM prompt. No model download required (uses a lightweight offline embedding).

```python
# RAG is on by default. Turn it off:
chat = ExasolChat("...", rag_enabled=False)

# Seed it with your own patterns:
chat.train("monthly revenue", "SELECT date_trunc('month', order_date) AS month, SUM(total) FROM orders GROUP BY 1 ORDER BY 1")

# Clear memory:
chat.rag.clear()

# Inspect stored pairs:
print(chat.rag.count)
print(chat.rag.list_all())
```

Memory persists at `~/.exachat/rag/` by default.

---

## Configuration Reference

```python
from exachat import ExasolChat
from exachat.llm import OllamaBackend

chat = ExasolChat(
    connection="duckdb:///sales.duckdb",
    llm=OllamaBackend(model="qwen2.5-coder:7b"),

    # Schema filtering
    schema="main",
    include_tables=["orders", "customers"],
    exclude_tables=["internal_logs"],

    # Access control
    allowed_schemas=["SALES", "ANALYTICS"],
    allowed_tables=["CUSTOMERS", "ORDERS", "PRODUCTS"],

    # Business context
    extra_context="revenue is in EUR. fiscal year starts April 1.",

    # Limits
    max_rows=10000,

    # RAG
    rag_enabled=True,

    # Charts
    chart_library="auto",  # "plotly", "altair", or "auto"
)
```

---

## LLM Recommendations

| Model | Pull command | Quality | Speed | Notes |
|-------|-------------|---------|-------|-------|
| `qwen2.5-coder:7b` | `ollama pull qwen2.5-coder:7b` | ⭐⭐⭐⭐ | Fast | **Recommended default** |
| `qwen2.5-coder:14b` | `ollama pull qwen2.5-coder:14b` | ⭐⭐⭐⭐⭐ | Medium | Best quality/speed tradeoff |
| `deepseek-coder-v2:16b` | `ollama pull deepseek-coder-v2:16b` | ⭐⭐⭐⭐⭐ | Medium | Excellent for complex joins |
| `sqlcoder:7b` | `ollama pull sqlcoder:7b` | ⭐⭐⭐⭐ | Fast | Fine-tuned purely for SQL |
| `llama3.1:8b` | `ollama pull llama3.1:8b` | ⭐⭐⭐ | Fast | Good general-purpose fallback |

---

## Limitations (honest)

- **SQL accuracy = LLM quality.** Smaller models produce worse SQL. 7B+ recommended; 14B+ for complex schemas.
- **Safety layer is regex-based.** It catches known patterns but is not a full SQL parser. Always use a read-only DB user.
- **No multi-turn SQL refinement** (yet). Each `.ask()` is independent.
- **Charts are LLM-suggested.** They're usually right but not always.
- **RAG similarity is bag-of-words.** Works well for SQL Q&A; not as precise as embedding models.

---

## License

MIT

"""Visual query builder — Streamlit component for the 📊 Build tab.

Compact dropdown-based field picker (no long stacked lists), type-aware
routing (numeric → measures, text/date → dimensions), and auto-scroll to
results after Run.
"""

from __future__ import annotations

from typing import Optional, TYPE_CHECKING

import streamlit as st
import streamlit.components.v1 as components

from exachat.builder import AGGREGATIONS, FILTER_OPS, QueryBuilder, _is_numeric
from exachat.safety import RiskLevel, validate_sql

if TYPE_CHECKING:
    from exachat.core import ExasolChat
    from exachat.metrics import MetricsCatalog


# ── Session state ─────────────────────────────────────────────────────

def _init_builder(first_table: str) -> None:
    if "builder" not in st.session_state:
        st.session_state.builder = _empty_config(first_table)


def _empty_config(table: str) -> dict:
    return {
        "table":        table,
        "dimensions":   [],
        "measures":     [],
        "metric_names": [],
        "filters":      [],
        "sort_field":   "",
        "sort_dir":     "DESC",
        "limit":        500,
    }


# ── Main entry point ──────────────────────────────────────────────────

def render_builder(
    chat: "ExasolChat",
    qb: QueryBuilder,
    metrics_catalog: Optional["MetricsCatalog"] = None,
) -> None:
    from exachat.charts import auto_chart

    tables = qb.table_names()
    if not tables:
        st.info("No tables found in schema.")
        return

    _init_builder(tables[0])
    cfg = st.session_state.builder

    cols     = qb.columns_for(cfg["table"])
    col_names = [c.name for c in cols]
    col_map   = {c.name: c for c in cols}
    metrics   = metrics_catalog.all() if metrics_catalog else []

    # Type-segregated column lists
    num_cols  = [c.name for c in cols if _is_numeric(c.type)]
    text_cols = [c.name for c in cols if not _is_numeric(c.type)]

    active_dims     = set(cfg["dimensions"])
    active_measures = {m["field"] for m in cfg["measures"]}
    active_metrics  = set(cfg["metric_names"])

    # ── Row 1: Table + Add-field dropdowns + Run ──────────────────
    r1c1, r1c2, r1c3, r1c4, r1c5 = st.columns([2, 2, 2, 2, 1])

    with r1c1:
        selected_table = st.selectbox(
            "Table", tables,
            index=tables.index(cfg["table"]) if cfg["table"] in tables else 0,
            key="b_table",
        )
        if selected_table != cfg["table"]:
            st.session_state.builder = _empty_config(selected_table)
            st.session_state.pop("builder_result", None)
            cfg = st.session_state.builder
            st.rerun()

    with r1c2:
        unused_text = ["— add dimension —"] + [c for c in text_cols if c not in active_dims]
        add_dim = st.selectbox("Dimension", unused_text, key="b_pick_dim")

    with r1c3:
        # All columns are valid measures (COUNT DISTINCT works on VARCHAR too)
        unused_all = ["— add measure —"] + [c for c in col_names if c not in active_measures]
        add_meas = st.selectbox("Measure", unused_all, key="b_pick_meas")

    with r1c4:
        if metrics:
            unused_met = ["— add metric —"] + [m["name"] for m in metrics if m["name"] not in active_metrics]
            add_met = st.selectbox("Metric", unused_met, key="b_pick_met")
        else:
            add_met = None
            st.empty()

    with r1c5:
        st.markdown("<div style='padding-top:1.55rem'></div>", unsafe_allow_html=True)
        run = st.button("▶ Run", type="primary", use_container_width=True, key="b_run")

    # Apply quick-adds (triggers rerun so dropdowns reset to placeholder)
    if add_dim and add_dim != "— add dimension —":
        cfg["dimensions"].append(add_dim)
        st.rerun()
    if add_meas and add_meas != "— add measure —":
        cfg["measures"].append({"field": add_meas, "agg": "SUM", "alias": f"Total {add_meas}"})
        st.rerun()
    if add_met and add_met not in (None, "— add metric —"):
        cfg["metric_names"].append(add_met)
        st.rerun()

    # ── Row 2: Active-field configuration (compact expander) ──────
    with st.expander("⚙️ Configure fields & filters", expanded=True):

        # DIMENSIONS ────────────────────────────────────────────────
        st.markdown(
            "**Dimensions** <span style='color:#6b7280;font-size:0.78rem'>— Group By (text / date)</span>",
            unsafe_allow_html=True,
        )
        if cfg["dimensions"]:
            for i, dim in enumerate(cfg["dimensions"]):
                _dc1, _dc2, _dc3, _dc4 = st.columns([5, 1, 1, 1])
                with _dc1:
                    st.markdown(f"`{dim}`")
                with _dc2:
                    if i > 0 and st.button("↑", key=f"dup_{i}", help="Move up"):
                        cfg["dimensions"][i - 1], cfg["dimensions"][i] = (
                            cfg["dimensions"][i], cfg["dimensions"][i - 1]
                        )
                        st.rerun()
                with _dc3:
                    if i < len(cfg["dimensions"]) - 1 and st.button("↓", key=f"ddn_{i}", help="Move down"):
                        cfg["dimensions"][i], cfg["dimensions"][i + 1] = (
                            cfg["dimensions"][i + 1], cfg["dimensions"][i]
                        )
                        st.rerun()
                with _dc4:
                    if st.button("✕", key=f"rdim_{i}", help="Remove"):
                        cfg["dimensions"].pop(i)
                        st.rerun()
        else:
            st.caption("None — pick a dimension above.")

        # MEASURES ──────────────────────────────────────────────────
        if cfg["measures"] or cfg["metric_names"]:
            st.markdown(
                "**Measures** <span style='color:#6b7280;font-size:0.78rem'>— Aggregated values (all column types)</span>",
                unsafe_allow_html=True,
            )
            for i, m in enumerate(cfg["measures"]):
                mc1, mc2, mc3, mc4 = st.columns([3, 2, 3, 1])
                with mc1:
                    # All columns available — COUNT DISTINCT works on any type
                    f_idx = col_names.index(m["field"]) if m["field"] in col_names else 0
                    cfg["measures"][i]["field"] = st.selectbox(
                        "Field", col_names, index=f_idx,
                        key=f"mf_{i}", label_visibility="collapsed",
                    )
                with mc2:
                    a_idx = AGGREGATIONS.index(m["agg"]) if m["agg"] in AGGREGATIONS else 0
                    cfg["measures"][i]["agg"] = st.selectbox(
                        "Agg", AGGREGATIONS, index=a_idx,
                        key=f"ma_{i}", label_visibility="collapsed",
                    )
                with mc3:
                    cfg["measures"][i]["alias"] = st.text_input(
                        "Alias", value=m["alias"],
                        key=f"mal_{i}", label_visibility="collapsed",
                    )
                with mc4:
                    if st.button("✕", key=f"rm_{i}"):
                        cfg["measures"].pop(i)
                        st.rerun()

            for mn in list(cfg["metric_names"]):
                t1, t2 = st.columns([8, 1])
                with t1:
                    st.markdown(
                        f'<span style="color:#f97316;font-size:0.82rem">📐 {mn}</span>',
                        unsafe_allow_html=True,
                    )
                with t2:
                    if st.button("✕", key=f"rmm_{mn}"):
                        cfg["metric_names"].remove(mn)
                        st.rerun()

        # FILTERS ────────────────────────────────────────────────────
        if cfg["filters"]:
            st.markdown("**Filters**")
        for i, f in enumerate(cfg["filters"]):
            fc1, fc2, fc3, fc4 = st.columns([3, 2, 3, 1])
            with fc1:
                ff_idx = col_names.index(f["field"]) if f["field"] in col_names else 0
                cfg["filters"][i]["field"] = st.selectbox(
                    "Field", col_names, index=ff_idx,
                    key=f"ff_{i}", label_visibility="collapsed",
                )
            with fc2:
                op_idx = FILTER_OPS.index(f["op"]) if f["op"] in FILTER_OPS else 0
                cfg["filters"][i]["op"] = st.selectbox(
                    "Op", FILTER_OPS, index=op_idx,
                    key=f"fo_{i}", label_visibility="collapsed",
                )
            with fc3:
                if f["op"] not in ("IS NULL", "IS NOT NULL"):
                    cfg["filters"][i]["value"] = st.text_input(
                        "Value", value=f.get("value", ""),
                        key=f"fv_{i}", label_visibility="collapsed",
                    )
                else:
                    st.empty()
            with fc4:
                if st.button("✕", key=f"rf_{i}"):
                    cfg["filters"].pop(i)
                    st.rerun()

        if st.button("＋ Add filter", key="b_add_f"):
            cfg["filters"].append({
                "field": col_names[0] if col_names else "",
                "op": "=", "value": "",
            })
            st.rerun()

        # SORT + LIMIT ───────────────────────────────────────────────
        st.divider()
        sort_options = ["—"] + cfg["dimensions"] + [m["alias"] for m in cfg["measures"]] + cfg["metric_names"]
        sc1, sc2, sc3 = st.columns([3, 2, 2])
        with sc1:
            sf_idx = sort_options.index(cfg["sort_field"]) if cfg["sort_field"] in sort_options else 0
            chosen = st.selectbox("Sort by", sort_options, index=sf_idx, key="b_sort")
            cfg["sort_field"] = "" if chosen == "—" else chosen
        with sc2:
            cfg["sort_dir"] = st.selectbox("Direction", ["DESC", "ASC"], key="b_sort_dir")
        with sc3:
            cfg["limit"] = st.number_input("Limit", value=cfg["limit"], min_value=1, max_value=50000, key="b_limit")

    # ── Results anchor (scroll target) ────────────────────────────
    st.markdown('<div id="builder-results"></div>', unsafe_allow_html=True)
    st.divider()

    # ── Execute ───────────────────────────────────────────────────
    if run:
        sql = qb.build_sql(cfg, metrics_catalog)
        exec_sql = chat.schema_context.denormalize_sql(sql)
        try:
            verdict = validate_sql(
                exec_sql,
                allowed_schemas=getattr(chat, "_allowed_schemas", None),
                allowed_tables=getattr(chat, "_allowed_tables", None),
            )
            if verdict.level == RiskLevel.BLOCKED:
                st.error(f"Blocked: {verdict.reason}")
            else:
                df = chat._db.execute_query(exec_sql, cfg["limit"])
                st.session_state.builder_result = {"sql": exec_sql, "df": df}
                st.session_state.builder_scroll = True
        except Exception as e:
            st.error(f"Query failed: {e}")
            st.code(exec_sql, language="sql")
            st.session_state.pop("builder_result", None)

    # Auto-scroll to results after Run
    if st.session_state.pop("builder_scroll", False):
        components.html(
            """<script>
                window.parent.document
                    .getElementById('builder-results')
                    ?.scrollIntoView({behavior:'smooth', block:'start'});
            </script>""",
            height=0,
        )

    # ── Render results ────────────────────────────────────────────
    res = st.session_state.get("builder_result")
    if res:
        with st.expander("🔍 Generated SQL", expanded=False):
            st.code(res["sql"], language="sql")

        df = res["df"]
        if len(df) > 0:
            num_cols_b = df.select_dtypes(include="number").columns.tolist()
            dim_cols_b = df.select_dtypes(exclude="number").columns.tolist()
            all_cols_b = list(df.columns)

            # ── Interactive viz controls ──────────────────────────
            bvc1, bvc2, bvc3, bvc4 = st.columns([2, 2, 3, 3])
            with bvc1:
                b_viz = st.selectbox(
                    "Chart type",
                    ["auto", "bar", "line", "area", "scatter", "pie", "table only"],
                    key="b_viz_type",
                    help="Chart type for results",
                )
            with bvc2:
                _xax_opts = dim_cols_b or all_cols_b
                if _xax_opts:
                    b_xax = st.selectbox(
                        "Dimension (X axis)",
                        _xax_opts,
                        key="b_xax",
                        help="Column to use on the X axis",
                    )
                else:
                    b_xax = None
            with bvc3:
                _meas_opts = num_cols_b or all_cols_b
                if len(_meas_opts) > 1:
                    b_measures = st.multiselect(
                        "Measures (left Y)",
                        _meas_opts,
                        default=_meas_opts,
                        key="b_measures",
                        help="Measures on the left Y axis",
                        placeholder="Select measures",
                    )
                else:
                    b_measures = _meas_opts
            with bvc4:
                b_y2: list = []
                if len(_meas_opts) > 1 and b_measures:
                    b_y2 = st.multiselect(
                        "Second axis (right Y)",
                        b_measures,
                        default=[],
                        key="b_y2",
                        help="Move these measures to the right Y axis",
                        placeholder="Select for dual axis",
                    )

            # ── Chart render ──────────────────────────────────────
            _b_y1 = [c for c in b_measures if c not in b_y2]
            if b_viz != "table only" and (_b_y1 or b_y2) and b_xax:
                try:
                    from exachat.app import _build_chart_figure
                    _ct = b_viz if b_viz != "auto" else "bar"
                    _fig = _build_chart_figure(df, b_xax, _b_y1, b_y2, _ct)
                    st.plotly_chart(_fig, use_container_width=True)
                except Exception:
                    pass

            st.dataframe(df, use_container_width=True, height=min(400, 35 * len(df) + 50))

            c_dl, _ = st.columns([1, 5])
            with c_dl:
                st.download_button(
                    "📥 CSV", df.to_csv(index=False),
                    "builder_result.csv", "text/csv",
                    key="b_dl",
                )
        else:
            st.info("Query returned no rows.")


# ── Metrics catalog tab ───────────────────────────────────────────────

def render_metrics_tab(metrics_catalog: Optional["MetricsCatalog"]) -> None:
    """Render the 📐 Metrics tab — browse, add, and remove metrics."""
    if metrics_catalog is None:
        st.info(
            "No metrics catalog loaded. Connect to a database first, "
            "or set **EXACHAT_METRICS_PATH** in your .env file."
        )
        return

    st.markdown(f"### 📐 Metrics Catalog ({metrics_catalog.count} metrics)")

    all_metrics = metrics_catalog.all()
    if all_metrics:
        for m in all_metrics:
            with st.expander(f"**{m['name']}** — {m.get('description', '')}", expanded=False):
                st.code(m["sql"], language="sql")
                c1, c2 = st.columns(2)
                with c1:
                    if m.get("dimensions"):
                        st.caption(f"📏 Dimensions: {', '.join(m['dimensions'])}")
                    if m.get("tables"):
                        st.caption(f"🗂 Tables: {', '.join(m['tables'])}")
                with c2:
                    if m.get("filters"):
                        st.caption(f"🔍 Filters: {', '.join(m['filters'])}")
                    if m.get("caveats"):
                        st.caption(f"⚠️ {m['caveats']}")
                if st.button("🗑 Delete", key=f"del_{m['name']}"):
                    metrics_catalog.remove(m["name"])
                    st.success(f"Deleted **{m['name']}**")
                    st.rerun()
    else:
        st.caption("No metrics defined yet. Add one below.")

    st.divider()

    st.markdown("#### ＋ Add Metric")
    with st.form("add_metric_form"):
        col1, col2 = st.columns(2)
        with col1:
            name    = st.text_input("Name *", placeholder="revenue")
            sql     = st.text_area("SQL Expression *", placeholder='SUM("order_amount") - SUM("refunds")', height=80)
            caveats = st.text_input("Caveats", placeholder="Finance-approved")
        with col2:
            description = st.text_input("Description", placeholder="Total net revenue excluding refunds")
            dimensions  = st.text_input("Valid dimensions (comma-separated)", placeholder="date, country")
            tables      = st.text_input("Source tables (comma-separated)", placeholder="orders, refunds")

        submitted = st.form_submit_button("Save Metric", type="primary")
        if submitted:
            if not name.strip() or not sql.strip():
                st.error("Name and SQL Expression are required.")
            else:
                try:
                    metric = {
                        "name":        name.strip(),
                        "description": description.strip(),
                        "sql":         sql.strip(),
                        "dimensions":  [d.strip() for d in dimensions.split(",") if d.strip()],
                        "tables":      [t.strip() for t in tables.split(",") if t.strip()],
                        "filters":     [],
                        "caveats":     caveats.strip(),
                    }
                    metrics_catalog.add(metric)
                    st.success(f"✅ Metric **{name}** saved!")
                    st.rerun()
                except Exception as e:
                    st.error(str(e))

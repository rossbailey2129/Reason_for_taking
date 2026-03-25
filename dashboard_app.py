"""
Interactive explorer for taxonomy ↔ health interest associations.
Run: streamlit run dashboard_app.py
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

BASE_DIR = Path(__file__).resolve().parent

LEAF_COL = "LOWEST_TAXONOMY"
HEALTH_COL = "HEALTH_INTEREST"
HEALTH_AREA_COL = "HEALTH_AREA"
TAXONOMY_FILTER_COLS = [
    "CATEGORY_NAME",
    "SUB_CATEGORY_NAME",
    "ATTRIBUTE_NAME",
    "SUB_ATTRIBUTE_NAME",
    LEAF_COL,
]
NUMERIC_METRICS = [
    "REC_COUNT",
    "SHARE_WITHIN_LOWEST_TAXONOMY",
    "SHARE_WITHIN_HEALTH_INTEREST",
]


def _max_label_len(labels: Sequence[object]) -> int:
    if not labels:
        return 0
    return max(len(str(x)) for x in labels)


def _hbar_fig_height(
    num_bars: int,
    category_labels: Sequence[str],
    *,
    min_h: int = 200,
    max_h: int = 2800,
) -> int:
    """Scale vertical size to bar count and longest category label (room for every tick)."""
    if num_bars < 1:
        return min_h
    chrome = 108
    ml = _max_label_len(category_labels)
    per_bar = max(24, min(56, 22 + ml // 3))
    return max(min_h, min(max_h, chrome + num_bars * per_bar))


def _heatmap_fig_height(
    num_rows: int,
    row_labels: Sequence[str],
    *,
    min_h: int = 240,
    max_h: int = 4000,
) -> int:
    if num_rows < 1:
        return min_h
    chrome = 128
    ml = _max_label_len(row_labels)
    per_row = max(22, min(56, 18 + ml // 4))
    return max(min_h, min(max_h, chrome + num_rows * per_row))


def _heatmap_bottom_margin(
    num_cols: int,
    max_x_label_len: int,
    *,
    min_b: int = 100,
    max_b: int = 640,
) -> int:
    """Bottom margin for angled x labels when every column tick is shown."""
    if num_cols < 1:
        return min_b
    col_w = min(num_cols, 100) * 8
    angled = int(min(180, 40 + max_x_label_len * 4.5))
    return max(min_b, min(max_b, 72 + col_w + angled))


def _categorical_axis_show_all(
    *,
    automargin: bool = True,
    tickangle: float | None = None,
    tick_font_size: int | None = None,
    tickson_labels: bool = True,
) -> dict:
    """Plotly often hides categorical ticks when dense; force every label."""
    d: dict = {
        "ticklabelstep": 1,
        "showticklabels": True,
        "automargin": automargin,
    }
    if tickson_labels:
        d["tickson"] = "labels"
    if tickangle is not None:
        d["tickangle"] = tickangle
    if tick_font_size is not None:
        d["tickfont"] = dict(size=tick_font_size)
    return d


def _padded_range(
    lo: float,
    hi: float,
    *,
    pad_frac: float = 0.06,
    floor: float | None = 0.0,
    ceiling: float | None = None,
    min_span: float | None = None,
) -> tuple[float, float]:
    """Axis limits with padding; optional floor/ceiling (e.g. [0, 1] for shares)."""
    if lo > hi:
        lo, hi = hi, lo
    if lo == hi:
        span = max(abs(lo) * 0.08, 1e-6) if lo != 0 else 0.02
        a, b = lo - span, hi + span
    else:
        pad = (hi - lo) * pad_frac
        a, b = lo - pad, hi + pad
    if floor is not None:
        a = max(floor, a)
    if ceiling is not None:
        b = min(ceiling, b)
    if a >= b:
        mid = (lo + hi) / 2 if lo <= hi else lo
        span = max(abs(mid) * 0.05, 0.02)
        a, b = mid - span, mid + span
        if floor is not None:
            a = max(floor, a)
        if ceiling is not None:
            b = min(ceiling, b)
    if min_span is not None and b > a and (b - a) < min_span:
        mid = 0.5 * (a + b)
        half = 0.5 * min_span
        na, nb = mid - half, mid + half
        if floor is not None:
            na = max(floor, na)
        if ceiling is not None:
            nb = min(ceiling, nb)
        if nb > na:
            a, b = na, nb
        if (b - a) < min_span and floor == 0.0 and ceiling == 1.0:
            a, b = 0.0, 1.0
    if a >= b:
        mid = (lo + hi) / 2 if lo <= hi else lo
        span = max(abs(mid) * 0.05, 0.02)
        a, b = mid - span, mid + span
        if floor is not None:
            a = max(floor, a)
        if ceiling is not None:
            b = min(ceiling, b)
    return a, b


def _hbar_xaxis_range(series: pd.Series, rank_metric: str) -> tuple[float, float]:
    """
    Horizontal bars extend from x=0. The visible x range must include 0 and the
    bar tips; zooming to [xmin, xmax] clips the bars off-screen.
    """
    s = series.astype(float)
    xmax = float(s.max())
    if pd.isna(xmax):
        xmax = 0.0
    xmax = max(0.0, xmax)
    lo = 0.0
    if "SHARE" in rank_metric:
        pad = max(0.015, xmax * 0.1)
        hi = min(1.0, xmax + pad)
        if hi <= lo:
            hi = min(1.0, 0.05)
        if hi - lo < 0.04:
            hi = min(1.0, max(hi, 0.06))
        return lo, hi
    if xmax == 0.0:
        return 0.0, 1.0
    slack = max(1.0, xmax * 0.08)
    hi = xmax + slack
    if hi - lo < max(2.0, xmax * 0.12):
        hi = lo + max(2.0, xmax * 0.12)
    return lo, float(hi)


def resolve_data_csv() -> Path:
    candidates = list(BASE_DIR.glob("*.csv"))
    if not candidates:
        raise FileNotFoundError(f"No CSV files found in {BASE_DIR}")
    return max(candidates, key=lambda p: p.stat().st_mtime)


@st.cache_data
def load_data(path_str: str) -> pd.DataFrame:
    path = Path(path_str)
    if not path.exists():
        raise FileNotFoundError(f"Expected CSV at {path}")
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]
    required = {*TAXONOMY_FILTER_COLS, HEALTH_COL, HEALTH_AREA_COL, *NUMERIC_METRICS}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"CSV missing columns: {sorted(missing)}")
    for col in NUMERIC_METRICS:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    for col in TAXONOMY_FILTER_COLS:
        df[col] = df[col].fillna("").astype(str)
    df[HEALTH_COL] = df[HEALTH_COL].fillna("").astype(str)
    df[HEALTH_AREA_COL] = df[HEALTH_AREA_COL].fillna("").astype(str)
    df = df.dropna(subset=[LEAF_COL, HEALTH_COL, "REC_COUNT"]).copy()
    return df


def sorted_unique(series: pd.Series) -> list[str]:
    return sorted(set(series.astype(str)))


def apply_filters(
    df: pd.DataFrame,
    taxonomy_selections: dict[str, list[str]],
    health_areas: list[str],
    interests: list[str],
    rec_lo: int,
    rec_hi: int,
    share_lt_lo: float,
    share_lt_hi: float,
    share_hi_lo: float,
    share_hi_hi: float,
) -> pd.DataFrame:
    out = df
    for col, selected in taxonomy_selections.items():
        if selected:
            out = out[out[col].isin(selected)]
    if health_areas:
        out = out[out[HEALTH_AREA_COL].isin(health_areas)]
    if interests:
        out = out[out[HEALTH_COL].isin(interests)]
    rc = out["REC_COUNT"]
    out = out[(rc >= rec_lo) & (rc <= rec_hi)]
    sl = out["SHARE_WITHIN_LOWEST_TAXONOMY"]
    sh = out["SHARE_WITHIN_HEALTH_INTEREST"]
    out = out[
        (sl >= share_lt_lo)
        & (sl <= share_lt_hi)
        & (sh >= share_hi_lo)
        & (sh <= share_hi_hi)
    ]
    return out


def main() -> None:
    st.set_page_config(
        page_title="Taxonomy ↔ Health interest",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    st.title("Taxonomy and health interest associations")
    st.caption(
        "Filter by taxonomy levels, then inspect recommendation count and share "
        "within lowest taxonomy vs within health interest."
    )

    data_path = resolve_data_csv()
    df = load_data(str(data_path))
    st.caption(f"Data file: `{data_path.name}`")

    all_leaf = sorted_unique(df[LEAF_COL])
    all_interests = sorted_unique(df[HEALTH_COL])
    all_health_areas = sorted_unique(df[HEALTH_AREA_COL])

    rmin, rmax = int(df["REC_COUNT"].min()), int(df["REC_COUNT"].max())
    smin_lt = float(df["SHARE_WITHIN_LOWEST_TAXONOMY"].min())
    smax_lt = float(df["SHARE_WITHIN_LOWEST_TAXONOMY"].max())
    smin_h = float(df["SHARE_WITHIN_HEALTH_INTEREST"].min())
    smax_h = float(df["SHARE_WITHIN_HEALTH_INTEREST"].max())

    taxonomy_labels = {
        "CATEGORY_NAME": "Category name",
        "SUB_CATEGORY_NAME": "Sub-category name",
        "ATTRIBUTE_NAME": "Attribute name",
        "SUB_ATTRIBUTE_NAME": "Sub-attribute name",
        LEAF_COL: "Lowest taxonomy",
    }

    if "rec_slider" not in st.session_state:
        st.session_state.rec_slider = (rmin, rmax)
    if "rec_min_input" not in st.session_state:
        st.session_state.rec_min_input = rmin
    if "rec_max_input" not in st.session_state:
        st.session_state.rec_max_input = rmax

    if "share_lt_slider" not in st.session_state:
        st.session_state.share_lt_slider = (smin_lt, smax_lt)
    if "share_lt_min_input" not in st.session_state:
        st.session_state.share_lt_min_input = float(smin_lt)
    if "share_lt_max_input" not in st.session_state:
        st.session_state.share_lt_max_input = float(smax_lt)

    if "share_hi_slider" not in st.session_state:
        st.session_state.share_hi_slider = (smin_h, smax_h)
    if "share_hi_min_input" not in st.session_state:
        st.session_state.share_hi_min_input = float(smin_h)
    if "share_hi_max_input" not in st.session_state:
        st.session_state.share_hi_max_input = float(smax_h)

    def _sync_rec_from_typing() -> None:
        mn = int(st.session_state.rec_min_input)
        mx = int(st.session_state.rec_max_input)
        if mn > mx:
            mn, mx = mx, mn
        mn = max(rmin, min(mn, rmax))
        mx = max(rmin, min(mx, rmax))
        st.session_state.rec_slider = (mn, mx)
        st.session_state.rec_min_input = mn
        st.session_state.rec_max_input = mx

    def _sync_rec_from_slider() -> None:
        lo, hi = st.session_state.rec_slider
        st.session_state.rec_min_input = int(lo)
        st.session_state.rec_max_input = int(hi)

    def _sync_share_lt_from_typing() -> None:
        mn = float(st.session_state.share_lt_min_input)
        mx = float(st.session_state.share_lt_max_input)
        if mn > mx:
            mn, mx = mx, mn
        mn = max(smin_lt, min(mn, smax_lt))
        mx = max(smin_lt, min(mx, smax_lt))
        st.session_state.share_lt_slider = (mn, mx)
        st.session_state.share_lt_min_input = mn
        st.session_state.share_lt_max_input = mx

    def _sync_share_lt_from_slider() -> None:
        lo, hi = st.session_state.share_lt_slider
        st.session_state.share_lt_min_input = float(lo)
        st.session_state.share_lt_max_input = float(hi)

    def _sync_share_hi_from_typing() -> None:
        mn = float(st.session_state.share_hi_min_input)
        mx = float(st.session_state.share_hi_max_input)
        if mn > mx:
            mn, mx = mx, mn
        mn = max(smin_h, min(mn, smax_h))
        mx = max(smin_h, min(mx, smax_h))
        st.session_state.share_hi_slider = (mn, mx)
        st.session_state.share_hi_min_input = mn
        st.session_state.share_hi_max_input = mx

    def _sync_share_hi_from_slider() -> None:
        lo, hi = st.session_state.share_hi_slider
        st.session_state.share_hi_min_input = float(lo)
        st.session_state.share_hi_max_input = float(hi)

    reset_keys: list[str] = []

    with st.sidebar:
        st.header("Filters")
        with st.expander("Taxonomy (empty = all)", expanded=True):
            taxonomy_selections: dict[str, list[str]] = {}
            for col in TAXONOMY_FILTER_COLS:
                opts = sorted_unique(df[col])
                key = f"ms_{col}"
                reset_keys.append(key)
                taxonomy_selections[col] = st.multiselect(
                    taxonomy_labels[col],
                    options=opts if opts else [""],
                    default=[],
                    key=key,
                )

        sel_health_areas = st.multiselect(
            "Health areas (empty = all)",
            options=all_health_areas if all_health_areas else [""],
            default=[],
            key="health_area_ms",
        )
        reset_keys.append("health_area_ms")

        sel_interests = st.multiselect(
            "Health interests (empty = all)",
            options=all_interests if all_interests else [""],
            default=[],
            key="interests_ms",
        )
        reset_keys.append("interests_ms")

        st.markdown("**Recommendation count**")
        st.caption("Set min/max below or use the slider; values stay in sync.")
        r1, r2 = st.columns(2)
        with r1:
            st.number_input(
                "Min",
                min_value=rmin,
                max_value=rmax,
                step=1,
                key="rec_min_input",
                on_change=_sync_rec_from_typing,
            )
        with r2:
            st.number_input(
                "Max",
                min_value=rmin,
                max_value=rmax,
                step=1,
                key="rec_max_input",
                on_change=_sync_rec_from_typing,
            )
        rec_range = st.slider(
            "Range",
            min_value=rmin,
            max_value=rmax,
            key="rec_slider",
            on_change=_sync_rec_from_slider,
        )
        reset_keys.extend(["rec_slider", "rec_min_input", "rec_max_input"])

        st.markdown("**Share within lowest taxonomy**")
        st.caption("Min/max fields accept decimals; slider is a quick adjustment.")
        s1, s2 = st.columns(2)
        with s1:
            st.number_input(
                "Min",
                min_value=smin_lt,
                max_value=smax_lt,
                step=0.0001,
                format="%.6f",
                key="share_lt_min_input",
                on_change=_sync_share_lt_from_typing,
            )
        with s2:
            st.number_input(
                "Max",
                min_value=smin_lt,
                max_value=smax_lt,
                step=0.0001,
                format="%.6f",
                key="share_lt_max_input",
                on_change=_sync_share_lt_from_typing,
            )
        share_lt = st.slider(
            "Range",
            min_value=smin_lt,
            max_value=smax_lt,
            step=0.0001,
            format="%.4f",
            key="share_lt_slider",
            on_change=_sync_share_lt_from_slider,
        )
        reset_keys.extend(
            ["share_lt_slider", "share_lt_min_input", "share_lt_max_input"]
        )

        st.markdown("**Share within health interest**")
        h1, h2 = st.columns(2)
        with h1:
            st.number_input(
                "Min",
                min_value=smin_h,
                max_value=smax_h,
                step=0.0001,
                format="%.6f",
                key="share_hi_min_input",
                on_change=_sync_share_hi_from_typing,
            )
        with h2:
            st.number_input(
                "Max",
                min_value=smin_h,
                max_value=smax_h,
                step=0.0001,
                format="%.6f",
                key="share_hi_max_input",
                on_change=_sync_share_hi_from_typing,
            )
        share_hi = st.slider(
            "Range",
            min_value=smin_h,
            max_value=smax_h,
            step=0.0001,
            format="%.4f",
            key="share_hi_slider",
            on_change=_sync_share_hi_from_slider,
        )
        reset_keys.extend(
            ["share_hi_slider", "share_hi_min_input", "share_hi_max_input"]
        )

        if st.button("Reset filters"):
            for k in reset_keys:
                st.session_state.pop(k, None)
            st.rerun()

    filtered = apply_filters(
        df,
        taxonomy_selections,
        sel_health_areas,
        sel_interests,
        rec_range[0],
        rec_range[1],
        share_lt[0],
        share_lt[1],
        share_hi[0],
        share_hi[1],
    )
    st.sidebar.metric("Rows after filters", len(filtered))

    hover_extra = [
        c
        for c in (
            "CATEGORY_NAME",
            "SUB_CATEGORY_NAME",
            "ATTRIBUTE_NAME",
            "SUB_ATTRIBUTE_NAME",
            HEALTH_AREA_COL,
        )
        if c in df.columns
    ]

    tab_table, tab_leaf, tab_hi, tab_matrix, tab_scatter = st.tabs(
        [
            "Data table",
            "By lowest taxonomy",
            "By health interest",
            "Heatmap (top pairs)",
            "Scatter",
        ]
    )

    with tab_table:
        st.subheader("Filtered rows")
        sort_opt = st.selectbox("Sort by", NUMERIC_METRICS)
        ascending = st.checkbox("Ascending", value=False)
        show = filtered.sort_values(sort_opt, ascending=ascending)
        table_h = max(220, min(900, 72 + min(len(show), 40) * 28))
        st.dataframe(
            show,
            use_container_width=True,
            hide_index=True,
            height=int(table_h),
        )
        csv_bytes = show.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Download filtered CSV",
            data=csv_bytes,
            file_name="filtered_associations.csv",
            mime="text/csv",
        )

    with tab_leaf:
        st.subheader("Which health interests associate most with a lowest taxonomy?")
        rank_metric = st.radio(
            "Rank bars by",
            NUMERIC_METRICS,
            horizontal=True,
            key="metric_leaf_tab",
        )
        pick_leaf = st.selectbox(
            "Lowest taxonomy",
            options=all_leaf,
            key="pick_leaf",
        )
        top_n = st.number_input("Top N interests", 5, 100, 20, key="topn_leaf")
        sub = filtered[filtered[LEAF_COL] == pick_leaf].nlargest(
            int(top_n), rank_metric
        )
        if sub.empty:
            st.info("No rows for this lowest taxonomy under current filters.")
        else:
            sub_disp = sub.sort_values(rank_metric, ascending=True)
            y_labels = sub_disp[HEALTH_COL].astype(str).tolist()
            tfs = 10 if len(sub_disp) > 35 else 11
            fig = px.bar(
                sub_disp,
                x=rank_metric,
                y=HEALTH_COL,
                orientation="h",
                hover_data=[*NUMERIC_METRICS, *hover_extra],
                labels={rank_metric: rank_metric.replace("_", " ")},
            )
            xr = sub_disp[rank_metric].astype(float)
            x0, x1 = _hbar_xaxis_range(xr, rank_metric)
            fig.update_layout(
                yaxis={
                    "categoryorder": "total ascending",
                    **_categorical_axis_show_all(tick_font_size=tfs),
                },
                xaxis={"range": [x0, x1], "automargin": True},
                height=_hbar_fig_height(len(sub_disp), y_labels),
                margin=dict(l=16, r=24, t=48, b=48),
            )
            st.plotly_chart(fig, use_container_width=True)

    with tab_hi:
        st.subheader("Which lowest taxonomies associate most with a health interest?")
        rank_metric_hi = st.radio(
            "Rank bars by",
            NUMERIC_METRICS,
            horizontal=True,
            key="metric_hi_tab",
        )
        pick_hi = st.selectbox(
            "Health interest",
            options=all_interests,
            key="pick_interest",
        )
        top_n_hi = st.number_input("Top N lowest taxonomies", 5, 100, 20, key="topn_hi")
        sub_hi = filtered[filtered[HEALTH_COL] == pick_hi].nlargest(
            int(top_n_hi), rank_metric_hi
        )
        if sub_hi.empty:
            st.info("No rows for this interest under current filters.")
        else:
            sub_hi_disp = sub_hi.sort_values(rank_metric_hi, ascending=True)
            y_labels_hi = sub_hi_disp[LEAF_COL].astype(str).tolist()
            tfs_hi = 10 if len(sub_hi_disp) > 35 else 11
            fig2 = px.bar(
                sub_hi_disp,
                x=rank_metric_hi,
                y=LEAF_COL,
                orientation="h",
                hover_data=[*NUMERIC_METRICS, *hover_extra],
                labels={rank_metric_hi: rank_metric_hi.replace("_", " ")},
            )
            xr2 = sub_hi_disp[rank_metric_hi].astype(float)
            x0b, x1b = _hbar_xaxis_range(xr2, rank_metric_hi)
            fig2.update_layout(
                yaxis={
                    "categoryorder": "total ascending",
                    **_categorical_axis_show_all(tick_font_size=tfs_hi),
                },
                xaxis={"range": [x0b, x1b], "automargin": True},
                height=_hbar_fig_height(len(sub_hi_disp), y_labels_hi),
                margin=dict(l=16, r=24, t=48, b=48),
            )
            st.plotly_chart(fig2, use_container_width=True)

    with tab_matrix:
        st.subheader("Heatmap of top lowest taxonomies × top interests (after filters)")
        st.caption(
            "Only the busiest lowest taxonomies and interests in the filtered set "
            "are shown."
        )
        col_a, col_b, col_c = st.columns(3)
        with col_a:
            k_leaf = st.number_input(
                "Top lowest taxonomies (by total rec count)", 5, 120, 25
            )
        with col_b:
            k_hi = st.number_input("Top health interests", 5, 120, 20)
        with col_c:
            cell_metric = st.selectbox("Cell value", NUMERIC_METRICS)
        if filtered.empty:
            st.warning("No data after filters.")
        else:
            leaf_totals = filtered.groupby(LEAF_COL, as_index=False)[
                "REC_COUNT"
            ].sum()
            top_leaves = leaf_totals.nlargest(int(k_leaf), "REC_COUNT")[
                LEAF_COL
            ].tolist()
            hi_totals = filtered.groupby(HEALTH_COL, as_index=False)[
                "REC_COUNT"
            ].sum()
            top_his = hi_totals.nlargest(int(k_hi), "REC_COUNT")[
                HEALTH_COL
            ].tolist()
            hm = filtered[
                filtered[LEAF_COL].isin(top_leaves)
                & filtered[HEALTH_COL].isin(top_his)
            ]
            pivot = hm.pivot_table(
                index=LEAF_COL,
                columns=HEALTH_COL,
                values=cell_metric,
                aggfunc="max",
            )
            pivot = pivot.reindex(index=top_leaves, columns=top_his)
            fig_hm = go.Figure(
                data=go.Heatmap(
                    z=pivot.values,
                    x=pivot.columns.tolist(),
                    y=pivot.index.tolist(),
                    colorscale="Blues",
                    hoverongaps=False,
                    colorbar=dict(title=cell_metric.replace("_", " ")),
                )
            )
            n_rows, n_cols = len(pivot.index), len(pivot.columns)
            row_ix = [str(i) for i in pivot.index.tolist()]
            col_ix = [str(c) for c in pivot.columns.tolist()]
            mx_len_x = _max_label_len(col_ix)
            xfs = 9 if n_cols > 40 else 10
            yfs = 9 if n_rows > 40 else 10
            fig_hm.update_layout(
                xaxis=dict(
                    side="bottom",
                    **_categorical_axis_show_all(
                        automargin=True,
                        tickangle=-50,
                        tick_font_size=xfs,
                        tickson_labels=False,
                    ),
                ),
                yaxis=_categorical_axis_show_all(
                    automargin=True,
                    tick_font_size=yfs,
                    tickson_labels=False,
                ),
                height=_heatmap_fig_height(n_rows, row_ix),
                margin=dict(
                    l=16,
                    r=24,
                    t=48,
                    b=_heatmap_bottom_margin(n_cols, mx_len_x),
                ),
            )
            st.plotly_chart(fig_hm, use_container_width=True)

    with tab_scatter:
        st.subheader("Share within lowest taxonomy vs within health interest")
        st.caption("Point size = recommendation count (filtered data).")
        if filtered.empty:
            st.warning("No data after filters.")
        else:
            max_points = st.slider("Max points to plot", 500, 9000, 4000)
            sample = filtered.nlargest(max_points, "REC_COUNT")
            hover_map = {
                HEALTH_COL: True,
                "REC_COUNT": True,
                "SHARE_WITHIN_LOWEST_TAXONOMY": ":.4f",
                "SHARE_WITHIN_HEALTH_INTEREST": ":.4f",
            }
            for c in hover_extra:
                hover_map[c] = True
            fig_s = px.scatter(
                sample,
                x="SHARE_WITHIN_LOWEST_TAXONOMY",
                y="SHARE_WITHIN_HEALTH_INTEREST",
                size="REC_COUNT",
                hover_name=LEAF_COL,
                hover_data=hover_map,
                opacity=0.65,
            )
            fig_s.update_traces(
                marker=dict(
                    sizemin=6,
                    line=dict(width=0.5, color="DarkSlateGrey"),
                )
            )
            xs = sample["SHARE_WITHIN_LOWEST_TAXONOMY"].astype(float)
            ys = sample["SHARE_WITHIN_HEALTH_INTEREST"].astype(float)
            sx0, sx1 = _padded_range(
                float(xs.min()),
                float(xs.max()),
                floor=0.0,
                ceiling=1.0,
                min_span=0.03,
            )
            sy0, sy1 = _padded_range(
                float(ys.min()),
                float(ys.max()),
                floor=0.0,
                ceiling=1.0,
                min_span=0.03,
            )
            scatter_h = max(360, min(900, 320 + int(len(sample) ** 0.45) * 8))
            fig_s.update_layout(
                height=int(scatter_h),
                xaxis=dict(range=[sx0, sx1], automargin=True),
                yaxis=dict(range=[sy0, sy1], automargin=True),
                margin=dict(l=16, r=16, t=48, b=48),
            )
            st.plotly_chart(fig_s, use_container_width=True)


if __name__ == "__main__":
    main()

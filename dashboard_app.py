"""
Interactive explorer for taxonomy ↔ health interest associations.
Run: streamlit run dashboard_app.py
"""

from __future__ import annotations

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

BAR_FILL = "#e6f1fc"
CHART_TEXT = "#36485c"
# Figure spaces: with insidetextanchor="end", trailing padding pushes digits left of the bar tip
_BAR_LABEL_END_PAD = "\u2007" * 1
# Plotly + browser; load Besley via _inject_app_font() for Streamlit UI
FONT_FAMILY = "Besley, Georgia, serif"


def _inject_app_font() -> None:
    """Load Besley from Google Fonts and apply across the Streamlit app."""
    st.markdown(
        """
        <link rel="preconnect" href="https://fonts.googleapis.com">
        <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
        <link href="https://fonts.googleapis.com/css2?family=Besley:ital,wght@0,400;0,600;0,700;1,400&display=swap" rel="stylesheet">
        <style>
            /* Besley as default text. Do NOT use `.stApp * { ... !important }` — that
               overrides Streamlit's Material Symbols font on expander/chevron icons and
               shows raw names like "arrow_back_ios_new" on top of labels. */
            html, body {
                font-family: 'Besley', Georgia, serif;
            }
            .stApp {
                font-family: 'Besley', Georgia, serif;
            }
            .stMarkdown, .stMarkdown p, .stMarkdown span,
            [data-testid="stSidebar"] label,
            [data-testid="stSidebar"] p,
            [data-testid="stSidebar"] .stMarkdown,
            [data-testid="stAppViewContainer"] .block-container,
            h1, h2, h3, h4, h5, h6 {
                font-family: 'Besley', Georgia, serif;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _plot_base_font() -> dict:
    return {"family": FONT_FAMILY, "color": CHART_TEXT}


def _tick_font() -> dict:
    return {"family": FONT_FAMILY, "color": CHART_TEXT}


def _axis_title_font() -> dict:
    return {"font": {"family": FONT_FAMILY, "color": CHART_TEXT}}


def _bar_data_labels(values: pd.Series, metric_col: str) -> list[str]:
    """Format values shown at the inside end of horizontal bars."""
    labels: list[str] = []
    for v in values:
        f = float(v)
        if "SHARE" in metric_col:
            labels.append(f"{f:.3f}{_BAR_LABEL_END_PAD}")
        else:
            labels.append(f"{int(round(f)):,}{_BAR_LABEL_END_PAD}")
    return labels


def _bar_label_font_size(num_bars: int) -> int:
    """Keep labels readable when many bars; taper slightly for very dense charts."""
    n = max(1, min(int(num_bars), 100))
    return int(max(13, min(18, 21.0 - n * 0.07)))


def _padded_range(
    lo: float,
    hi: float,
    *,
    pad_frac: float = 0.06,
    floor: float | None = 0.0,
    ceiling: float | None = None,
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
    return a, b


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
    _inject_app_font()
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
        st.dataframe(
            show,
            use_container_width=True,
            hide_index=True,
            height=480,
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
            fig = px.bar(
                sub,
                x=rank_metric,
                y=HEALTH_COL,
                orientation="h",
                hover_data=[*NUMERIC_METRICS, *hover_extra],
                labels={rank_metric: rank_metric.replace("_", " ")},
                color_discrete_sequence=[BAR_FILL],
            )
            bar_lbl = _bar_data_labels(sub[rank_metric], rank_metric)
            n_bars = len(sub)
            lbl_px = _bar_label_font_size(n_bars)
            fig.update_traces(
                marker=dict(
                    color=BAR_FILL,
                    line=dict(width=0.5, color=CHART_TEXT),
                ),
                text=bar_lbl,
                textposition="inside",
                insidetextanchor="end",
                textangle=0,
                constraintext="none",
                textfont=dict(family=FONT_FAMILY, color=CHART_TEXT, size=lbl_px),
            )
            fig.update_layout(
                font=_plot_base_font(),
                hoverlabel=dict(font=dict(family=FONT_FAMILY, size=13)),
                yaxis={"categoryorder": "total ascending"},
                height=max(400, 24 * n_bars),
                margin=dict(l=200),
                uniformtext_minsize=12,
                uniformtext_mode="show",
            )
            fig.update_xaxes(
                tickfont=_tick_font(),
                title_font=dict(family=FONT_FAMILY, color=CHART_TEXT),
            )
            fig.update_yaxes(
                tickfont=_tick_font(),
                title_font=dict(family=FONT_FAMILY, color=CHART_TEXT),
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
            fig2 = px.bar(
                sub_hi,
                x=rank_metric_hi,
                y=LEAF_COL,
                orientation="h",
                hover_data=[*NUMERIC_METRICS, *hover_extra],
                labels={rank_metric_hi: rank_metric_hi.replace("_", " ")},
                color_discrete_sequence=[BAR_FILL],
            )
            bar_lbl_hi = _bar_data_labels(sub_hi[rank_metric_hi], rank_metric_hi)
            n_bars_hi = len(sub_hi)
            lbl_px_hi = _bar_label_font_size(n_bars_hi)
            fig2.update_traces(
                marker=dict(
                    color=BAR_FILL,
                    line=dict(width=0.5, color=CHART_TEXT),
                ),
                text=bar_lbl_hi,
                textposition="inside",
                insidetextanchor="end",
                textangle=0,
                constraintext="none",
                textfont=dict(family=FONT_FAMILY, color=CHART_TEXT, size=lbl_px_hi),
            )
            fig2.update_layout(
                font=_plot_base_font(),
                hoverlabel=dict(font=dict(family=FONT_FAMILY, size=13)),
                yaxis={"categoryorder": "total ascending"},
                height=max(400, 24 * n_bars_hi),
                margin=dict(l=220),
                uniformtext_minsize=12,
                uniformtext_mode="show",
            )
            fig2.update_xaxes(
                tickfont=_tick_font(),
                title_font=dict(family=FONT_FAMILY, color=CHART_TEXT),
            )
            fig2.update_yaxes(
                tickfont=_tick_font(),
                title_font=dict(family=FONT_FAMILY, color=CHART_TEXT),
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
                "Top lowest taxonomies (by total rec count)", 5, 80, 25
            )
        with col_b:
            k_hi = st.number_input("Top health interests", 5, 80, 20)
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
                    colorbar=dict(
                        title=dict(
                            text=cell_metric.replace("_", " "),
                            font=dict(family=FONT_FAMILY, color=CHART_TEXT, size=12),
                        ),
                        tickfont=dict(family=FONT_FAMILY, color=CHART_TEXT),
                    ),
                )
            )
            n_rows = len(pivot.index)
            fig_hm.update_layout(
                font=_plot_base_font(),
                hoverlabel=dict(font=dict(family=FONT_FAMILY, size=13)),
                xaxis=dict(
                    side="bottom",
                    tickangle=-45,
                    tickfont=_tick_font(),
                    title=_axis_title_font(),
                ),
                yaxis=dict(tickfont=_tick_font(), title=_axis_title_font()),
                height=max(500, 14 * n_rows),
                margin=dict(l=200, b=200),
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
            fig_s.update_traces(marker=dict(line=dict(width=0.5, color="DarkSlateGrey")))
            xs = sample["SHARE_WITHIN_LOWEST_TAXONOMY"].astype(float)
            ys = sample["SHARE_WITHIN_HEALTH_INTEREST"].astype(float)
            sx0, sx1 = _padded_range(
                float(xs.min()), float(xs.max()), floor=0.0, ceiling=1.0
            )
            sy0, sy1 = _padded_range(
                float(ys.min()), float(ys.max()), floor=0.0, ceiling=1.0
            )
            fig_s.update_layout(
                font=_plot_base_font(),
                hoverlabel=dict(font=dict(family=FONT_FAMILY, size=13)),
                height=640,
                xaxis=dict(range=[sx0, sx1]),
                yaxis=dict(range=[sy0, sy1]),
            )
            fig_s.update_xaxes(
                tickfont=_tick_font(),
                title_font=dict(family=FONT_FAMILY, color=CHART_TEXT),
            )
            fig_s.update_yaxes(
                tickfont=_tick_font(),
                title_font=dict(family=FONT_FAMILY, color=CHART_TEXT),
            )
            st.plotly_chart(fig_s, use_container_width=True)


if __name__ == "__main__":
    main()

"""
Interactive explorer for taxonomy ↔ health interest associations.
Run: streamlit run dashboard_app.py
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.colors as pc
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components

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
SHARE_COLS = frozenset(
    {"SHARE_WITHIN_LOWEST_TAXONOMY", "SHARE_WITHIN_HEALTH_INTEREST"}
)
# Bump when in-memory scale for share columns changes (invalidates old 0–1 slider state).
_SHARE_UI_VERSION = 2
# Quadrant tab: Top N widget session key (fresh key avoids stale “all taxa” values).
_QUADRANT_TOP_N_KEY = "quad_top_n_chart"
_QUAD_PAIR_KEY_SEP = "\x1f"

BAR_FILL = "#e6f1fc"
CHART_TEXT = "#36485c"
HEATMAP_COLORSCALE = "Blues"
# Embedded heatmap viewport: larger values use more of the browser window before inner scroll.
HEATMAP_EMBED_MAX_VIEWPORT_PX = 2000
HEATMAP_EMBED_VIEWPORT_VH = 1000
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
            /* Let wide layout use more of the viewport (helps large heatmaps). */
            .main .block-container {
                max-width: min(1680px, 96vw);
                padding-left: 2rem;
                padding-right: 2rem;
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


def _metric_axis_label(metric_col: str) -> str:
    """Axis / colorbar titles; share columns are stored as 0–100 (percent points)."""
    base = metric_col.replace("_", " ")
    if metric_col in SHARE_COLS:
        return f"{base} (%)"
    return base


def _bar_data_labels(values: pd.Series, metric_col: str) -> list[str]:
    """Formatted values for bar annotations (all labels use textposition='outside')."""
    labels: list[str] = []
    for v in values.astype(float):
        f = float(v)
        if metric_col in SHARE_COLS:
            labels.append(f"{f:.1f}%")
        else:
            labels.append(f"{int(round(f)):,}")
    return labels


def _bar_label_font_size(num_bars: int) -> int:
    """Keep labels readable when many bars; taper slightly for very dense charts."""
    n = max(1, min(int(num_bars), 100))
    return int(max(13, min(14, 26.0 - n * 0.07)))


def _heatmap_cell_labels(pivot: pd.DataFrame, cell_metric: str) -> list[list[str]]:
    """Human-readable strings for each heatmap cell (NaN → empty)."""
    rows: list[list[str]] = []
    for _, row in pivot.iterrows():
        cells: list[str] = []
        for v in row:
            if pd.isna(v):
                cells.append("")
            elif cell_metric == "REC_COUNT":
                cells.append(f"{int(round(float(v))):,}")
            else:
                cells.append(f"{float(v):.1f}%")
        rows.append(cells)
    return rows


def _heatmap_label_font_size(n_rows: int, n_cols: int) -> int:
    n = max(n_rows * n_cols, 1)
    return int(max(7, min(11, 18 - n**0.35)))


def _heatmap_figure_height(n_rows: int) -> int:
    """
    Total figure height scales with the number of Y-axis categories so each row
    stays tall enough for axis labels and in-cell annotations (not a fixed squat chart).
    """
    if n_rows <= 0:
        return 420
    # Margins already reserve space for long y-labels and tilted x-labels; this is
    # incremental plot area per category row.
    margin_stack = 340
    per_row_px = 34
    return int(min(8000, max(480, margin_stack + n_rows * per_row_px)))


def _heatmap_figure_width(n_cols: int) -> int:
    """
    Total figure width scales with X-axis categories so columns stay wide enough
    for tilted tick labels and in-cell text (embedded via _show_plotly_figure_scrollable).
    """
    if n_cols <= 0:
        return 840
    # Matches margin.l=200; colorbar + right padding ~160px inside the figure.
    horizontal_margin = 380
    per_col_px = 54
    return int(min(6000, max(840, horizontal_margin + n_cols * per_col_px)))


def _heatmap_bottom_margin(n_cols: int) -> int:
    """Extra space for -45° x tick labels when there are many columns."""
    return int(min(320, max(180, 140 + n_cols * 4)))


def _show_plotly_figure_scrollable(fig: go.Figure, *, iframe_pad: int = 40) -> None:
    """
    Embed Plotly in an iframe: inner min-width preserves horizontal layout; outer
    max-height + overflow auto adds vertical (and horizontal) scrolling so large
    heatmaps are not squished (st.plotly_chart does not do this reliably).
    """
    lw = fig.layout.width
    lh = fig.layout.height
    w = int(lw) if lw is not None else 900
    h = int(lh) if lh is not None else 600
    inner = fig.to_html(
        include_plotlyjs="cdn",
        full_html=False,
        config={"responsive": False, "displayModeBar": True},
    )
    cap_px = HEATMAP_EMBED_MAX_VIEWPORT_PX
    cap_vh = HEATMAP_EMBED_VIEWPORT_VH
    html = (
        '<div style="'
        f"max-height:min({cap_px}px,{cap_vh}vh);"
        "overflow-x:auto;overflow-y:auto;width:100%;"
        '-webkit-overflow-scrolling:touch;">'
        f'<div style="min-width:{w}px;width:{w}px;max-width:none;">'
        f"{inner}"
        "</div></div>"
    )
    iframe_h = min(h + iframe_pad, cap_px) + 8
    components.html(html, height=iframe_h, scrolling=True)


def _parse_plotly_color_to_rgb(color: str) -> tuple[float, float, float]:
    """Parse Plotly rgb()/rgba() or #RRGGBB into 0–255 components."""
    c = color.strip()
    if c.startswith("rgb"):
        inner = c.split("(", 1)[1].rsplit(")", 1)[0]
        parts = [float(x.strip()) for x in inner.split(",")[:3]]
        return parts[0], parts[1], parts[2]
    if c.startswith("#") and len(c) == 7:
        return (
            float(int(c[1:3], 16)),
            float(int(c[3:5], 16)),
            float(int(c[5:7], 16)),
        )
    raise ValueError(f"Unsupported color string: {color!r}")


def _relative_luminance_srgb(r: float, g: float, b: float) -> float:
    """WCAG relative luminance; r,g,b are 0–255 sRGB."""
    def channel_lin(x: float) -> float:
        u = x / 255.0
        return u / 12.92 if u <= 0.03928 else ((u + 0.055) / 1.055) ** 2.4

    return (
        0.2126 * channel_lin(r)
        + 0.7152 * channel_lin(g)
        + 0.0722 * channel_lin(b)
    )


def _heatmap_cell_text_colors(
    z: np.ndarray,
    colorscale_name: str,
    *,
    dark_text: str = CHART_TEXT,
    light_text: str = "#f8fafc",
    lum_threshold: float = 0.52,
) -> list[list[str]]:
    """
    One text color per cell so labels stay readable on the chosen colorscale.
    Matches Plotly's mapping of z to the scale using the same zmin/zmax as the trace.
    """
    arr = np.asarray(z, dtype=float)
    valid = arr[~np.isnan(arr)]
    if valid.size == 0:
        zmin, zmax = 0.0, 1.0
    else:
        zmin = float(np.nanmin(valid))
        zmax = float(np.nanmax(valid))
    rows: list[list[str]] = []
    for i in range(arr.shape[0]):
        row: list[str] = []
        for j in range(arr.shape[1]):
            v = arr[i, j]
            if np.isnan(v):
                row.append(dark_text)
                continue
            if zmax > zmin:
                t = (float(v) - zmin) / (zmax - zmin)
            else:
                t = 0.5
            t = max(0.0, min(1.0, t))
            cstr = pc.sample_colorscale(colorscale_name, [t], colortype="rgb")[0]
            r, g, b = _parse_plotly_color_to_rgb(cstr)
            lum = _relative_luminance_srgb(r, g, b)
            row.append(light_text if lum < lum_threshold else dark_text)
        rows.append(row)
    return rows


def _weighted_mean(g: pd.DataFrame, col: str) -> float:
    w = float(g["REC_COUNT"].sum())
    if w <= 0:
        return float("nan")
    return float((g[col].astype(float) * g["REC_COUNT"].astype(float)).sum() / w)


def _aggregate_for_bar_chart(
    df: pd.DataFrame, group_col: str, rank_metric: str
) -> pd.DataFrame:
    """
    One row per group_col value: summed REC_COUNT and REC_COUNT-weighted means for
    share columns (for ranking / hover after filtering by taxonomy or interest).
    """
    if df.empty:
        return pd.DataFrame()
    rows: list[dict] = []
    for name, g in df.groupby(group_col, sort=False):
        rec = int(g["REC_COUNT"].sum())
        sm_lt = _weighted_mean(g, "SHARE_WITHIN_LOWEST_TAXONOMY")
        sm_hi = _weighted_mean(g, "SHARE_WITHIN_HEALTH_INTEREST")
        if rank_metric == "REC_COUNT":
            rank_val = float(rec)
        elif rank_metric == "SHARE_WITHIN_LOWEST_TAXONOMY":
            rank_val = sm_lt
        else:
            rank_val = sm_hi
        rows.append(
            {
                group_col: name,
                rank_metric: rank_val,
                "REC_COUNT": rec,
                "SHARE_WITHIN_LOWEST_TAXONOMY": sm_lt,
                "SHARE_WITHIN_HEALTH_INTEREST": sm_hi,
            }
        )
    return pd.DataFrame(rows)


def _padded_range(
    lo: float,
    hi: float,
    *,
    pad_frac: float = 0.06,
    floor: float | None = 0.0,
    ceiling: float | None = None,
) -> tuple[float, float]:
    """Axis limits with padding; optional floor/ceiling (e.g. [0, 100] for shares)."""
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
    for col in SHARE_COLS:
        df[col] = df[col] * 100.0
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


def apply_tab_excludes(
    df: pd.DataFrame,
    exclude_leaves: list[str],
    exclude_interests: list[str],
) -> pd.DataFrame:
    """Remove rows whose lowest taxonomy or health interest appears in exclude lists."""
    out = df
    if exclude_leaves:
        out = out[~out[LEAF_COL].isin(exclude_leaves)]
    if exclude_interests:
        out = out[~out[HEALTH_COL].isin(exclude_interests)]
    return out


def _tab_exclude_expander(tab_id: str, base: pd.DataFrame) -> pd.DataFrame:
    """
    Per-tab exclude multiselects (choices from current sidebar-filtered data).
    Returns base with excluded taxonomies / interests removed.
    """
    leaf_opts = sorted_unique(base[LEAF_COL]) if len(base) else []
    hi_opts = sorted_unique(base[HEALTH_COL]) if len(base) else []
    with st.expander(
        "Exclude from this tab (applies on top of sidebar filters)",
        expanded=False,
    ):
        st.caption(
            "Drop rows for any selected lowest taxonomy and/or health interest. "
            "Other tabs keep their own exclude lists."
        )
        c1, c2 = st.columns(2)
        with c1:
            ex_leaves = st.multiselect(
                "Exclude lowest taxonomies",
                options=leaf_opts if leaf_opts else [""],
                default=[],
                key=f"tab_ex_leaf_{tab_id}",
            )
        with c2:
            ex_interests = st.multiselect(
                "Exclude health interests",
                options=hi_opts if hi_opts else [""],
                default=[],
                key=f"tab_ex_hi_{tab_id}",
            )
    return apply_tab_excludes(base, ex_leaves, ex_interests)


def _snap_symmetric_half_for_ticks(half: float, *, min_half: float = 1.0) -> float:
    """Ceil half to a readable step: 0.1s when small, then integers, 5s, 10s."""
    half = max(float(half), min_half)
    if half < 1.0:
        step = 0.1
        return float(max(min_half, math.ceil(half / step - 1e-12) * step))
    if half <= 5:
        return float(max(min_half, math.ceil(half)))
    if half < 100:
        return float(max(min_half, math.ceil(half / 5.0) * 5.0))
    step = 10.0
    return float(max(min_half, math.ceil(half / step) * step))


def _quadrant_axis_half_from_series(
    centered: pd.Series,
    *,
    pad_frac: float = 0.08,
    min_pts_pad: float = 10.0,
    min_half: float = 1.0,
) -> float:
    """
    Symmetric half-range from the **plotted** axis values (same rows as the scatter).
    Uses max(|min|, |max|), padding, ``min_pts_pad``, then tick-friendly snapping.
    """
    v = pd.to_numeric(centered, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    if v.empty:
        return min_half
    ext = max(abs(float(v.min())), abs(float(v.max())))
    if ext == 0:
        return float(max(min_half, 0.1 if min_half < 1.0 else 1.0))
    half = max(
        ext * (1.0 + pad_frac),
        ext + min_pts_pad,
        min_half,
    )
    return _snap_symmetric_half_for_ticks(half, min_half=min_half)


def _quadrant_label_annotations() -> list[dict]:
    """
    Quadrant titles centered above the upper half and below the lower half
    (axis domain coordinates: 0–1 = plot area).
    """
    fs = 11
    bg = "rgba(255,255,255,0.88)"
    common = dict(
        xref="x domain",
        yref="y domain",
        showarrow=False,
        font=dict(family=FONT_FAMILY, size=fs, color=CHART_TEXT),
        opacity=1.0,
        bgcolor=bg,
        borderpad=3,
        xanchor="center",
    )
    y_above = 1.05
    y_below = -0.05
    return [
        {
            **common,
            "x": 0.75,
            "y": y_above,
            "yanchor": "bottom",
            "text": "Core Specialists",
        },
        {
            **common,
            "x": 0.25,
            "y": y_above,
            "yanchor": "bottom",
            "text": "Specialized Options",
        },
        {
            **common,
            "x": 0.25,
            "y": y_below,
            "yanchor": "top",
            "text": "Low Impact",
        },
        {
            **common,
            "x": 0.75,
            "y": y_below,
            "yanchor": "top",
            "text": "Multi-use Staples",
        },
    ]


def _quadrant_plot_frame(
    df: pd.DataFrame,
    selected_interests: list[str],
    top_n_taxonomies: int,
) -> tuple[pd.DataFrame, float, float]:
    """
    Rows for selected health interests (empty = all), restricted to top-N lowest taxonomies
    by total REC_COUNT in that slice. Uses **precomputed** share columns from the CSV.

    **Log baseline fix:** ``x' = ln(x+1)``, ``y' = ln(y+1)`` on share percentages (natural log),
    then subtract the **median of x'** and **median of y'** among plotted rows. So **(0, 0)**
    is the cohort median in **log share space** (not the same as raw-share median when skewed).
    """
    sub = (
        df[df[HEALTH_COL].isin(selected_interests)]
        if selected_interests
        else df
    )
    if sub.empty:
        return pd.DataFrame(), float("nan"), float("nan")
    leaf_totals = sub.groupby(LEAF_COL, as_index=False)["REC_COUNT"].sum()
    n_take = min(max(1, int(top_n_taxonomies)), len(leaf_totals))
    top_leaves = leaf_totals.nlargest(n_take, "REC_COUNT")[LEAF_COL].tolist()
    plot_df = sub[sub[LEAF_COL].isin(top_leaves)].copy()
    if plot_df.empty:
        return plot_df, float("nan"), float("nan")
    sh = np.maximum(plot_df["SHARE_WITHIN_HEALTH_INTEREST"].astype(float), 0.0)
    sl = np.maximum(plot_df["SHARE_WITHIN_LOWEST_TAXONOMY"].astype(float), 0.0)
    med_hi = float(sh.median())
    med_lt = float(sl.median())
    xp = np.log(sh + 1.0)
    yp = np.log(sl + 1.0)
    med_xp = float(np.median(xp))
    med_yp = float(np.median(yp))
    plot_df["x_vs_median_log"] = xp - med_xp
    plot_df["y_vs_median_log"] = yp - med_yp
    return plot_df, med_hi, med_lt


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
        "within lowest taxonomy vs within health interest (shares shown as percentages, 0–100)."
    )

    data_path = resolve_data_csv()
    df = load_data(str(data_path))
    st.caption(f"Data file: `{data_path.name}`")

    if st.session_state.get("_dashboard_ui_version") != _SHARE_UI_VERSION:
        for k in (
            "share_lt_slider",
            "share_lt_min_input",
            "share_lt_max_input",
            "share_hi_slider",
            "share_hi_min_input",
            "share_hi_max_input",
        ):
            st.session_state.pop(k, None)
        st.session_state["_dashboard_ui_version"] = _SHARE_UI_VERSION

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

        st.markdown("**Share within lowest taxonomy (%)**")
        st.caption("Values are percent points (0–100); min/max and slider stay in sync.")
        s1, s2 = st.columns(2)
        with s1:
            st.number_input(
                "Min",
                min_value=smin_lt,
                max_value=smax_lt,
                step=0.01,
                format="%.2f",
                key="share_lt_min_input",
                on_change=_sync_share_lt_from_typing,
            )
        with s2:
            st.number_input(
                "Max",
                min_value=smin_lt,
                max_value=smax_lt,
                step=0.01,
                format="%.2f",
                key="share_lt_max_input",
                on_change=_sync_share_lt_from_typing,
            )
        share_lt = st.slider(
            "Range",
            min_value=smin_lt,
            max_value=smax_lt,
            step=0.1,
            format="%.1f",
            key="share_lt_slider",
            on_change=_sync_share_lt_from_slider,
        )
        reset_keys.extend(
            ["share_lt_slider", "share_lt_min_input", "share_lt_max_input"]
        )

        st.markdown("**Share within health interest (%)**")
        h1, h2 = st.columns(2)
        with h1:
            st.number_input(
                "Min",
                min_value=smin_h,
                max_value=smax_h,
                step=0.01,
                format="%.2f",
                key="share_hi_min_input",
                on_change=_sync_share_hi_from_typing,
            )
        with h2:
            st.number_input(
                "Max",
                min_value=smin_h,
                max_value=smax_h,
                step=0.01,
                format="%.2f",
                key="share_hi_max_input",
                on_change=_sync_share_hi_from_typing,
            )
        share_hi = st.slider(
            "Range",
            min_value=smin_h,
            max_value=smax_h,
            step=0.1,
            format="%.1f",
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

    tab_table, tab_leaf, tab_hi, tab_matrix, tab_quadrant = st.tabs(
        [
            "Data table",
            "By lowest taxonomy",
            "By health interest",
            "Heatmap (top pairs)",
            "Condition focus VS Condition reliance",
        ]
    )

    with tab_table:
        st.subheader("Filtered rows")
        tab_table_df = _tab_exclude_expander("table", filtered)
        if tab_table_df.empty:
            st.warning("No rows left after sidebar filters and this tab's excludes.")
        sort_opt = st.selectbox("Sort by", NUMERIC_METRICS)
        ascending = st.checkbox("Ascending", value=False)
        show = tab_table_df.sort_values(sort_opt, ascending=ascending)
        st.dataframe(
            show,
            use_container_width=True,
            hide_index=True,
            height=480,
            column_config={
                "SHARE_WITHIN_LOWEST_TAXONOMY": st.column_config.NumberColumn(
                    "Share within lowest taxonomy (%)",
                    format="%.2f",
                ),
                "SHARE_WITHIN_HEALTH_INTEREST": st.column_config.NumberColumn(
                    "Share within health interest (%)",
                    format="%.2f",
                ),
            },
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
        st.caption(
            "Uses sidebar-filtered data. Leave **Lowest taxonomies** empty to rank "
            "interests across **all** rows; choose one or more taxonomies to narrow."
        )
        tab_leaf_df = _tab_exclude_expander("leaf", filtered)
        rank_metric = st.radio(
            "Rank bars by",
            NUMERIC_METRICS,
            horizontal=True,
            key="metric_leaf_tab",
        )
        leaf_opts = sorted_unique(tab_leaf_df[LEAF_COL]) if len(tab_leaf_df) else []
        sel_leaves_tab = st.multiselect(
            "Lowest taxonomies (empty = all)",
            options=leaf_opts,
            default=[],
            key="tab_pick_leaves",
        )
        top_n = st.number_input("Top N interests", 5, 100, 20, key="topn_leaf")
        base_leaf = (
            tab_leaf_df
            if not sel_leaves_tab
            else tab_leaf_df[tab_leaf_df[LEAF_COL].isin(sel_leaves_tab)]
        )
        if base_leaf.empty:
            st.info("No rows match the current sidebar and tab filters.")
        else:
            agg_leaf = _aggregate_for_bar_chart(base_leaf, HEALTH_COL, rank_metric)
            sub = agg_leaf.nlargest(int(top_n), rank_metric)
            if sub.empty:
                st.info("Nothing to plot for this selection.")
            else:
                fig = px.bar(
                    sub,
                    x=rank_metric,
                    y=HEALTH_COL,
                    orientation="h",
                    hover_data=[c for c in NUMERIC_METRICS if c in sub.columns],
                    labels={rank_metric: _metric_axis_label(rank_metric)},
                    color_discrete_sequence=[BAR_FILL],
                )
                n_bars = len(sub)
                bar_lbl = _bar_data_labels(sub[rank_metric], rank_metric)
                lbl_px = _bar_label_font_size(n_bars)
                fig.update_traces(
                    marker=dict(
                        color=BAR_FILL,
                        line=dict(width=0.5, color=CHART_TEXT),
                    ),
                    text=bar_lbl,
                    textposition="outside",
                    cliponaxis=False,
                    textfont=dict(family=FONT_FAMILY, color=CHART_TEXT, size=lbl_px),
                    outsidetextfont=dict(
                        family=FONT_FAMILY, color=CHART_TEXT, size=lbl_px
                    ),
                )
                fig.update_layout(
                    font=_plot_base_font(),
                    hoverlabel=dict(font=dict(family=FONT_FAMILY, size=13)),
                    yaxis={"categoryorder": "total ascending"},
                    height=max(400, 24 * n_bars),
                    margin=dict(l=200, r=120),
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
        st.caption(
            "Uses sidebar-filtered data. Leave **Health interests** empty to rank "
            "lowest taxonomies across **all** rows; choose one or more interests to narrow."
        )
        tab_hi_df = _tab_exclude_expander("hi", filtered)
        rank_metric_hi = st.radio(
            "Rank bars by",
            NUMERIC_METRICS,
            horizontal=True,
            key="metric_hi_tab",
        )
        hi_opts = sorted_unique(tab_hi_df[HEALTH_COL]) if len(tab_hi_df) else []
        sel_interests_tab = st.multiselect(
            "Health interests (empty = all)",
            options=hi_opts,
            default=[],
            key="tab_pick_interests",
        )
        top_n_hi = st.number_input("Top N lowest taxonomies", 5, 100, 20, key="topn_hi")
        base_hi = (
            tab_hi_df
            if not sel_interests_tab
            else tab_hi_df[tab_hi_df[HEALTH_COL].isin(sel_interests_tab)]
        )
        if base_hi.empty:
            st.info("No rows match the current sidebar and tab filters.")
        else:
            agg_hi = _aggregate_for_bar_chart(base_hi, LEAF_COL, rank_metric_hi)
            sub_hi = agg_hi.nlargest(int(top_n_hi), rank_metric_hi)
            if sub_hi.empty:
                st.info("Nothing to plot for this selection.")
            else:
                fig2 = px.bar(
                    sub_hi,
                    x=rank_metric_hi,
                    y=LEAF_COL,
                    orientation="h",
                    hover_data=[c for c in NUMERIC_METRICS if c in sub_hi.columns],
                    labels={rank_metric_hi: _metric_axis_label(rank_metric_hi)},
                    color_discrete_sequence=[BAR_FILL],
                )
                n_bars_hi = len(sub_hi)
                bar_lbl_hi = _bar_data_labels(sub_hi[rank_metric_hi], rank_metric_hi)
                lbl_px_hi = _bar_label_font_size(n_bars_hi)
                fig2.update_traces(
                    marker=dict(
                        color=BAR_FILL,
                        line=dict(width=0.5, color=CHART_TEXT),
                    ),
                    text=bar_lbl_hi,
                    textposition="outside",
                    cliponaxis=False,
                    textfont=dict(family=FONT_FAMILY, color=CHART_TEXT, size=lbl_px_hi),
                    outsidetextfont=dict(
                        family=FONT_FAMILY, color=CHART_TEXT, size=lbl_px_hi
                    ),
                )
                fig2.update_layout(
                    font=_plot_base_font(),
                    hoverlabel=dict(font=dict(family=FONT_FAMILY, size=13)),
                    yaxis={"categoryorder": "total ascending"},
                    height=max(400, 24 * n_bars_hi),
                    margin=dict(l=220, r=120),
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
            "are shown. Large heatmaps keep their pixel size—scroll horizontally "
            "or vertically in the chart area when it exceeds the viewport."
        )
        tab_matrix_df = _tab_exclude_expander("matrix", filtered)
        col_a, col_b, col_c = st.columns(3)
        with col_a:
            k_leaf = st.number_input(
                "Top lowest taxonomies (by total rec count)", 1, 80, 20
            )
        with col_b:
            k_hi = st.number_input("Top health interests", 1, 80, 20)
        with col_c:
            cell_metric = st.selectbox("Cell value", NUMERIC_METRICS)
        if tab_matrix_df.empty:
            st.warning("No data after sidebar filters and this tab's excludes.")
        else:
            leaf_totals = tab_matrix_df.groupby(LEAF_COL, as_index=False)[
                "REC_COUNT"
            ].sum()
            top_leaves = leaf_totals.nlargest(int(k_leaf), "REC_COUNT")[
                LEAF_COL
            ].tolist()
            hi_totals = tab_matrix_df.groupby(HEALTH_COL, as_index=False)[
                "REC_COUNT"
            ].sum()
            top_his = hi_totals.nlargest(int(k_hi), "REC_COUNT")[
                HEALTH_COL
            ].tolist()
            hm = tab_matrix_df[
                tab_matrix_df[LEAF_COL].isin(top_leaves)
                & tab_matrix_df[HEALTH_COL].isin(top_his)
            ]
            pivot = hm.pivot_table(
                index=LEAF_COL,
                columns=HEALTH_COL,
                values=cell_metric,
                aggfunc="max",
            )
            pivot = pivot.reindex(index=top_leaves, columns=top_his)
            hm_text = _heatmap_cell_labels(pivot, cell_metric)
            n_cols_hm = len(pivot.columns)
            n_rows = len(pivot.index)
            hm_lbl_px = _heatmap_label_font_size(n_rows, n_cols_hm)
            hm_z_fmt = (
                "%{z:,.0f}" if cell_metric == "REC_COUNT" else "%{z:.2f}%"
            )
            hm_hover = (
                f"%{{y}}<br>%{{x}}<br>{_metric_axis_label(cell_metric)}: {hm_z_fmt}<extra></extra>"
            )
            z_hm = pivot.values.astype(float)
            hm_text_colors = _heatmap_cell_text_colors(
                z_hm, HEATMAP_COLORSCALE
            )
            colorbar_cfg: dict = dict(
                title=dict(
                    text=_metric_axis_label(cell_metric),
                    font=dict(family=FONT_FAMILY, color=CHART_TEXT, size=12),
                ),
                tickfont=dict(family=FONT_FAMILY, color=CHART_TEXT),
            )
            if cell_metric in SHARE_COLS:
                colorbar_cfg["tickformat"] = ".1f"
                colorbar_cfg["ticksuffix"] = "%"
                colorbar_cfg["showticksuffix"] = "all"
            else:
                colorbar_cfg["tickformat"] = ",.0f"
            x_hm = pivot.columns.tolist()
            y_hm = pivot.index.tolist()
            fig_hm = go.Figure(
                data=go.Heatmap(
                    z=z_hm,
                    x=x_hm,
                    y=y_hm,
                    colorscale=HEATMAP_COLORSCALE,
                    hoverongaps=False,
                    hovertemplate=hm_hover,
                    colorbar=colorbar_cfg,
                )
            )
            hm_ann: list[dict] = []
            for i, yi in enumerate(y_hm):
                for j, xj in enumerate(x_hm):
                    label = hm_text[i][j]
                    if not label:
                        continue
                    hm_ann.append(
                        {
                            "xref": "x",
                            "yref": "y",
                            "x": xj,
                            "y": yi,
                            "text": label,
                            "showarrow": False,
                            "font": {
                                "family": FONT_FAMILY,
                                "size": hm_lbl_px,
                                "color": hm_text_colors[i][j],
                            },
                            "xanchor": "center",
                            "yanchor": "middle",
                        }
                    )
            fig_hm.update_layout(
                font=_plot_base_font(),
                annotations=hm_ann,
                hoverlabel=dict(font=dict(family=FONT_FAMILY, size=13)),
                xaxis=dict(
                    side="bottom",
                    tickangle=-45,
                    tickfont=_tick_font(),
                    title=_axis_title_font(),
                ),
                yaxis=dict(tickfont=_tick_font(), title=_axis_title_font()),
                width=_heatmap_figure_width(n_cols_hm),
                height=_heatmap_figure_height(n_rows),
                margin=dict(l=200, b=_heatmap_bottom_margin(n_cols_hm)),
            )
            _show_plotly_figure_scrollable(fig_hm)

    with tab_quadrant:
        st.subheader("Condition focus VS Condition reliance")
        st.caption(
            "**Horizontal:** how much the pair sits within its **health interest**. "
            "**Vertical:** how much it sits within its **product category** (lowest taxonomy). "
            "**(0, 0)** is what’s typical among the points on this chart. "
            "Leave **Health interests** empty to use every interest in the current slice."
        )
        tab_quad_df = _tab_exclude_expander("quadrant", filtered)
        if tab_quad_df.empty:
            st.warning("No data after sidebar filters and this tab's excludes.")
        else:
            hi_opts_q = sorted_unique(tab_quad_df[HEALTH_COL])
            sel_hi_q = st.multiselect(
                "Health interests (empty = all)",
                options=hi_opts_q,
                default=[],
                key="quad_sel_interests",
                help="Restrict rows to these interests before choosing Top N taxonomies.",
            )
            n_leaf_unique = max(1, len(sorted_unique(tab_quad_df[LEAF_COL])))
            top_n_max = n_leaf_unique
            quad_top_n_default = min(20, n_leaf_unique)
            _qnk = _QUADRANT_TOP_N_KEY
            if _qnk not in st.session_state:
                st.session_state[_qnk] = quad_top_n_default
            else:
                try:
                    cur = int(st.session_state[_qnk])
                    if cur > top_n_max:
                        st.session_state[_qnk] = top_n_max
                    elif cur < 1:
                        st.session_state[_qnk] = 1
                except (TypeError, ValueError):
                    st.session_state[_qnk] = quad_top_n_default
            top_n_quad = st.number_input(
                "Top N lowest taxonomies (by total rec count in slice)",
                min_value=1,
                max_value=top_n_max,
                key=_qnk,
                help="Plotted points are rows for those taxonomies (after interest filter). "
                f"Maximum is {top_n_max} (taxonomies present in this slice).",
            )
            plot_df, med_hi, med_lt = _quadrant_plot_frame(
                tab_quad_df, sel_hi_q, int(top_n_quad)
            )
            if plot_df.empty:
                st.info("No rows for this selection after filters.")
            else:
                st.caption(
                    f"**Typical raw shares** for this plot (%, file values): "
                    f"within interest **{med_hi:.2f}**, within category **{med_lt:.2f}**."
                )
                _pair_rows = (
                    plot_df[[LEAF_COL, HEALTH_COL]]
                    .drop_duplicates()
                    .sort_values([LEAF_COL, HEALTH_COL], kind="mergesort")
                )
                _pair_keys: list[str] = []
                _pair_labels: dict[str, str] = {}
                for _, _pr in _pair_rows.iterrows():
                    _pk = f"{_pr[LEAF_COL]}{_QUAD_PAIR_KEY_SEP}{_pr[HEALTH_COL]}"
                    _pair_keys.append(_pk)
                    _pair_labels[_pk] = f"{_pr[LEAF_COL]} — {_pr[HEALTH_COL]}"
                _quad_show_lbl = st.checkbox(
                    "Show point labels (lowest taxonomy)",
                    value=True,
                    key="quad_show_point_labels",
                )
                _quad_hidden_pairs = st.multiselect(
                    "Hide labels for these taxonomy × health interest pairs",
                    options=_pair_keys,
                    format_func=lambda k: _pair_labels[k],
                    default=[],
                    key="quad_hidden_label_pairs",
                    disabled=not _quad_show_lbl,
                    help="Select pairs to drop the text label only; the marker stays. "
                    "Options match the current plotted slice.",
                )
                x_col = "ln(interest share+1) − median"
                y_col = "ln(taxonomy share+1) − median"
                plot_show = plot_df.rename(
                    columns={"x_vs_median_log": x_col, "y_vs_median_log": y_col}
                )
                _lbl_col = "quad_label_text"
                _row_keys = (
                    plot_show[LEAF_COL].astype(str)
                    + _QUAD_PAIR_KEY_SEP
                    + plot_show[HEALTH_COL].astype(str)
                )
                _hidden_set = frozenset(_quad_hidden_pairs)
                if _quad_show_lbl:
                    plot_show[_lbl_col] = np.where(
                        _row_keys.isin(_hidden_set),
                        "",
                        plot_show[LEAF_COL].astype(str),
                    )
                else:
                    plot_show[_lbl_col] = ""
                fig_q = px.scatter(
                    plot_show,
                    x=x_col,
                    y=y_col,
                    size="REC_COUNT",
                    color=HEALTH_COL,
                    text=_lbl_col,
                    custom_data=[LEAF_COL, HEALTH_COL],
                    labels={
                        x_col: "Condition focus (vs typical on chart)",
                        y_col: "Condition reliance (vs typical on chart)",
                    },
                    opacity=0.65,
                )
                fig_q.update_traces(
                    mode="markers+text",
                    textposition="top center",
                    textfont=dict(family=FONT_FAMILY, size=9, color=CHART_TEXT),
                    marker=dict(line=dict(width=0.5, color="DarkSlateGrey")),
                    hovertemplate=(
                        "Taxonomy: <b>%{customdata[0]}</b><br>"
                        "Health interest: <b>%{customdata[1]}</b><extra></extra>"
                    ),
                )
                fig_q.add_hline(
                    y=0,
                    line_width=1.5,
                    line_dash="solid",
                    line_color=CHART_TEXT,
                    opacity=0.55,
                )
                fig_q.add_vline(
                    x=0,
                    line_width=1.5,
                    line_dash="solid",
                    line_color=CHART_TEXT,
                    opacity=0.55,
                )
                x_half = _quadrant_axis_half_from_series(
                    plot_show[x_col],
                    min_pts_pad=0.08,
                    min_half=0.02,
                )
                y_half = _quadrant_axis_half_from_series(
                    plot_show[y_col],
                    min_pts_pad=0.08,
                    min_half=0.02,
                )
                xr0, xr1 = -x_half, x_half
                yr0, yr1 = -y_half, y_half
                fig_q.update_layout(
                    font=_plot_base_font(),
                    hoverlabel=dict(font=dict(family=FONT_FAMILY, size=13)),
                    height=720,
                    margin=dict(l=96, r=96, t=100, b=100),
                    annotations=_quadrant_label_annotations(),
                    xaxis=dict(range=[xr0, xr1], zeroline=False),
                    yaxis=dict(range=[yr0, yr1], zeroline=False),
                    legend=dict(
                        title=dict(text="Health interest"),
                        font=dict(family=FONT_FAMILY, color=CHART_TEXT),
                    ),
                )
                fig_q.update_xaxes(tickfont=_tick_font(), title="")
                fig_q.update_yaxes(tickfont=_tick_font(), title="")
                st.plotly_chart(fig_q, use_container_width=True)


if __name__ == "__main__":
    main()

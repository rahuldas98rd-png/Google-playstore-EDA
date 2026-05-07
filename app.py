import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
import joblib

# ─────────────────────────────────────────────
#  Page config  (must be FIRST Streamlit call)
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Google Play Store · EDA Dashboard",
    page_icon="📱",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
#  Custom CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
/* ── Google Font ─────────────────────────────*/
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600;700&family=DM+Mono:wght@400;500&display=swap');

html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }

/* ── Main background ─────────────────────────*/
.stApp { background: #f4f6fb; }

/* ── KPI cards ───────────────────────────────*/
.kpi-card {
    background: linear-gradient(135deg, #ffffff 0%, #f0f4ff 100%);
    border: 1px solid #d1d9f0;
    border-radius: 14px;
    padding: 22px 26px;
    text-align: center;
    transition: transform .2s, border-color .2s;
}
.kpi-card:hover { transform: translateY(-3px); border-color: #3b6bff; box-shadow: 0 4px 18px rgba(59,107,255,.12); }
.kpi-label { font-size: 12px; font-weight: 600; letter-spacing: 1.4px;
             text-transform: uppercase; color: #64748b; margin-bottom: 6px; }
.kpi-value { font-size: 32px; font-weight: 700; color: #0f172a; line-height: 1.1; }
.kpi-delta { font-size: 12px; color: #3b6bff; margin-top: 4px; }

/* ── Section headers ─────────────────────────*/
.section-title {
    font-size: 18px; font-weight: 600; color: #1e293b;
    letter-spacing: .5px; margin: 32px 0 14px;
    padding-bottom: 8px;
    border-bottom: 2px solid #dde3f5;
}

/* ── Sidebar ─────────────────────────────────*/
[data-testid="stSidebar"] {
    background: #ffffff !important;
    border-right: 1px solid #e2e8f0;
}
[data-testid="stSidebar"] * { color: #1e293b !important; }

/* ── Expander ────────────────────────────────*/
.streamlit-expanderHeader {
    background: #f8faff !important;
    border-radius: 10px !important;
    font-weight: 500;
}

/* ── Prediction panel ────────────────────────*/
.predict-box {
    background: linear-gradient(135deg, #ffffff, #f0f4ff);
    border: 1px solid #d1d9f0;
    border-radius: 14px;
    padding: 28px;
}

/* ── Scrollable table ────────────────────────*/
.stDataFrame { border-radius: 10px; overflow: hidden; }

/* ── Tab styling ─────────────────────────────*/
.stTabs [data-baseweb="tab-list"] {
    gap: 6px;
    background: #e8edf8;
    border-radius: 10px;
    padding: 4px;
}
.stTabs [data-baseweb="tab"] {
    border-radius: 8px;
    color: #475569;
    font-weight: 500;
    padding: 8px 18px;
}
.stTabs [aria-selected="true"] {
    background: #3b6bff !important;
    color: #fff !important;
}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
#  Paths
# ─────────────────────────────────────────────
DATA_PATH      = "data/raw/googleplaystore.csv"
OUTPUT_PATH    = "data/processed"
CLEAN_FILE     = os.path.join(OUTPUT_PATH, "clean.csv")
PROCESSED_FILE = os.path.join(OUTPUT_PATH, "processed.csv")
MODEL_DIR      = "artifacts/models"

# ─────────────────────────────────────────────
#  Helpers
# ─────────────────────────────────────────────
# Only keys that px.* functions actually accept as kwargs
PLOTLY_PX = dict(template="plotly_white")

# Layout-level keys applied via fig.update_layout()
PLOTLY_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="DM Sans", color="#1e293b"),
    margin=dict(l=20, r=20, t=40, b=20),
)


def theme(fig, **extra):
    """Apply shared layout; extra kwargs override PLOTLY_LAYOUT defaults."""
    merged = {**PLOTLY_LAYOUT, **extra}
    fig.update_layout(**merged)
    return fig

PALETTE = px.colors.qualitative.Bold


def fmt_number(n):
    if n >= 1_000_000_000:
        return f"{n/1_000_000_000:.1f}B"
    if n >= 1_000_000:
        return f"{n/1_000_000:.1f}M"
    if n >= 1_000:
        return f"{n/1_000:.1f}K"
    return str(int(n))


# ─────────────────────────────────────────────
#  Data loading
# ─────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def load_raw(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    return df


@st.cache_data(show_spinner=False)
def load_clean(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


@st.cache_data(show_spinner=False)
def prepare_data(df_raw: pd.DataFrame) -> pd.DataFrame:
    """Clean & engineer features on the fly from the raw CSV."""
    df = df_raw.copy()

    # ── Remove bad rows ──────────────────────────
    df = df[df["Category"] != "1.9"]

    # ── Deduplicate ──────────────────────────────
    df = df.drop_duplicates(subset="App", keep="first")

    # ── Reviews ─────────────────────────────────
    df["Reviews"] = pd.to_numeric(df["Reviews"], errors="coerce")

    # ── Installs ─────────────────────────────────
    df["Installs"] = (
        df["Installs"]
        .str.replace(r"[+,]", "", regex=True)
        .pipe(pd.to_numeric, errors="coerce")
    )

    # ── Price ────────────────────────────────────
    df["Price"] = (
        df["Price"]
        .str.replace("$", "", regex=False)
        .pipe(pd.to_numeric, errors="coerce")
        .fillna(0)
    )

    # ── Rating ───────────────────────────────────
    df["Rating"] = pd.to_numeric(df["Rating"], errors="coerce")
    df["Rating"] = df["Rating"].fillna(df["Rating"].mean())

    # ── Size ─────────────────────────────────────
    def parse_size(s):
        if pd.isna(s) or s == "Varies with device":
            return np.nan
        if "M" in str(s):
            return float(str(s).replace("M", ""))
        if "k" in str(s):
            return float(str(s).replace("k", "")) / 1024
        return np.nan

    df["Size_MB"] = df["Size"].apply(parse_size)

    # ── Dates ────────────────────────────────────
    df["Last Updated"] = pd.to_datetime(df["Last Updated"], format="mixed", errors="coerce")
    df["Year_updated"]  = df["Last Updated"].dt.year
    df["Month_updated"] = df["Last Updated"].dt.month

    # ── Derived ──────────────────────────────────
    df["Is_Free"]     = df["Type"] == "Free"
    df["Success"]     = df["Installs"] > 100_000
    df["Rating_Band"] = pd.cut(
        df["Rating"],
        bins=[0, 2, 3, 4, 4.5, 5],
        labels=["< 2", "2–3", "3–4", "4–4.5", "4.5–5"],
    )

    return df


@st.cache_resource(show_spinner=False)
def load_models():
    """Load models; silently retrain if pickles are from a different sklearn version."""
    import warnings
    from sklearn.exceptions import InconsistentVersionWarning

    reg_path = os.path.join(MODEL_DIR, "rating_model.pkl")
    clf_path = os.path.join(MODEL_DIR, "success_model.pkl")

    if not (os.path.exists(reg_path) and os.path.exists(clf_path)):
        return None, None

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", InconsistentVersionWarning)
        reg = joblib.load(reg_path)
        clf = joblib.load(clf_path)

    if any(issubclass(w.category, InconsistentVersionWarning) for w in caught):
        # Stale pickles — retrain from the cleaned dataframe in memory
        reg, clf = _retrain_and_save(reg_path, clf_path)

    return reg, clf


def _retrain_and_save(reg_path: str, clf_path: str):
    """Retrain both models and overwrite the stale pickle files."""
    from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
    from sklearn.model_selection import train_test_split

    try:
        src = CLEAN_FILE if os.path.exists(CLEAN_FILE) else DATA_PATH
        frame = pd.read_csv(src)
        frame["Installs"] = pd.to_numeric(
            frame["Installs"].astype(str).str.replace(r"[+,]", "", regex=True),
            errors="coerce",
        )
        frame["Reviews"] = pd.to_numeric(frame["Reviews"], errors="coerce")
        frame = frame.dropna(subset=["Reviews", "Installs", "Rating"])
        X = frame[["Reviews", "Installs"]]
        y_reg = frame["Rating"]
        y_clf = (frame["Installs"] > 100_000).astype(int)
        X_tr, _, y_tr, _ = train_test_split(X, y_reg, test_size=0.2, random_state=42)
        reg = RandomForestRegressor(n_estimators=100, random_state=42)
        reg.fit(X_tr, y_tr)
        clf = RandomForestClassifier(n_estimators=100, random_state=42)
        clf.fit(X, y_clf)
        os.makedirs(os.path.dirname(reg_path), exist_ok=True)
        joblib.dump(reg, reg_path)
        joblib.dump(clf, clf_path)
    except Exception:
        pass  # Return whatever was loaded; prediction tab will still work
    return reg, clf


# ─────────────────────────────────────────────
#  Load data
# ─────────────────────────────────────────────
with st.spinner("Loading dataset…"):
    try:
        df_raw = load_raw(DATA_PATH)
        df = prepare_data(df_raw)
    except FileNotFoundError:
        st.error(
            "Dataset not found at `data/raw/googleplaystore.csv`. "
            "Please ensure the file exists before launching the app."
        )
        st.stop()

# ─────────────────────────────────────────────
#  Sidebar
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 📱 Play Store EDA")
    st.markdown("---")

    st.markdown("### Filters")

    all_categories = sorted(df["Category"].dropna().unique())
    selected_cats = st.multiselect(
        "Category",
        options=all_categories,
        default=all_categories,
        help="Leave empty to select all",
    )

    content_ratings = sorted(df["Content Rating"].dropna().unique())
    selected_content = st.multiselect(
        "Content Rating",
        options=content_ratings,
        default=content_ratings,
    )

    app_type = st.radio("App Type", ["All", "Free", "Paid"], horizontal=True)

    rating_range = st.slider(
        "Minimum Rating",
        min_value=0.0, max_value=5.0, value=1.0, step=0.1,
    )

    top_n = st.slider("Top N for charts", 5, 20, 10)

    st.markdown("---")
    st.markdown("### Pipeline")
    if st.button("🔄 Re-run Pipeline", use_container_width=True):
        try:
            from pipeline.pipeline import DataPipeline
            DataPipeline(DATA_PATH, OUTPUT_PATH).run()
            st.success("✅ Pipeline ran successfully!")
            st.cache_data.clear()
        except ModuleNotFoundError as e:
            missing = e.name or str(e)
            st.error(
                f"Missing dependency: **{missing}**. "
                f"Run `pip install {missing}` in your venv and restart the app."
            )
        except Exception as e:
            st.error(f"Pipeline failed: {e}")

    st.markdown("---")
    st.caption(f"Dataset: **{len(df):,}** apps · **{df['Category'].nunique()}** categories")
    st.caption("💡 Clear category filter to show all apps.")


# ─────────────────────────────────────────────
#  Apply filters
# ─────────────────────────────────────────────
# Short-circuit: if all options selected, skip the isin() filter (faster)
_all_cats     = set(selected_cats) == set(all_categories) or not selected_cats
_all_content  = set(selected_content) == set(content_ratings) or not selected_content

mask = pd.Series(True, index=df.index)
if not _all_cats:
    mask &= df["Category"].isin(selected_cats)
if not _all_content:
    mask &= df["Content Rating"].isin(selected_content)
mask &= (df["Rating"] >= rating_range)
if app_type == "Free":
    mask &= df["Is_Free"]
elif app_type == "Paid":
    mask &= ~df["Is_Free"]

dff = df[mask].copy()

# ─────────────────────────────────────────────
#  Header
# ─────────────────────────────────────────────
st.markdown(
    "<h1 style='color:#0f172a;font-weight:700;margin-bottom:4px;'>"
    "📱 Google Play Store Analytics</h1>"
    "<p style='color:#64748b;margin-bottom:24px;'>"
    "Exploratory data analysis across 10,000+ Android apps · 34 categories</p>",
    unsafe_allow_html=True,
)

# ─────────────────────────────────────────────
#  KPI row
# ─────────────────────────────────────────────
k1, k2, k3, k4, k5 = st.columns(5)

kpis = [
    (k1, "Total Apps",       fmt_number(len(dff)),                          "in selection"),
    (k2, "Avg Rating",       f"{dff['Rating'].mean():.2f} ★",               "out of 5.0"),
    (k3, "Total Installs",   fmt_number(dff["Installs"].sum()),              "combined"),
    (k4, "Free Apps",        f"{dff['Is_Free'].mean()*100:.1f}%",           "of selection"),
    (k5, "Median Reviews",   fmt_number(dff["Reviews"].median()),           "per app"),
]

for col, label, value, delta in kpis:
    col.markdown(
        f"""<div class="kpi-card">
              <div class="kpi-label">{label}</div>
              <div class="kpi-value">{value}</div>
              <div class="kpi-delta">{delta}</div>
            </div>""",
        unsafe_allow_html=True,
    )

st.markdown("<br>", unsafe_allow_html=True)

# ─────────────────────────────────────────────
#  Tabs
# ─────────────────────────────────────────────
tab_overview, tab_installs, tab_ratings, tab_missing, tab_predict, tab_data = st.tabs([
    "📊 Overview",
    "📥 Installs",
    "⭐ Ratings",
    "🔍 Data Quality",
    "🤖 Predict",
    "🗃 Raw Data",
])


# ═══════════════════════════════════════════════
#  TAB 1 — OVERVIEW
# ═══════════════════════════════════════════════
with tab_overview:

    # ── Row 1: Popular categories bar + Free/Paid donut ──
    col_a, col_b = st.columns([3, 2])

    with col_a:
        st.markdown('<div class="section-title">Top Categories by App Count</div>', unsafe_allow_html=True)
        cat_counts = (
            dff["Category"].value_counts().head(top_n).reset_index()
        )
        cat_counts.columns = ["Category", "Count"]
        cat_counts["Pct"] = (cat_counts["Count"] / cat_counts["Count"].sum() * 100).round(2)

        fig = px.bar(
            cat_counts,
            x="Count", y="Category",
            orientation="h",
            text=cat_counts["Pct"].map(lambda x: f"{x}%"),
            color="Count",
            color_continuous_scale="Blues",
            labels={"Count": "Number of Apps", "Category": ""},
            **PLOTLY_PX,
        )
        theme(fig)
        fig.update_traces(textposition="outside", textfont_size=11)
        fig.update_layout(showlegend=False, coloraxis_showscale=False, height=420)
        fig.update_yaxes(categoryorder="total ascending")
        st.plotly_chart(fig, width='stretch')

    with col_b:
        st.markdown('<div class="section-title">Free vs Paid Split</div>', unsafe_allow_html=True)
        type_counts = dff["Type"].value_counts().reset_index()
        type_counts.columns = ["Type", "Count"]
        fig2 = px.pie(
            type_counts,
            names="Type",
            values="Count",
            hole=0.55,
            color_discrete_sequence=["#3b6bff", "#f87171"],
            **PLOTLY_PX,
        )
        theme(fig2)
        fig2.update_traces(textinfo="percent+label", textfont_size=13)
        fig2.update_layout(height=420, showlegend=True)
        st.plotly_chart(fig2, width='stretch')

    # ── Row 2: Content Rating + App Type counts ──
    col_c, col_d = st.columns(2)

    with col_c:
        st.markdown('<div class="section-title">Content Rating Distribution</div>', unsafe_allow_html=True)
        cr = dff["Content Rating"].value_counts().reset_index()
        cr.columns = ["Content Rating", "Count"]
        fig3 = px.bar(
            cr, x="Content Rating", y="Count",
            color="Content Rating",
            color_discrete_sequence=PALETTE,
            text="Count",
            **PLOTLY_PX,
        )
        theme(fig3)
        fig3.update_traces(textposition="outside")
        fig3.update_layout(showlegend=False, height=320)
        st.plotly_chart(fig3, width='stretch')

    with col_d:
        st.markdown('<div class="section-title">Most Popular Categories (Pie)</div>', unsafe_allow_html=True)
        cat_top = dff["Category"].value_counts().head(top_n).reset_index()
        cat_top.columns = ["Category", "Count"]
        fig4 = px.pie(
            cat_top, names="Category", values="Count",
            color_discrete_sequence=PALETTE,
            **PLOTLY_PX,
        )
        theme(fig4)
        fig4.update_traces(textinfo="percent+label", textfont_size=10)
        fig4.update_layout(height=320, showlegend=False)
        st.plotly_chart(fig4, width='stretch')


# ═══════════════════════════════════════════════
#  TAB 2 — INSTALLS
# ═══════════════════════════════════════════════
with tab_installs:

    # ── Sunburst: App Installs by Category + App ──
    st.markdown('<div class="section-title">App Installs by Category (Sunburst)</div>', unsafe_allow_html=True)

    sun_df = (
        dff[["Category", "App", "Installs"]]
        .dropna()
        .sort_values("Installs", ascending=False)
    )
    # Keep top 3 apps per category — restore Category from group index after apply
    top_apps = (
        sun_df.groupby("Category")[["App", "Installs"]]
        .apply(lambda g: g.nlargest(3, "Installs"))
        .reset_index(level=0)          # brings Category back as a column
        .reset_index(drop=True)
    )
    fig_sun = px.sunburst(
        top_apps,
        path=["Category", "App"],
        values="Installs",
        color="Installs",
        color_continuous_scale="Blues",
        **PLOTLY_PX,
    )
    theme(fig_sun)
    fig_sun.update_layout(height=600, coloraxis_showscale=False)
    fig_sun.update_traces(textfont_size=12)
    st.plotly_chart(fig_sun, width='stretch')

    col_e, col_f = st.columns(2)

    with col_e:
        st.markdown('<div class="section-title">Most Installed Categories (Total)</div>', unsafe_allow_html=True)
        inst_cat = (
            dff.groupby("Category")["Installs"]
            .sum()
            .nlargest(top_n)
            .reset_index()
        )
        inst_cat.columns = ["Category", "Total Installs"]
        inst_cat["Label"] = inst_cat["Total Installs"].apply(fmt_number)

        fig5 = px.pie(
            inst_cat,
            names="Category",
            values="Total Installs",
            hole=0.4,
            color_discrete_sequence=PALETTE,
            **PLOTLY_PX,
        )
        theme(fig5)
        fig5.update_traces(textinfo="percent+label", textfont_size=11)
        fig5.update_layout(height=400, showlegend=False)
        st.plotly_chart(fig5, width='stretch')

    with col_f:
        st.markdown('<div class="section-title">Top 10 Most Installed Apps</div>', unsafe_allow_html=True)
        top_apps_bar = dff.nlargest(10, "Installs")[["App", "Installs", "Category"]]
        fig6 = px.bar(
            top_apps_bar,
            x="Installs",
            y="App",
            orientation="h",
            color="Category",
            color_discrete_sequence=PALETTE,
            text=top_apps_bar["Installs"].apply(fmt_number),
            **PLOTLY_PX,
        )
        theme(fig6)
        fig6.update_traces(textposition="outside", textfont_size=10)
        fig6.update_layout(showlegend=False, height=400)
        fig6.update_yaxes(categoryorder="total ascending")
        st.plotly_chart(fig6, width='stretch')

    # ── Installs distribution per category (box) ──
    st.markdown('<div class="section-title">Install Distribution per Category (Log Scale)</div>', unsafe_allow_html=True)
    box_df = dff[dff["Installs"] > 0].copy()
    fig7 = px.box(
        box_df,
        x="Category",
        y="Installs",
        log_y=True,
        color="Category",
        color_discrete_sequence=PALETTE,
        **PLOTLY_PX,
    )
    theme(fig7)
    fig7.update_layout(
        showlegend=False, height=420,
        xaxis=dict(tickangle=-40, tickfont_size=10),
    )
    st.plotly_chart(fig7, width='stretch')


# ═══════════════════════════════════════════════
#  TAB 3 — RATINGS
# ═══════════════════════════════════════════════
with tab_ratings:

    col_g, col_h = st.columns(2)

    with col_g:
        st.markdown('<div class="section-title">Rating Distribution (KDE)</div>', unsafe_allow_html=True)
        fig_kde = px.histogram(
            dff.dropna(subset=["Rating"]),
            x="Rating",
            nbins=40,
            marginal="violin",
            color_discrete_sequence=["#3b6bff"],
            opacity=0.8,
            **PLOTLY_PX,
        )
        theme(fig_kde)
        fig_kde.add_vline(
            x=dff["Rating"].mean(),
            line_dash="dash", line_color="#ef4444",
            annotation_text=f"Mean {dff['Rating'].mean():.2f}",
            annotation_position="top right",
        )
        fig_kde.update_layout(height=380)
        st.plotly_chart(fig_kde, width='stretch')

    with col_h:
        st.markdown('<div class="section-title">Avg Rating by Category</div>', unsafe_allow_html=True)
        avg_rat = (
            dff.groupby("Category")["Rating"]
            .mean()
            .sort_values(ascending=False)
            .head(top_n)
            .reset_index()
        )
        fig_rat = px.bar(
            avg_rat,
            x="Rating", y="Category",
            orientation="h",
            color="Rating",
            color_continuous_scale="RdYlGn",
            text=avg_rat["Rating"].round(2),
            range_x=[3.5, 5.0],
            **PLOTLY_PX,
        )
        theme(fig_rat)
        fig_rat.update_traces(textposition="outside")
        fig_rat.update_layout(showlegend=False, coloraxis_showscale=False, height=380)
        fig_rat.update_yaxes(categoryorder="total ascending")
        st.plotly_chart(fig_rat, width='stretch')

    # ── Numerical univariate ──
    st.markdown('<div class="section-title">Univariate Analysis — Numerical Features</div>', unsafe_allow_html=True)
    num_cols = ["Rating", "Reviews", "Size_MB", "Installs", "Price",
                "Day_updated", "Month_updated", "Year_updated"]
    num_cols = [c for c in num_cols if c in dff.columns]

    rows = (len(num_cols) + 3) // 4
    fig_uni = make_subplots(
        rows=rows, cols=4,
        subplot_titles=num_cols,
        vertical_spacing=0.12,
        horizontal_spacing=0.06,
    )
    for idx, col in enumerate(num_cols):
        r, c = divmod(idx, 4)
        series = dff[col].dropna()
        if series.empty:
            continue
        fig_uni.add_trace(
            go.Histogram(
                x=series, name=col,
                marker_color="#3b6bff",
                opacity=0.8,
                showlegend=False,
            ),
            row=r + 1, col=c + 1,
        )
    theme(fig_uni, height=rows * 220, showlegend=False,
          margin=dict(l=20, r=20, t=50, b=20))
    st.plotly_chart(fig_uni, width='stretch')

    # ── Categorical univariate ──
    st.markdown('<div class="section-title">Univariate Analysis — Categorical Features</div>', unsafe_allow_html=True)
    cat_features = ["Type", "Content Rating"]
    fig_cat = make_subplots(
        rows=1, cols=len(cat_features),
        subplot_titles=cat_features,
        horizontal_spacing=0.10,
    )
    colors = ["#4f7cff", "#ff6b6b", "#34d399", "#fbbf24", "#a78bfa", "#f97316"]
    for idx, col in enumerate(cat_features):
        vc = dff[col].value_counts().reset_index()
        vc.columns = [col, "Count"]
        for i, row in vc.iterrows():
            fig_cat.add_trace(
                go.Bar(
                    x=[row[col]], y=[row["Count"]],
                    name=str(row[col]),
                    marker_color=colors[i % len(colors)],
                    showlegend=False,
                    text=[row["Count"]],
                    textposition="outside",
                ),
                row=1, col=idx + 1,
            )
    theme(fig_cat, height=360, barmode="group",
          margin=dict(l=20, r=20, t=50, b=20))
    st.plotly_chart(fig_cat, width='stretch')

    # ── Rating vs Installs scatter ──
    st.markdown('<div class="section-title">Rating vs Installs (coloured by Category)</div>', unsafe_allow_html=True)
    scat_df = dff[dff["Installs"] > 0].dropna(subset=["Rating", "Installs"]).copy()
    scat_df["Installs_log"] = np.log10(scat_df["Installs"])
    fig_scat = px.scatter(
        scat_df.sample(min(3000, len(scat_df)), random_state=42),
        x="Rating", y="Installs_log",
        color="Category",
        size="Reviews",
        size_max=18,
        hover_data=["App", "Installs"],
        color_discrete_sequence=PALETTE,
        labels={"Installs_log": "log₁₀(Installs)"},
        opacity=0.65,
        **PLOTLY_PX,
    )
    theme(fig_scat)
    fig_scat.update_layout(height=440, showlegend=True)
    st.plotly_chart(fig_scat, width='stretch')


# ═══════════════════════════════════════════════
#  TAB 4 — DATA QUALITY
# ═══════════════════════════════════════════════
with tab_missing:

    st.markdown('<div class="section-title">Missing Value Analysis</div>', unsafe_allow_html=True)

    raw_missing = df_raw.isnull().sum().reset_index()
    raw_missing.columns = ["Feature", "Missing Count"]
    raw_missing["Missing %"] = (raw_missing["Missing Count"] / len(df_raw) * 100).round(2)

    fig_miss = px.bar(
        raw_missing,
        x="Feature", y="Missing Count",
        text="Missing Count",
        color="Missing Count",
        color_continuous_scale=["#dde3f5", "#ef4444"],
        **PLOTLY_PX,
    )
    theme(fig_miss)
    fig_miss.update_traces(textposition="outside")
    fig_miss.update_layout(showlegend=False, coloraxis_showscale=False, height=380)
    st.plotly_chart(fig_miss, width='stretch')

    st.markdown('<div class="section-title">Missing Value Summary Table</div>', unsafe_allow_html=True)
    st.dataframe(
        raw_missing.style
            .background_gradient(subset=["Missing Count"], cmap="Reds")
            .format({"Missing %": "{:.2f}%"}),
        use_container_width=True,
        height=350,
    )

    col_i, col_j = st.columns(2)

    with col_i:
        st.markdown('<div class="section-title">Duplicate Apps</div>', unsafe_allow_html=True)
        n_raw = len(df_raw)
        n_dedup = df["App"].nunique()
        dupes = n_raw - n_dedup
        fig_dup = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=dupes,
            delta={"reference": n_raw, "valueformat": "d", "suffix": " removed"},
            title={"text": "Duplicate Records", "font": {"color": "#1e293b", "size": 14}},
            gauge={
                "axis": {"range": [0, n_raw], "tickcolor": "#94a3b8"},
                "bar": {"color": "#3b6bff"},
                "bgcolor": "#f0f4ff",
                "bordercolor": "#d1d9f0",
                "steps": [
                    {"range": [0, n_raw * 0.05], "color": "#e8edf8"},
                    {"range": [n_raw * 0.05, n_raw * 0.15], "color": "#f0f4ff"},
                ],
            },
            number={"font": {"color": "#0f172a", "size": 36}},
        ))
        theme(fig_dup, height=300, margin=dict(l=30, r=30, t=40, b=20))
        st.plotly_chart(fig_dup, width='stretch')

    with col_j:
        st.markdown('<div class="section-title">Data Type Overview</div>', unsafe_allow_html=True)
        dtype_df = pd.DataFrame({
            "Column": df_raw.columns,
            "Raw dtype": df_raw.dtypes.astype(str).values,
            "Non-null": df_raw.notnull().sum().values,
            "Null": df_raw.isnull().sum().values,
            "Unique": df_raw.nunique().values,
        })
        st.dataframe(dtype_df, use_container_width=True, height=300)


# ═══════════════════════════════════════════════
#  TAB 5 — PREDICT
# ═══════════════════════════════════════════════
with tab_predict:

    st.markdown('<div class="section-title">App Performance Predictor</div>', unsafe_allow_html=True)
    st.markdown(
        "<p style='color:#64748b;margin-bottom:20px;'>"
        "Enter estimated reviews and installs to predict a rating and commercial success "
        "using the trained RandomForest models.</p>",
        unsafe_allow_html=True,
    )

    reg_model, clf_model = load_models()

    if reg_model is None:
        st.warning(
            "⚠️ Trained models not found in `artifacts/models/`. "
            "Run the pipeline from the sidebar to train them first."
        )
    else:
        col_p1, col_p2, col_p3 = st.columns([2, 2, 3])

        with col_p1:
            reviews_input = st.number_input(
                "Number of Reviews", min_value=0, value=10_000, step=1_000,
            )
        with col_p2:
            installs_input = st.number_input(
                "Number of Installs", min_value=0, value=500_000, step=10_000,
            )
        with col_p3:
            st.markdown("<br>", unsafe_allow_html=True)
            predict_btn = st.button("🔮 Predict", use_container_width=True, type="primary")

        if predict_btn:
            X_input = [[reviews_input, installs_input]]
            pred_rating  = reg_model.predict(X_input)[0]
            pred_success = clf_model.predict(X_input)[0]

            r1, r2, r3 = st.columns(3)
            r1.markdown(
                f"""<div class="kpi-card">
                      <div class="kpi-label">Predicted Rating</div>
                      <div class="kpi-value">{pred_rating:.2f} ★</div>
                    </div>""",
                unsafe_allow_html=True,
            )
            r2.markdown(
                f"""<div class="kpi-card">
                      <div class="kpi-label">Success Prediction</div>
                      <div class="kpi-value">{"✅ Hit" if pred_success else "⚠️ Niche"}</div>
                      <div class="kpi-delta">{'Installs > 100K expected' if pred_success else 'Installs ≤ 100K expected'}</div>
                    </div>""",
                unsafe_allow_html=True,
            )
            r3.markdown(
                f"""<div class="kpi-card">
                      <div class="kpi-label">Input Reviews</div>
                      <div class="kpi-value">{fmt_number(reviews_input)}</div>
                      <div class="kpi-delta">{fmt_number(installs_input)} installs provided</div>
                    </div>""",
                unsafe_allow_html=True,
            )

        # ── Feature importance (if accessible) ──
        st.markdown('<div class="section-title">Model Feature Importance</div>', unsafe_allow_html=True)
        features = ["Reviews", "Installs"]
        try:
            imp = reg_model.feature_importances_
            fi_df = pd.DataFrame({"Feature": features, "Importance": imp})
            fig_fi = px.bar(
                fi_df, x="Importance", y="Feature",
                orientation="h",
                color="Importance",
                color_continuous_scale="Blues",
                text=fi_df["Importance"].round(3),
                **PLOTLY_PX,
            )
            theme(fig_fi)
            fig_fi.update_layout(
                showlegend=False, coloraxis_showscale=False, height=240,
            )
            fig_fi.update_traces(textposition="outside")
            st.plotly_chart(fig_fi, width='stretch')
        except Exception:
            st.info("Feature importance not available for this model.")


# ═══════════════════════════════════════════════
#  TAB 6 — RAW DATA
# ═══════════════════════════════════════════════
with tab_data:
    st.markdown('<div class="section-title">Filtered Dataset Preview</div>', unsafe_allow_html=True)

    search_term = st.text_input("🔍 Search app name", placeholder="e.g. WhatsApp")
    disp_df = dff.copy()
    if search_term:
        disp_df = disp_df[disp_df["App"].str.contains(search_term, case=False, na=False)]

    cols_to_show = ["App", "Category", "Rating", "Reviews", "Installs",
                    "Type", "Price", "Content Rating", "Is_Free", "Success"]
    cols_to_show = [c for c in cols_to_show if c in disp_df.columns]

    st.dataframe(
        disp_df[cols_to_show].reset_index(drop=True),
        use_container_width=True,
        height=450,
    )

    st.caption(f"Showing {len(disp_df):,} rows · {len(cols_to_show)} columns")

    csv = disp_df[cols_to_show].to_csv(index=False).encode("utf-8")
    st.download_button(
        "⬇️ Download filtered CSV",
        data=csv,
        file_name="playstore_filtered.csv",
        mime="text/csv",
    )

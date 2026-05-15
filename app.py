"""
Tracksuit Survey Allocation Optimisation — Dashboard
"""

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import numpy as np
import pandas as pd

from main import (
    setup_environment, generate_respondents, target_setup,
    naive_allocation, greedy_approach, lp_optimal, simulate_allocation,
    NZ_GENDER, NZ_AGE, SAMPLE_SIZE, TARGET_QUALIFIED, TIME_BUDGET_S, CONFIDENCE,
)

# ── Palette ────────────────────────────────────────────────────────────────────
C_NAIVE  = "#9A8E82"
C_GREEDY = "#D4A574"
C_LP     = "#C05E3C"
C_GREEN  = "#4A7C59"
C_BLUE   = "#4A6FA5"
C_BG     = "#F5F0EA"

st.set_page_config(
    page_title="Tracksuit Survey Optimisation",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  /* ── Global text & backgrounds ── */
  [data-testid="stAppViewContainer"],
  [data-testid="stAppViewContainer"] > section { background: #F5F0EA !important; }
  [data-testid="stSidebar"] { background: #EDE7DD !important; }

  /* Force all Streamlit text to dark */
  body, p, span, div, li, label,
  [data-testid="stMarkdownContainer"],
  [data-testid="stSidebar"] *,
  .stTextInput label, .stSlider label,
  .stMultiSelect label { color: #2C2520 !important; }

  /* Headings */
  h1, h2, h3, h4, h5, h6 { color: #2C2520 !important; }

  /* Tabs */
  [data-baseweb="tab"] { color: #2C2520 !important; }
  [data-baseweb="tab"][aria-selected="true"] { color: #C05E3C !important; }

  /* Dataframe text */
  .stDataFrame td, .stDataFrame th { color: #2C2520 !important; }

  /* Metric cards */
  .metric-card {
    background: #FFFFFF; border: 1px solid #DDD5C8;
    border-radius: 14px; padding: 20px 24px; text-align: center;
  }
  .metric-label { font-size: 11px; color: #9A8E82 !important; letter-spacing: 2px;
                  text-transform: uppercase; margin-bottom: 6px; }
  .metric-value { font-size: 28px; font-weight: 700; color: #2C2520 !important; }
  .metric-delta { font-size: 13px; color: #4A7C59 !important; margin-top: 4px; }
</style>
""", unsafe_allow_html=True)


# ── Data loading (cached) ──────────────────────────────────────────────────────
@st.cache_data
def load_base():
    rng, df         = setup_environment()
    pool            = generate_respondents(SAMPLE_SIZE, rng)
    targets, lookup = target_setup(df)
    return rng, df, pool, targets, lookup


@st.cache_data
def run_naive(_lookup, probabilistic: bool = True, overage: float = 0.0):
    return naive_allocation(_lookup, probabilistic=probabilistic, overage=overage)

@st.cache_data
def run_greedy(_pool, _lookup, probabilistic: bool = True, overage: float = 0.0):
    return greedy_approach(_pool, _lookup, probabilistic=probabilistic, overage=overage)

@st.cache_data(show_spinner="Solving LP (CBC)… this takes ~10–20 s")
def run_lp(_lookup, probabilistic: bool = True, overage: float = 0.0):
    return lp_optimal(_lookup, probabilistic=probabilistic, overage=overage)

def run_sim(_bundles, _lookup, n_months):
    # No caching — fresh unseeded RNG every call so each run is unique
    return simulate_allocation(_bundles, _lookup, n_months=n_months,
                               rng=np.random.default_rng())


# ── Helpers ────────────────────────────────────────────────────────────────────
def bundle_size_breakdown(bundles):
    counts = {1: 0, 2: 0, 3: 0}
    resp   = {1: 0, 2: 0, 3: 0}
    for b in bundles:
        s = len(b["categories"])
        counts[s] = counts.get(s, 0) + 1
        resp[s]   = resp.get(s, 0) + b["n_respondents"]
    return counts, resp

def sim_per_category_means(sim, lookup):
    q   = sim["qualified"]
    ids = sim["cat_ids"]
    return pd.DataFrame({
        "category_id":    ids,
        "mean_qualified": q.mean(axis=0),
        "median_qualified": np.median(q, axis=0),
        "p5":  np.percentile(q, 5,  axis=0),
        "p95": np.percentile(q, 95, axis=0),
        "pct_met": (q >= TARGET_QUALIFIED).mean(axis=0) * 100,
    })


# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ Controls")
    st.divider()

    models_selected = st.multiselect(
        "Models to run",
        ["Naive", "Greedy", "LP-Optimal"],
        default=["Naive", "Greedy", "LP-Optimal"],
    )

    n_months = st.slider("Simulation months", 1, 24, 12, step=1)

    st.divider()
    probabilistic = st.toggle(
        f"Probabilistic guarantee ({CONFIDENCE:.0%} confidence)",
        value=True,
        help=(
            "ON — inflates respondent counts so each category meets the 200 "
            "qualified target with 95% probability (accounts for Bernoulli variance). "
            "OFF — uses the deterministic ceil(200 / p) baseline."
        ),
    )

    overage_pct = st.select_slider(
        "Safety buffer (overage)",
        options=[0, 5, 10, 15, 20],
        value=0,
        format_func=lambda x: f"+{x}%",
        help=(
            "Inflate the recruitment target by this percentage above 200. "
            "+10% sizes each category for 220 qualified instead of 200, "
            "giving an extra margin against stochastic shortfalls. "
            "Has no effect on simulation scoring — categories are still "
            "graded against the original 200 target."
        ),
    )
    overage = overage_pct / 100.0

    st.divider()
    run_btn = st.button("▶  Run / Refresh", width="stretch", type="primary")

    st.divider()
    st.markdown("**Constraints**")
    _eff = int(round(TARGET_QUALIFIED * (1 + overage)))
    _target_str = f"**{TARGET_QUALIFIED}** / category" if overage == 0 else f"**{_eff}** / category *(+{overage_pct}% buffer)*"
    st.markdown(f"- Recruitment target: {_target_str}")
    st.markdown(f"- Time budget: **{TIME_BUDGET_S:.0f} s**")
    st.markdown(f"- Max bundle size: **3**")
    st.divider()
    st.markdown("**Data**")
    st.markdown("77 categories · NZ population")


# ── Header ─────────────────────────────────────────────────────────────────────
st.markdown("## Survey Allocation Optimisation")
st.markdown(
    "Compare three allocation strategies across respondent cost, "
    "qualification yield, and constraint compliance."
)
st.divider()

# ── Load data ──────────────────────────────────────────────────────────────────
rng, df, pool, targets, lookup = load_base()

# ── Run models (only on button press or first load) ────────────────────────────
should_run = run_btn or "results" not in st.session_state

if should_run:
    results = {}

    if "Naive" in models_selected:
        with st.spinner("Running Naive..."):
            nb = run_naive(lookup, probabilistic, overage)
            results["Naive"] = {
                "bundles": nb,
                "sim":     run_sim(nb, lookup, n_months),
                "colour":  C_NAIVE,
            }

    if "Greedy" in models_selected:
        with st.spinner("Running Greedy..."):
            gb = run_greedy(pool, lookup, probabilistic, overage)
            results["Greedy"] = {
                "bundles": gb,
                "sim":     run_sim(gb, lookup, n_months),
                "colour":  C_GREEDY,
            }

    if "LP-Optimal" in models_selected:
        with st.spinner("Running LP-Optimal (CBC solver)..."):
            lb = run_lp(lookup, probabilistic, overage)
            results["LP-Optimal"] = {
                "bundles": lb,
                "sim":     run_sim(lb, lookup, n_months),
                "colour":  C_LP,
            }

    if results:
        st.session_state["results"] = results
    else:
        st.info("Select at least one model in the sidebar and click Run.")
        st.stop()

results = st.session_state.get("results", {})

if not results:
    st.info("Select at least one model in the sidebar and click ▶ Run / Refresh.")
    st.stop()


# ══════════════════════════════════════════════════════════════════════════════
# KPI Cards
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("### Key Metrics")
kpi_cols = st.columns(len(results) + 1)

naive_total = sum(b["n_respondents"] for b in naive_allocation(lookup, overage=overage)) if "Naive" not in results else results["Naive"]["sim"]["total_respondents"]

with kpi_cols[0]:
    st.markdown(f"""
    <div class="metric-card">
      <div class="metric-label">Categories</div>
      <div class="metric-value">77</div>
      <div class="metric-delta">Target: 200 qual / cat</div>
    </div>""", unsafe_allow_html=True)

for col, (model, data) in zip(kpi_cols[1:], results.items()):
    total = data["sim"]["total_respondents"]
    saving = f"−{100*(naive_total - total)/naive_total:.1f}% vs naive" if model != "Naive" else "baseline"
    pct_over = data["sim"]["pct_over"].mean()
    with col:
        st.markdown(f"""
        <div class="metric-card">
          <div class="metric-label">{model}</div>
          <div class="metric-value">{total:,}</div>
          <div class="metric-delta">{saving}</div>
        </div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# Tab Layout
# ══════════════════════════════════════════════════════════════════════════════
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📊 Respondent Cost",
    "📈 Yield Distribution",
    "⏱ Time Budget",
    "🗂 Bundle Breakdown",
    "🔬 Pool & Demographics",
    "🎯 Category Fill",
])


FONT = dict(family="Plus Jakarta Sans, sans-serif", color="#2C2520")
LAYOUT = dict(plot_bgcolor=C_BG, paper_bgcolor=C_BG, font=FONT)


# ── Tab 1: Respondent Cost Comparison ─────────────────────────────────────────
with tab1:
    st.markdown("#### Total Unique Respondents by Strategy")
    st.markdown(
        "Total respondents recruited across all bundles for one month. "
        "Lower is better — each reduction in respondents is a direct cost saving."
    )
    fig = go.Figure()
    for model, data in results.items():
        fig.add_trace(go.Bar(
            name=model,
            x=[model],
            y=[data["sim"]["total_respondents"]],
            marker_color=data["colour"],
            text=[f"{data['sim']['total_respondents']:,}"],
            textposition="outside",
            textfont=dict(color="#2C2520"),
        ))
    fig.update_layout(
        **LAYOUT, showlegend=False, height=380,
        xaxis_title="Allocation strategy",
        yaxis_title="Unique respondents recruited",
    )
    st.plotly_chart(fig, width="stretch")

    st.markdown("#### Respondents Saved vs Naive Baseline")
    st.markdown(
        "Respondents that each optimised strategy avoids recruiting per month relative to the naive baseline. "
        "Every saved respondent is a direct reduction in panel recruitment cost."
    )
    has_non_naive = any(m != "Naive" for m in results)
    if has_non_naive:
        fig_sav = go.Figure()
        for model, data in results.items():
            if model == "Naive":
                continue
            saved = naive_total - data["sim"]["total_respondents"]
            pct   = 100 * saved / naive_total if naive_total else 0
            fig_sav.add_trace(go.Bar(
                name=model,
                x=[model],
                y=[saved],
                marker_color=data["colour"],
                text=[f"{saved:,}<br>({pct:.1f}% reduction)"],
                textposition="outside",
                textfont=dict(color="#2C2520"),
            ))
        fig_sav.update_layout(
            **LAYOUT, showlegend=False, height=320,
            xaxis_title="Allocation strategy",
            yaxis_title="Respondents saved vs naive (per month)",
        )
        st.plotly_chart(fig_sav, width="stretch")
    else:
        st.info("Add Greedy or LP-Optimal to see savings vs naive.")


# ── Tab 2: Yield Distribution (mean ≈ median check) ───────────────────────────
with tab2:
    st.markdown("#### Qualified Respondents per Category — Distribution")
    st.markdown(
        "Each bar represents one category's average monthly qualified count across the simulation. "
        "When mean ≈ median, the allocation is balanced — no categories are chronically under- or over-served."
    )

    n_cols = len(results)
    fig = make_subplots(rows=1, cols=n_cols, subplot_titles=list(results.keys()), shared_yaxes=True)

    for col_i, (model, data) in enumerate(results.items(), 1):
        cat_stats = sim_per_category_means(data["sim"], lookup)
        means     = cat_stats["mean_qualified"].values
        mn, md    = means.mean(), np.median(means)

        fig.add_trace(go.Histogram(
            x=means, nbinsx=25,
            marker_color=data["colour"], opacity=0.8,
            name=model, showlegend=False,
        ), row=1, col=col_i)
        fig.add_vline(x=mn, line_color="#2C2520", line_dash="solid", line_width=2, row=1, col=col_i)
        fig.add_vline(x=md, line_color=C_GREEN,   line_dash="dash",  line_width=2, row=1, col=col_i)
        fig.add_vline(x=TARGET_QUALIFIED, line_color=C_BLUE, line_dash="dot", line_width=1.5, row=1, col=col_i)

        # Place μ/M label inside the plot at top-right using domain coordinates (0–1 range)
        xref = "x domain" if col_i == 1 else f"x{col_i} domain"
        yref = "y domain" if col_i == 1 else f"y{col_i} domain"
        fig.add_annotation(
            x=0.97, y=0.97, xref=xref, yref=yref,
            xanchor="right", yanchor="top",
            text=f"μ = {mn:.0f}<br>M = {md:.0f}",
            showarrow=False,
            font=dict(size=11, color="#2C2520"),
            bgcolor="rgba(255,255,255,0.8)",
            bordercolor="#DDD5C8",
            borderwidth=1,
        )

    fig.update_layout(
        **LAYOUT, height=420,
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color="#2C2520")),
    )
    fig.update_xaxes(title_text="Mean qualified respondents per month", color="#2C2520", tickfont=dict(color="#2C2520"))
    fig.update_yaxes(title_text="Number of categories", col=1, color="#2C2520", tickfont=dict(color="#2C2520"))
    st.plotly_chart(fig, width="stretch")

    st.markdown(
        "**─** Mean &nbsp;&nbsp; "
        "<span style='color:#4A7C59'>**- -**</span> Median &nbsp;&nbsp; "
        "<span style='color:#4A6FA5'>**···**</span> Target (200)",
        unsafe_allow_html=True,
    )

    st.markdown("#### Mean / Median Alignment")
    annual_target = TARGET_QUALIFIED * 12
    rows = []
    for model, data in results.items():
        means = sim_per_category_means(data["sim"], lookup)["mean_qualified"].values
        projected_annual = means.mean() * 12
        rows.append({
            "Model":                   model,
            "Mean / month":            f"{means.mean():.1f}",
            "Median / month":          f"{np.median(means):.1f}",
            "Std Dev":                 f"{means.std():.1f}",
            "% cats ≥200 / month":     f"{(means >= TARGET_QUALIFIED).mean()*100:.1f}%",
            f"Proj. annual (×12)":     f"{projected_annual:.0f}",
            f"≥ {annual_target}/yr":   "✓" if projected_annual >= annual_target else "✗",
        })
    st.dataframe(pd.DataFrame(rows).set_index("Model"), width="stretch")
    st.caption(f"Annual projection = mean qualified/month × 12. Contract requires ≥{annual_target:,} qualified per category per year.")



# ── Tab 3: Time Budget Compliance ─────────────────────────────────────────────
with tab3:
    st.markdown("#### Mean Respondent Time Distribution (across simulated months)")
    st.markdown(
        "Distribution of the average interview length per simulated month. "
        "All strategies should stay well left of the 480s contractual limit."
    )

    fig = go.Figure()
    for model, data in results.items():
        fig.add_trace(go.Histogram(
            x=data["sim"]["mean_times"],
            name=model,
            marker_color=data["colour"],
            opacity=0.7,
            nbinsx=40,
        ))
    fig.add_vline(x=TIME_BUDGET_S, line_color=C_LP, line_dash="dash",
                  line_width=2, annotation_text=f"{TIME_BUDGET_S:.0f}s budget",
                  annotation_font_color="#2C2520")
    fig.update_layout(
        **LAYOUT, barmode="overlay", height=380,
        xaxis_title="Mean respondent interview time (seconds)",
        yaxis_title="Number of simulated months",
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color="#2C2520")),
    )
    st.plotly_chart(fig, width="stretch")

    st.markdown("#### % Respondents Exceeding 480s Budget")
    rows = []
    for model, data in results.items():
        rows.append({
            "Model":              model,
            "Mean time (s)":      f"{data['sim']['mean_times'].mean():.1f}",
            "Median time (s)":    f"{data['sim']['median_times'].mean():.1f}",
            "99th pct (s)":       f"{np.percentile(data['sim']['mean_times'], 99):.1f}",
            "Mean % over budget": f"{data['sim']['pct_over'].mean():.2f}%",
            "Max % over budget":  f"{data['sim']['pct_over'].max():.2f}%",
        })
    st.dataframe(pd.DataFrame(rows).set_index("Model"), width="stretch")

    st.divider()
    st.markdown("#### Mean vs. Median Respondent Time per Month")
    st.markdown(
        "Distributions of the monthly mean (solid) and median (outlined) interview time per respondent. "
        "When every respondent spends a similar amount of time — i.e. bundles are sized consistently "
        "and incidence rates are close — the mean and median track together and the two distributions "
        "overlap. A large gap signals high within-month variance: some respondents qualifying for many "
        "surveys while others qualify for none."
    )

    n_cols   = len(results)
    fig_mm   = make_subplots(rows=1, cols=n_cols, subplot_titles=list(results.keys()), shared_yaxes=True)

    for col_i, (model, data) in enumerate(results.items(), 1):
        sim    = data["sim"]
        colour = data["colour"]
        mn_arr = sim["mean_times"]
        md_arr = sim["median_times"]
        avg_mn = mn_arr.mean()
        avg_md = md_arr.mean()

        # Solid bars for mean
        fig_mm.add_trace(go.Histogram(
            x=mn_arr, nbinsx=30,
            marker_color=colour, opacity=0.75,
            name="Mean" if col_i == 1 else None,
            showlegend=(col_i == 1),
            legendgroup="mean",
        ), row=1, col=col_i)

        # Outlined bars (no fill) for median — same colour so misalignment is visible
        fig_mm.add_trace(go.Histogram(
            x=md_arr, nbinsx=30,
            marker_color="rgba(0,0,0,0)",
            marker_line_color=colour,
            marker_line_width=2,
            name="Median" if col_i == 1 else None,
            showlegend=(col_i == 1),
            legendgroup="median",
        ), row=1, col=col_i)

        xref = "x domain" if col_i == 1 else f"x{col_i} domain"
        yref = "y domain" if col_i == 1 else f"y{col_i} domain"
        fig_mm.add_annotation(
            x=0.97, y=0.97, xref=xref, yref=yref,
            xanchor="right", yanchor="top",
            text=f"Avg mean: {avg_mn:.1f}s<br>Avg median: {avg_md:.1f}s",
            showarrow=False,
            font=dict(size=11, color="#2C2520"),
            bgcolor="rgba(255,255,255,0.85)",
            bordercolor="#DDD5C8",
            borderwidth=1,
        )

    fig_mm.update_layout(
        **LAYOUT, barmode="overlay", height=380,
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color="#2C2520")),
    )
    fig_mm.update_xaxes(title_text="Respondent interview time (seconds)", color="#2C2520", tickfont=dict(color="#2C2520"))
    fig_mm.update_yaxes(title_text="Simulated months", col=1, color="#2C2520", tickfont=dict(color="#2C2520"))
    st.plotly_chart(fig_mm, width="stretch")


# ── Tab 4: Bundle Breakdown ────────────────────────────────────────────────────
with tab4:
    st.markdown("#### Allocation by Bundle Size")
    st.markdown(
        "Respondents broken down by how many categories they were asked to screen for. "
        "Triples share one respondent across three categories, maximising cost efficiency."
    )
    fig = go.Figure()
    size_labels  = {1: "Singles", 2: "Pairs", 3: "Triples"}
    size_colours = {1: C_BLUE, 2: C_GREEN, 3: C_LP}

    for size in [1, 2, 3]:
        y_vals = []
        for model, data in results.items():
            _, resp = bundle_size_breakdown(data["bundles"])
            y_vals.append(resp.get(size, 0))
        fig.add_trace(go.Bar(
            name=size_labels[size],
            x=list(results.keys()),
            y=y_vals,
            marker_color=size_colours[size],
            opacity=0.85,
        ))
    fig.update_layout(
        **LAYOUT, barmode="stack", height=380,
        xaxis_title="Allocation strategy",
        yaxis_title="Respondents recruited",
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color="#2C2520")),
    )
    st.plotly_chart(fig, width="stretch")

    st.markdown("#### Bundle Count by Size")
    rows = []
    for model, data in results.items():
        counts, resp = bundle_size_breakdown(data["bundles"])
        rows.append({
            "Model":             model,
            "Singles":           counts.get(1, 0),
            "Pairs":             counts.get(2, 0),
            "Triples":           counts.get(3, 0),
            "Total bundles":     sum(counts.values()),
            "Total respondents": f"{sum(resp.values()):,}",
        })
    st.dataframe(pd.DataFrame(rows).set_index("Model"), width="stretch")

    st.divider()

    # ── Bundle strategy deep-dive ──────────────────────────────────────────────
    st.markdown("#### Bundle Strategy — Respondents per Bundle")
    st.markdown(
        "Each bar is one bundle, sized by respondents allocated to it. "
        "Hover to see which categories are grouped together, their incidence rates, "
        "and the worst-case survey time if a respondent qualifies for all surveys."
    )

    bundle_model = st.selectbox("Model", list(results.keys()), key="bundle_model_select")
    bundles      = results[bundle_model]["bundles"]
    cat_names    = {int(row["category_id"]): row["category_name"] for _, row in targets.iterrows()}
    cat_inc      = {int(row["category_id"]): row["incidence_rate"] for _, row in targets.iterrows()}

    # Build one row per bundle, sorted by n_respondents descending
    bundle_rows = []
    for i, b in enumerate(sorted(bundles, key=lambda x: x["n_respondents"], reverse=True)):
        size    = len(b["categories"])
        cats    = b["categories"]
        names   = [cat_names.get(c, str(c)) for c in cats]
        incs    = [cat_inc.get(c, 0) for c in cats]
        label   = " + ".join(n[:28] + "…" if len(n) > 28 else n for n in names)
        tooltip = "<br>".join(
            f"{cat_names.get(c, c)} (p={cat_inc.get(c,0):.3f})" for c in cats
        )
        bundle_rows.append({
            "label":          label,
            "n_respondents":  b["n_respondents"],
            "size":           size,
            "worst_case_s":   b.get("worst_case_time", 0),
            "expected_s":     b.get("expected_time", 0),
            "tooltip":        tooltip,
            "colour":         size_colours.get(size, C_NAIVE),
        })

    bdf = pd.DataFrame(bundle_rows)

    fig_b = go.Figure()
    for size, label, colour in [(1, "Single", C_BLUE), (2, "Pair", C_GREEN), (3, "Triple", C_LP)]:
        sub = bdf[bdf["size"] == size]
        if sub.empty:
            continue
        fig_b.add_trace(go.Bar(
            x=sub["n_respondents"],
            y=sub["label"],
            orientation="h",
            name=label,
            marker_color=colour,
            opacity=0.85,
            customdata=np.stack([
                sub["worst_case_s"].round(1),
                sub["expected_s"].round(1),
                sub["tooltip"],
            ], axis=-1),
            hovertemplate=(
                "<b>%{y}</b><br>"
                "Respondents: %{x:,}<br>"
                "Worst-case time: %{customdata[0]}s<br>"
                "Expected time: %{customdata[1]}s<br>"
                "<br>Categories:<br>%{customdata[2]}<extra></extra>"
            ),
        ))

    fig_b.update_layout(
        **LAYOUT,
        barmode="stack",
        height=max(500, len(bdf) * 22),
        xaxis_title="Respondents allocated to bundle",
        yaxis_title="",
        yaxis=dict(tickfont=dict(size=10, color="#2C2520"), autorange="reversed"),
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color="#2C2520")),
        margin=dict(l=260, r=40, t=20, b=40),
    )
    st.plotly_chart(fig_b, width="stretch")


# ── Tab 5: Pool & Demographics ─────────────────────────────────────────────────
with tab5:
    st.markdown("#### Respondent Pool Demographics")
    col_a, col_b = st.columns(2)

    st.markdown(
        "Census-derived quotas used to stratify the synthetic respondent pool. "
        "Every bundle's exposure pool is drawn from this same distribution, ensuring demographic representativeness."
    )
    with col_a:
        st.markdown("**Gender split** (Stats NZ, Dec 2025)")
        gen_df = pd.DataFrame({
            "Gender":   list(NZ_GENDER.keys()),
            "Target %": [v * 100 for v in NZ_GENDER.values()],
        })
        fig = px.bar(gen_df, x="Gender", y="Target %",
                     color="Gender", color_discrete_sequence=[C_LP, C_BLUE],
                     text_auto=".2f")
        fig.update_layout(
            **LAYOUT, showlegend=False, height=280,
            xaxis_title="Gender",
            yaxis_title="Population share (%)",
        )
        st.plotly_chart(fig, width="stretch")

    with col_b:
        st.markdown("**Age bracket distribution** (Infometrics NZ, Jun 2025)")
        age_df = pd.DataFrame({
            "Bracket":      list(NZ_AGE.keys()),
            "Population %": [v * 100 for v in NZ_AGE.values()],
        })
        fig = px.bar(age_df, x="Bracket", y="Population %", text_auto=".1f")
        fig.update_traces(marker_color=C_GREEN)
        fig.update_layout(
            **LAYOUT, height=280,
            xaxis_title="Age bracket",
            yaxis_title="Population share (%)",
        )
        st.plotly_chart(fig, width="stretch")

    st.markdown("#### Per-Category Incidence Rate Distribution")
    st.markdown(
        "How common each category is in the NZ population. "
        "Low-incidence categories are the costliest to recruit for — they drive most of the optimisation value."
    )
    fig = px.histogram(
        targets, x="incidence_rate", nbins=20,
        color_discrete_sequence=[C_LP],
        labels={"incidence_rate": "Incidence rate", "count": "# categories"},
    )
    fig.update_layout(
        **LAYOUT, height=300,
        xaxis_title="Incidence rate (proportion of population that qualifies)",
        yaxis_title="Number of categories",
    )
    st.plotly_chart(fig, width="stretch")

    st.markdown("#### Expected Survey Time per Category (incidence × survey length)")
    st.markdown(
        "Expected seconds one respondent spends on a category (p × l). "
        "Categories further right and higher up consume more of the 480s time budget when bundled."
    )
    fig = px.scatter(
        targets, x="incidence_rate", y="expected_time",
        hover_name="category_name",
        color="expected_time",
        color_continuous_scale=["#4A7C59", "#C05E3C"],
        labels={"incidence_rate": "Incidence rate", "expected_time": "Expected time (s)"},
    )
    fig.add_hline(y=TIME_BUDGET_S, line_dash="dash", line_color=C_NAIVE,
                  annotation_text="480s budget", annotation_font_color="#2C2520")
    fig.update_layout(
        **LAYOUT, height=350, coloraxis_showscale=False,
        xaxis_title="Incidence rate",
        yaxis_title="Expected time per respondent (seconds)",
    )
    st.plotly_chart(fig, width="stretch")


# ── Tab 6: Category Fill ───────────────────────────────────────────────────────
with tab6:
    # Toggle between two heatmap views
    heat_metric = st.radio(
        "Colour metric",
        ["% of months meeting target (≥200)", "Mean qualified as % of target"],
        horizontal=True,
        help=(
            "'% of months' shows how reliably each category hits 200 across the "
            "simulated months — the probabilistic toggle has a strong visible effect here. "
            "'Mean qualified' shows the average fill level."
        ),
    )

    # Build matrix: rows = categories, cols = models
    cat_order = (
        targets[["category_id", "category_name", "incidence_rate"]]
        .sort_values("incidence_rate")
        .reset_index(drop=True)
    )

    model_cols = list(results.keys())
    cat_labels = cat_order["category_name"].tolist()
    z_matrix   = []

    for model in model_cols:
        sim_q  = results[model]["sim"]["qualified"]           # (n_months, n_cats)
        cat_ids = results[model]["sim"]["cat_ids"]
        # Align to cat_order
        id_to_col = {cid: j for j, cid in enumerate(cat_ids)}
        col_idx   = [id_to_col[int(cid)] for cid in cat_order["category_id"]]
        aligned   = sim_q[:, col_idx]                         # (n_months, 77)

        if heat_metric.startswith("% of months"):
            vals = (aligned >= TARGET_QUALIFIED).mean(axis=0) * 100  # % months hitting target
        else:
            vals = aligned.mean(axis=0) / TARGET_QUALIFIED * 100     # mean as % of target

        z_matrix.append(vals.round(1))

    # Cell text: for mean-% view show raw qualified count, for reliability show %
    if heat_metric.startswith("% of months"):
        # z_vals are already 0–100 (% of months)
        z_vals    = np.column_stack(z_matrix)
        cell_text = [[f"{v:.0f}%" for v in row] for row in z_vals]
        zmid, zmin, zmax = 80, 0, 100
        cbar_ticks  = [0, 50, 80, 100]
        cbar_labels = ["0%", "50%", "80%", "100%"]
        cbar_title  = "% months ≥200"
        desc = (
            f"Colour = % of the {n_months} simulated month(s) where a category received ≥200 qualified respondents. "
            "Red = frequently missing. Green = consistently hitting. "
            "Toggle the probabilistic guarantee in the sidebar to see the reliability shift."
        )
    else:
        # z_matrix holds mean_qualified/200*100 — convert back to raw counts for display
        pct_matrix = np.column_stack(z_matrix)          # (77, n_models), % of target
        raw_matrix = pct_matrix / 100 * TARGET_QUALIFIED # mean qualified respondents

        # Colour on a tight ±15% window around 100 % — captures all meaningful variation
        # Values above 115 % are all-green; below 85 % are all-red.
        z_vals    = np.clip(pct_matrix, 85, 115)
        cell_text = [[f"{raw_matrix[r, c]:.0f}/{TARGET_QUALIFIED}" for c in range(raw_matrix.shape[1])]
                     for r in range(raw_matrix.shape[0])]
        zmid, zmin, zmax = 100, 85, 115
        cbar_ticks  = [85, 95, 100, 105, 115]
        cbar_labels = ["85%", "95%", "100%", "105%", "≥115%"]
        cbar_title  = "Mean qualified / target"
        desc = (
            f"Each cell shows the average qualified respondents per month across {n_months} simulated month(s) "
            f"(e.g. 197/{TARGET_QUALIFIED} means a category averaged 197 qualified respondents per month). "
            "Colour centres on 100% of target — red = under-served, green = over-served. "
            "Scale clipped at ±15% so near-target variation is visible; "
            "high-incidence categories bundled with rare ones naturally over-deliver."
        )

    st.markdown(f"#### Category Fill — {heat_metric}")
    st.markdown(desc + " Categories sorted by incidence rate (rarest at top).")

    fig_heat = go.Figure(go.Heatmap(
        z=z_vals,
        x=model_cols,
        y=cat_labels,
        text=cell_text,
        texttemplate="%{text}",
        textfont=dict(size=9, color="#2C2520"),
        colorscale=[
            [0.0,  "#C05E3C"],
            [0.5,  "#F5E6DC"],
            [0.75, "#FFFFFF"],
            [1.0,  "#4A7C59"],
        ],
        zmid=zmid,
        zmin=zmin,
        zmax=zmax,
        colorbar=dict(
            title=dict(text=cbar_title, font=dict(color="#2C2520")),
            tickvals=cbar_ticks,
            ticktext=cbar_labels,
            tickfont=dict(color="#2C2520"),
        ),
        hovertemplate="<b>%{y}</b><br>%{x}: %{z:.1f}%<extra></extra>",
    ))
    fig_heat.update_layout(
        **LAYOUT,
        height=max(500, len(cat_labels) * 16),
        xaxis=dict(side="top", title="", tickfont=dict(color="#2C2520")),
        yaxis=dict(
            title="Category (sorted by incidence rate, rarest at top)",
            autorange="reversed",
            tickfont=dict(size=10, color="#2C2520"),
        ),
        margin=dict(l=220, r=60, t=60, b=20),
    )
    st.plotly_chart(fig_heat, width="stretch")

    # ── Companion horizontal bar chart for a single model ─────────────────────
    st.divider()
    st.markdown("#### Per-Category Mean Qualified — Single Model")
    st.markdown(
        "Select one model to see exact mean qualified counts per category, "
        "sorted from most under-served to most over-served."
    )

    selected_model = st.selectbox("Model", list(results.keys()), key="fill_model_select")

    cat_stats  = sim_per_category_means(results[selected_model]["sim"], lookup)
    bar_df     = cat_order.merge(cat_stats, on="category_id").sort_values("mean_qualified")
    bar_colour = [C_LP if v >= TARGET_QUALIFIED else C_NAIVE for v in bar_df["mean_qualified"]]

    fig_bar = go.Figure(go.Bar(
        x=bar_df["mean_qualified"],
        y=bar_df["category_name"],
        orientation="h",
        marker_color=bar_colour,
        text=bar_df["mean_qualified"].round(1),
        textposition="outside",
        textfont=dict(size=9, color="#2C2520"),
        hovertemplate="<b>%{y}</b><br>Mean qualified: %{x:.1f}<extra></extra>",
    ))
    fig_bar.add_vline(
        x=TARGET_QUALIFIED, line_dash="dot", line_color=C_BLUE, line_width=2,
        annotation_text="Target 200", annotation_font_color="#2C2520",
        annotation_position="top",
    )
    fig_bar.update_layout(
        **LAYOUT,
        height=max(600, len(bar_df) * 18),
        xaxis_title="Mean qualified respondents per month",
        yaxis_title="",
        yaxis=dict(tickfont=dict(size=10, color="#2C2520")),
        margin=dict(l=220, r=80, t=40, b=40),
        showlegend=False,
    )
    st.plotly_chart(fig_bar, width="stretch")

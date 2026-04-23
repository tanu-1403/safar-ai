"""
app.py — Safar AI Dashboard (Streamlit)
=================================================
Interactive command center for highway reflectivity intelligence.

Tabs:
  1. 🗺️  Highway Map         — Geographic heatmap of segments
  2. 📊  Reflectivity Monitor — Per-segment current status + trends
  3. 🔭  Spectral Analysis    — Hyperspectral curves per segment
  4. 📈  Degradation Forecast — ARIMA/LSTM predictions + CI bands
  5. 🔧  Maintenance Planner  — Priority alerts + budget analysis
  6. 🧠  AI Explainability    — Feature importance + SHAP overview
  7. ⚙️  System Config        — Run simulation, retrain model

Run:
    cd safar_ai && streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import sys, os, time, warnings
from datetime import datetime, timedelta

warnings.filterwarnings("ignore")

# ── Path setup ────────────────────────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(__file__))

# ── Module imports ────────────────────────────────────────────────────────────
from utils import (
    reflectivity_to_status, ALERT_CRITICAL, ALERT_WARNING, ALERT_GOOD,
    SPECTRAL_BANDS, NH_CORRIDORS, FORECAST_DAYS, HISTORY_DAYS, timestamp_range
)
from modules.spectral   import SpectralReflectivityEngine
from modules.model      import ReflectivityPredictor, build_training_dataset
from modules.digital_twin import DigitalTwinRegistry, build_digital_twin_from_dataset
from modules.prediction import PredictionPipeline
from modules.ingestion  import RoadImageIngestor


# ─────────────────────────────────────────────────────────────────────────────
# Page Config
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Safar AI — Har Safar, Surakshit Safar",
    page_icon="🛣️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# Custom CSS — Dark industrial theme befitting NHAI infrastructure
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  /* Font */
  @import url('https://fonts.googleapis.com/css2?family=Rajdhani:wght@400;600;700&family=Inter:wght@300;400;500&display=swap');

  html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
    background-color: #0a0e1a;
    color: #e2e8f0;
  }

  .main { background-color: #0a0e1a; }

  /* Header */
  .nhai-header {
    background: linear-gradient(135deg, #0f172a 0%, #1e3a5f 50%, #0f172a 100%);
    border: 1px solid #1e40af;
    border-radius: 12px;
    padding: 20px 30px;
    margin-bottom: 20px;
    display: flex;
    align-items: center;
    gap: 16px;
  }
  .nhai-title {
    font-family: 'Rajdhani', sans-serif;
    font-size: 2rem;
    font-weight: 700;
    color: #f59e0b;
    letter-spacing: 1px;
    margin: 0;
  }
  .nhai-subtitle { color: #94a3b8; font-size: 0.85rem; margin: 0; }

  /* KPI Cards */
  .kpi-card {
    background: linear-gradient(145deg, #0f172a, #1e293b);
    border: 1px solid #1e40af33;
    border-radius: 10px;
    padding: 16px 20px;
    text-align: center;
    transition: border-color 0.3s;
  }
  .kpi-card:hover { border-color: #3b82f6; }
  .kpi-value {
    font-family: 'Rajdhani', sans-serif;
    font-size: 2.4rem;
    font-weight: 700;
    color: #f59e0b;
    line-height: 1;
  }
  .kpi-label { color: #64748b; font-size: 0.75rem; text-transform: uppercase; letter-spacing: 1px; margin-top: 4px; }

  /* Alert badges */
  .badge-critical { background:#ef44441a; color:#ef4444; border:1px solid #ef444433; padding:2px 10px; border-radius:20px; font-size:0.75rem; font-weight:600; }
  .badge-warning  { background:#f973161a; color:#f97316; border:1px solid #f9731633; padding:2px 10px; border-radius:20px; font-size:0.75rem; font-weight:600; }
  .badge-fair     { background:#f59e0b1a; color:#f59e0b; border:1px solid #f59e0b33; padding:2px 10px; border-radius:20px; font-size:0.75rem; font-weight:600; }
  .badge-good     { background:#22c55e1a; color:#22c55e; border:1px solid #22c55e33; padding:2px 10px; border-radius:20px; font-size:0.75rem; font-weight:600; }

  /* Tabs */
  .stTabs [data-baseweb="tab-list"] { background: #0f172a; border-radius: 8px; gap: 4px; }
  .stTabs [data-baseweb="tab"] { background: transparent; color: #64748b; border-radius: 6px; font-weight: 500; }
  .stTabs [aria-selected="true"] { background: #1e40af !important; color: white !important; }

  /* Metric delta */
  [data-testid="stMetricDelta"] { font-size: 0.75rem; }

  /* Progress bars */
  .stProgress > div > div > div { background: linear-gradient(90deg, #1d4ed8, #f59e0b); }

  /* Sidebar */
  [data-testid="stSidebar"] { background: #0a0e1a; border-right: 1px solid #1e293b; }

  /* DataFrame */
  .dataframe { background: #0f172a !important; }

  /* Section headers */
  .section-header {
    font-family: 'Rajdhani', sans-serif;
    font-size: 1.2rem;
    font-weight: 600;
    color: #f59e0b;
    border-left: 3px solid #f59e0b;
    padding-left: 10px;
    margin: 16px 0 12px;
  }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# Session State & Cache
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_resource(show_spinner="Initializing Digital Twin…")
def initialize_system(n_segments: int = 20, n_days: int = 180, seed: int = 42):
    """Build the full pipeline once and cache it."""
    engine  = SpectralReflectivityEngine()
    df      = engine.generate_synthetic_dataset(n_segments, n_days, seed)

    # Save dataset
    os.makedirs("data", exist_ok=True)
    df.to_csv("data/synthetic_spectral_data.csv", index=False)

    # Build training features
    train_df = build_training_dataset(df)

    # Train reflectivity predictor
    predictor = ReflectivityPredictor()
    metrics   = predictor.train(train_df)
    predictor.save()

    # Build digital twin
    registry  = build_digital_twin_from_dataset(df, n_segments=n_segments)
    registry.save()

    # History DataFrame
    hist_df   = registry.history_dataframe()

    # Fit prediction pipeline
    pipeline  = PredictionPipeline(use_lstm=False)
    pipeline.fit_all(hist_df)
    forecasts = pipeline.forecast_all(hist_df, steps=FORECAST_DAYS)

    return {
        "engine":       engine,
        "df":           df,
        "train_df":     train_df,
        "predictor":    predictor,
        "predictor_metrics": metrics,
        "registry":     registry,
        "hist_df":      hist_df,
        "pipeline":     pipeline,
        "forecasts":    forecasts,
    }


def get_system():
    return initialize_system()


# ─────────────────────────────────────────────────────────────────────────────
# Helper: Plotly Theme
# ─────────────────────────────────────────────────────────────────────────────
PLOTLY_THEME = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="#0d1117",
    font=dict(color="#94a3b8", family="Inter"),
    xaxis=dict(gridcolor="#1e293b", zerolinecolor="#1e293b"),
    yaxis=dict(gridcolor="#1e293b", zerolinecolor="#1e293b"),
)

STATUS_COLORS = {
    "CRITICAL": "#ef4444",
    "WARNING":  "#f97316",
    "FAIR":     "#f59e0b",
    "GOOD":     "#22c55e",
    "NO DATA":  "#64748b",
}


# ─────────────────────────────────────────────────────────────────────────────
# Header
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="nhai-header">
  <div style="font-size:3rem;">🛣️</div>
  <div>
    <p class="nhai-title" style="font-size:1.8rem;">SAFAR AI — HAR SAFAR, SURAKSHIT SAFAR</p>
    <p class="nhai-subtitle">Har Safar, Surakshit Safar — AI-Powered Highway Safety Intelligence</p>
  </div>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ System Controls")

    corridor  = st.selectbox("Highway Corridor", list(NH_CORRIDORS.keys()), index=0)
    n_segs    = st.slider("Segments to Simulate", 10, 30, 20)
    n_days    = st.slider("History Days", 60, 365, 180)

    st.markdown("---")
    st.markdown("### 🌦️ Live Simulation")
    sim_weather = st.selectbox("Weather Condition", ["clear","haze","rain","heavy_rain","fog"])
    sim_segment = st.selectbox("Test Segment", [f"SEG_{i+1:03d}" for i in range(n_segs)])

    if st.button("🔄 Re-run Simulation", use_container_width=True):
        st.cache_resource.clear()
        st.rerun()

    st.markdown("---")
    st.markdown("### 📊 Model Performance")

    sys_data = get_system()
    m = sys_data["predictor_metrics"]
    st.metric("Test MAE", f"{m.get('test_mae', 0):.4f}")
    st.metric("Test R²",  f"{m.get('test_r2',  0):.4f}")

    st.markdown("---")
    st.caption(f"🕐 Last updated: {datetime.now().strftime('%H:%M:%S')}")
    st.caption("© 2024 Safar AI · NHAI Hackathon · Har Safar, Surakshit Safar")


# ─────────────────────────────────────────────────────────────────────────────
# Load System
# ─────────────────────────────────────────────────────────────────────────────
with st.spinner("Loading Digital Twin…"):
    sys_data  = get_system()

registry    = sys_data["registry"]
hist_df     = sys_data["hist_df"]
forecasts   = sys_data["forecasts"]
pipeline    = sys_data["pipeline"]
predictor   = sys_data["predictor"]
engine      = sys_data["engine"]
df          = sys_data["df"]

summary_df  = registry.summary_dataframe()
alerts_df   = registry.get_alerts()
seg_ids     = summary_df["segment_id"].tolist()


# ─────────────────────────────────────────────────────────────────────────────
# KPI Strip
# ─────────────────────────────────────────────────────────────────────────────
n_critical = int((summary_df["status_label"] == "CRITICAL").sum())
n_warning  = int((summary_df["status_label"] == "WARNING").sum())
n_fair     = int((summary_df["status_label"] == "FAIR").sum())
n_good     = int((summary_df["status_label"] == "GOOD").sum())
avg_r      = summary_df["reflectivity_score"].mean()

k1, k2, k3, k4, k5, k6 = st.columns(6)
with k1:
    st.markdown(f'<div class="kpi-card"><div class="kpi-value">{len(summary_df)}</div><div class="kpi-label">Total Segments</div></div>', unsafe_allow_html=True)
with k2:
    st.markdown(f'<div class="kpi-card"><div class="kpi-value" style="color:#22c55e">{n_good}</div><div class="kpi-label">Good ✅</div></div>', unsafe_allow_html=True)
with k3:
    st.markdown(f'<div class="kpi-card"><div class="kpi-value" style="color:#f59e0b">{n_fair}</div><div class="kpi-label">Fair ⚠️</div></div>', unsafe_allow_html=True)
with k4:
    st.markdown(f'<div class="kpi-card"><div class="kpi-value" style="color:#f97316">{n_warning}</div><div class="kpi-label">Warning 🔶</div></div>', unsafe_allow_html=True)
with k5:
    st.markdown(f'<div class="kpi-card"><div class="kpi-value" style="color:#ef4444">{n_critical}</div><div class="kpi-label">Critical 🚨</div></div>', unsafe_allow_html=True)
with k6:
    color = "#22c55e" if avg_r >= ALERT_GOOD else ("#f59e0b" if avg_r >= ALERT_WARNING else "#ef4444")
    st.markdown(f'<div class="kpi-card"><div class="kpi-value" style="color:{color}">{avg_r:.2f}</div><div class="kpi-label">Avg Reflectivity</div></div>', unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# Main Tabs
# ─────────────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "🗺️ Highway Map",
    "📊 Reflectivity Monitor",
    "🔭 Spectral Analysis",
    "📈 Degradation Forecast",
    "🔧 Maintenance Planner",
    "🧠 AI Explainability",
])


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — Highway Map
# ══════════════════════════════════════════════════════════════════════════════
with tab1:
    st.markdown('<div class="section-header">Geographic Reflectivity Heatmap — NH Corridor</div>', unsafe_allow_html=True)

    col_map, col_info = st.columns([3, 1])

    with col_map:
        # Scatter Mapbox
        map_df = summary_df.copy()
        map_df["size"]   = 14
        map_df["text"]   = (
            map_df["segment_id"] + "<br>" +
            "Score: " + map_df["reflectivity_score"].round(3).astype(str) + "<br>" +
            "Status: " + map_df["status_label"] + "<br>" +
            "Action: " + map_df["action"]
        )

        fig_map = go.Figure()

        for status, color in STATUS_COLORS.items():
            mask = map_df["status_label"] == status
            if not mask.any():
                continue
            sub = map_df[mask]
            fig_map.add_trace(go.Scattermapbox(
                lat=sub["lat"], lon=sub["lon"],
                mode="markers",
                marker=dict(
                    size=sub["reflectivity_score"].clip(0.3, 1.0) * 22 + 8,
                    color=color,
                    opacity=0.85,
                    sizemode="diameter",
                ),
                text=sub["text"],
                hovertemplate="%{text}<extra></extra>",
                name=status,
            ))

        # Route line (NH-48 simplified path)
        fig_map.add_trace(go.Scattermapbox(
            lat=map_df.sort_values("start_km")["lat"].tolist(),
            lon=map_df.sort_values("start_km")["lon"].tolist(),
            mode="lines",
            line=dict(width=2, color="rgba(59,130,246,0.5)"),
            hoverinfo="skip",
            name="NH Route",
        ))

        fig_map.update_layout(
            mapbox=dict(
                style="carto-darkmatter",
                center=dict(lat=23.5, lon=75.5),
                zoom=4.5,
            ),
            margin=dict(l=0, r=0, t=0, b=0),
            height=480,
            paper_bgcolor="rgba(0,0,0,0)",
            legend=dict(
                bgcolor="#0f172a", bordercolor="#1e293b",
                font=dict(color="#94a3b8")
            ),
        )
        st.plotly_chart(fig_map, use_container_width=True)

    with col_info:
        st.markdown('<div class="section-header">Legend</div>', unsafe_allow_html=True)
        for status, color in STATUS_COLORS.items():
            cnt = int((map_df["status_label"] == status).sum())
            if cnt:
                st.markdown(
                    f'<div style="display:flex;align-items:center;gap:8px;margin-bottom:8px;">'
                    f'<div style="width:14px;height:14px;border-radius:50%;background:{color};flex-shrink:0;"></div>'
                    f'<span style="color:#e2e8f0;font-size:0.9rem;">{status} ({cnt})</span>'
                    f'</div>',
                    unsafe_allow_html=True
                )

        st.markdown("---")
        st.markdown('<div class="section-header">Active Alerts</div>', unsafe_allow_html=True)
        for _, row in alerts_df.head(8).iterrows():
            score = row["score"]
            badge_cls = "badge-critical" if score < ALERT_CRITICAL else "badge-warning"
            st.markdown(
                f'<div style="background:#0f172a;border:1px solid #1e293b;border-radius:8px;padding:8px 10px;margin-bottom:6px;">'
                f'<div style="font-weight:600;color:#e2e8f0;font-size:0.85rem;">{row["segment_id"]}</div>'
                f'<div style="color:#64748b;font-size:0.75rem;">km {row["last_km"]}</div>'
                f'<div style="margin-top:4px;"><span class="{badge_cls}">{row["status"]}: {score:.3f}</span></div>'
                f'</div>',
                unsafe_allow_html=True
            )


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — Reflectivity Monitor
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown('<div class="section-header">Per-Segment Reflectivity Dashboard</div>', unsafe_allow_html=True)

    col_sel, col_metric = st.columns([1, 3])
    with col_sel:
        sel_seg = st.selectbox("Select Segment", seg_ids, key="monitor_seg")

    seg_obj    = registry.get_segment(sel_seg)
    seg_hist   = hist_df[hist_df["segment_id"] == sel_seg].sort_values("timestamp")
    curr_score = seg_obj.latest_reflectivity() if seg_obj else 0.5
    curr_status = reflectivity_to_status(curr_score)

    with col_metric:
        m1, m2, m3, m4 = st.columns(4)
        with m1:
            st.metric("Current Reflectivity", f"{curr_score:.3f}",
                      delta=f"{curr_status['emoji']} {curr_status['label']}")
        with m2:
            trend = seg_obj.reflectivity_trend() if seg_obj else "N/A"
            trend_icon = "📉" if trend == "degrading" else ("📈" if trend == "improving" else "➡️")
            st.metric("Trend (7-day)", trend.title(), delta=trend_icon)
        with m3:
            n_readings = len(seg_hist)
            st.metric("Total Readings", n_readings)
        with m4:
            days_maint = seg_obj.days_since_maintenance() if seg_obj else None
            st.metric("Days Since Maintenance", days_maint if days_maint else "No Record")

    # Reflectivity time series
    if len(seg_hist):
        fig_ts = go.Figure()

        # Threshold bands
        fig_ts.add_hrect(y0=0,             y1=ALERT_CRITICAL, fillcolor="rgba(239,68,68,0.09)", line_width=0)
        fig_ts.add_hrect(y0=ALERT_CRITICAL,y1=ALERT_WARNING,  fillcolor="rgba(249,115,22,0.09)", line_width=0)
        fig_ts.add_hrect(y0=ALERT_WARNING, y1=ALERT_GOOD,     fillcolor="rgba(245,158,11,0.09)", line_width=0)
        fig_ts.add_hrect(y0=ALERT_GOOD,    y1=1.0,            fillcolor="rgba(34,197,94,0.09)", line_width=0)

        # Threshold lines
        for thresh, color, label in [
            (ALERT_CRITICAL, "#ef4444", "Critical"),
            (ALERT_WARNING,  "#f97316", "Warning"),
            (ALERT_GOOD,     "#22c55e", "Good"),
        ]:
            fig_ts.add_hline(y=thresh, line_dash="dot", line_color=color,
                             annotation_text=label, annotation_position="right")

        # Historical line
        fig_ts.add_trace(go.Scatter(
            x=seg_hist["timestamp"],
            y=seg_hist["reflectivity_score"],
            mode="lines",
            line=dict(color="#3b82f6", width=2),
            name="Measured",
            fill="tozeroy",
            fillcolor="rgba(59,130,246,0.08)",
        ))

        # Scatter colored by status
        colors_per_pt = [
            STATUS_COLORS.get(reflectivity_to_status(v)["label"], "#3b82f6")
            for v in seg_hist["reflectivity_score"]
        ]
        fig_ts.add_trace(go.Scatter(
            x=seg_hist["timestamp"], y=seg_hist["reflectivity_score"],
            mode="markers",
            marker=dict(color=colors_per_pt, size=4),
            hovertemplate="Date: %{x}<br>Score: %{y:.4f}<extra></extra>",
            name="Data Points",
            showlegend=False,
        ))

        # Forecast (first 30 days)
        fc_data = forecasts.get(sel_seg)
        if fc_data is not None:
            last_date   = seg_hist["timestamp"].max()
            fc_dates    = pd.date_range(start=last_date + timedelta(days=1), periods=30, freq="D")
            fc_mean     = fc_data["mean"][:30]
            fc_lower    = fc_data["lower"][:30]
            fc_upper    = fc_data["upper"][:30]

            fig_ts.add_trace(go.Scatter(
                x=list(fc_dates) + list(fc_dates[::-1]),
                y=list(fc_upper) + list(fc_lower[::-1]),
                fill="toself", fillcolor="rgba(245,158,11,0.12)",
                line=dict(color="rgba(0,0,0,0)"),
                name="Forecast CI", hoverinfo="skip",
            ))
            fig_ts.add_trace(go.Scatter(
                x=fc_dates, y=fc_mean,
                mode="lines",
                line=dict(color="#f59e0b", width=2, dash="dot"),
                name="Forecast (30d)",
            ))

        fig_ts.update_layout(
            title=f"Reflectivity History — {sel_seg}",
            height=380,
            **PLOTLY_THEME,
            legend=dict(
                bgcolor="#0f172a",
                bordercolor="#1e293b",
                font=dict(color="#94a3b8")
            ),
        )

        fig_ts.update_xaxes(title="Date")

        fig_ts.update_yaxes(
            title="Reflectivity Score",
            range=[0, 1]
        )

        st.plotly_chart(fig_ts, use_container_width=True)

    # All segments comparison bar chart
    st.markdown('<div class="section-header">All Segments — Current Status</div>', unsafe_allow_html=True)

    bar_df = summary_df.sort_values("reflectivity_score")
    fig_bar = go.Figure(go.Bar(
        x=bar_df["segment_id"],
        y=bar_df["reflectivity_score"],
        marker_color=[STATUS_COLORS.get(s, "#3b82f6") for s in bar_df["status_label"]],
        text=bar_df["reflectivity_score"].round(3),
        textposition="outside",
        hovertemplate="<b>%{x}</b><br>Score: %{y:.4f}<br><extra></extra>",
    ))
    fig_bar.add_hline(y=ALERT_CRITICAL, line_dash="dot", line_color="#ef4444")
    fig_bar.add_hline(y=ALERT_WARNING,  line_dash="dot", line_color="#f97316")
    fig_bar.add_hline(y=ALERT_GOOD,     line_dash="dot", line_color="#22c55e")
    fig_bar.update_layout(
        height=280,
        **PLOTLY_THEME,
        showlegend=False,
    )

    fig_bar.update_yaxes(
        title="Reflectivity Score",
        range=[0, 1.05]
    )

    st.plotly_chart(fig_bar, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — Spectral Analysis
# ══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.markdown('<div class="section-header">Hyperspectral Reflectance Curves (400–700 nm)</div>', unsafe_allow_html=True)

    col_ctrl, col_chart = st.columns([1, 3])
    with col_ctrl:
        sp_material  = st.selectbox("Road Material", list(["new_asphalt","aged_asphalt","worn_asphalt","cracked_asphalt","road_marking_new","road_marking_faded","concrete"]))
        sp_weather   = st.selectbox("Weather",       ["clear","haze","rain","heavy_rain","fog"])
        sp_age       = st.slider("Age Factor", 0.0, 1.0, 0.3, 0.05)
        sp_dirt      = st.slider("Dirt Level", 0.0, 1.0, 0.1, 0.05)
        sp_wear      = st.slider("Wear Level", 0.0, 1.0, 0.2, 0.05)

        result = engine.analyze_segment(sp_material, sp_weather, sp_age, sp_dirt, sp_wear)
        score  = result["reflectivity_score"]
        status = reflectivity_to_status(score)

        st.markdown(f"""
        <div style="background:#0f172a;border:1px solid {status['color']}44;border-radius:10px;padding:16px;margin-top:12px;text-align:center;">
          <div style="font-size:2.2rem;font-family:Rajdhani;font-weight:700;color:{status['color']}">{score:.3f}</div>
          <div style="color:{status['color']};font-size:0.9rem;margin-top:4px;">{status['emoji']} {status['label']}</div>
          <div style="color:#64748b;font-size:0.75rem;margin-top:6px;">{status['action']}</div>
        </div>
        """, unsafe_allow_html=True)

    with col_chart:
        bands     = np.array(result["bands"])
        raw_curve = np.array(result["raw_curve"])
        noisy     = np.array(result["noisy_curve"])

        # Compare multiple conditions
        compare_results = {}
        for cond in ["clear", "rain", "fog", "haze"]:
            r = engine.analyze_segment(sp_material, cond, sp_age, sp_dirt, sp_wear)
            compare_results[cond] = np.array(r["noisy_curve"])

        fig_sp = go.Figure()

        # Visible spectrum color background regions
        spectrum_colors = [
            (400, 440, "rgba(100,0,200,0.04)"),
            (440, 490, "rgba(0,0,255,0.04)"),
            (490, 560, "rgba(0,200,0,0.04)"),
            (560, 590, "rgba(200,200,0,0.04)"),
            (590, 625, "rgba(255,120,0,0.04)"),
            (625, 700, "rgba(255,0,0,0.04)"),
        ]
        for lo, hi, clr in spectrum_colors:
            fig_sp.add_vrect(x0=lo, x1=hi, fillcolor=clr, line_width=0)

        # Ideal (raw) curve
        fig_sp.add_trace(go.Scatter(
            x=bands, y=raw_curve,
            name="Ideal (no noise)",
            line=dict(color="#64748b", dash="dot", width=1.5),
        ))

        # Condition comparison curves
        cond_colors = {"clear":"#22c55e", "rain":"#3b82f6", "fog":"#94a3b8", "haze":"#f59e0b"}
        for cond, curve in compare_results.items():
            fig_sp.add_trace(go.Scatter(
                x=bands, y=curve,
                name=f"{cond.title()}",
                line=dict(color=cond_colors[cond], width=2),
                fill="tozeroy" if cond == sp_weather else None,
                fillcolor=f"rgba{tuple(int(cond_colors[cond].lstrip('#')[i:i+2], 16) for i in (0, 2, 4)) + (0.08,)}",
            ))

        fig_sp.update_layout(
            title=f"Spectral Reflectance — {sp_material.replace('_',' ').title()} | Age={sp_age:.2f}",
            height=380,
            **PLOTLY_THEME,
            legend=dict(
                bgcolor="#0f172a",
                bordercolor="#1e293b",
                font=dict(color="#94a3b8")
            ),
        )

        fig_sp.update_xaxes(
            title="Wavelength (nm)",
            tickvals=list(range(400, 710, 50)),
            gridcolor="#1e293b",
        )

        fig_sp.update_yaxes(
            title="Reflectance [0–1]",
            range=[0, 1.05]
        )

        st.plotly_chart(fig_sp, use_container_width=True)

    # Spectral comparison across materials
    st.markdown('<div class="section-header">Material Spectral Fingerprints</div>', unsafe_allow_html=True)

    fig_fingerprint = go.Figure()
    mat_colors = ["#22c55e","#3b82f6","#f59e0b","#ef4444","#a855f7","#06b6d4","#f97316"]
    for i, mat in enumerate(["new_asphalt","aged_asphalt","road_marking_new","road_marking_faded","concrete","worn_asphalt","cracked_asphalt"]):
        r = engine.analyze_segment(mat, "clear", 0.0)
        fig_fingerprint.add_trace(go.Scatter(
            x=np.array(r["bands"]), y=np.array(r["raw_curve"]),
            name=mat.replace("_"," ").title(),
            line=dict(color=mat_colors[i], width=2),
        ))
    fig_fingerprint.update_layout(
        xaxis_title="Wavelength (nm)", yaxis_title="Reflectance [0–1]",
        height=300,
        **PLOTLY_THEME,
        legend=dict(bgcolor="#0f172a", bordercolor="#1e293b", font=dict(color="#94a3b8")),
    )
    st.plotly_chart(fig_fingerprint, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — Degradation Forecast
# ══════════════════════════════════════════════════════════════════════════════
with tab4:
    st.markdown('<div class="section-header">Reflectivity Degradation Predictions (ARIMA)</div>', unsafe_allow_html=True)

    fc_seg = st.selectbox("Select Segment for Forecast", seg_ids, key="fc_seg")
    fc_days = st.slider("Forecast Horizon (days)", 14, FORECAST_DAYS, 30)

    seg_hist_fc = hist_df[hist_df["segment_id"] == fc_seg].sort_values("timestamp")
    fc_data     = forecasts.get(fc_seg, {})

    if fc_data and len(seg_hist_fc):
        last_date = seg_hist_fc["timestamp"].max()
        fc_dates  = pd.date_range(start=last_date + timedelta(days=1), periods=fc_days, freq="D")
        fc_mean   = fc_data["mean"][:fc_days]
        fc_lower  = fc_data["lower"][:fc_days]
        fc_upper  = fc_data["upper"][:fc_days]

        fig_fc = go.Figure()

        # Threshold bands
        fig_fc.add_hrect(y0=0,             y1=ALERT_CRITICAL, fillcolor="rgba(239,68,68,0.08)", line_width=0, annotation_text="CRITICAL ZONE", annotation_font_color="#ef4444")
        fig_fc.add_hrect(y0=ALERT_CRITICAL,y1=ALERT_WARNING,  fillcolor="rgba(249,115,22,0.08)", line_width=0, annotation_text="WARNING ZONE",  annotation_font_color="#f97316", annotation_position="bottom right")
        fig_fc.add_hrect(y0=ALERT_WARNING, y1=ALERT_GOOD,     fillcolor="rgba(245,158,11,0.08)", line_width=0)
        fig_fc.add_hrect(y0=ALERT_GOOD,    y1=1.0,            fillcolor="rgba(34,197,94,0.08)", line_width=0)

        # Historical
        hist_window = seg_hist_fc.tail(60)
        fig_fc.add_trace(go.Scatter(
            x=hist_window["timestamp"],
            y=hist_window["reflectivity_score"],
            name="Historical",
            line=dict(color="#3b82f6", width=2.5),
        ))

        # CI band
        fig_fc.add_trace(go.Scatter(
            x=list(fc_dates) + list(fc_dates[::-1]),
            y=list(fc_upper) + list(fc_lower[::-1]),
            fill="toself",
            fillcolor="rgba(245,158,11,0.15)",
            line=dict(color="rgba(0,0,0,0)"),
            name="90% CI",
            hoverinfo="skip",
        ))
        # Forecast mean
        fig_fc.add_trace(go.Scatter(
            x=fc_dates, y=fc_mean,
            name="Forecast (ARIMA)",
            line=dict(color="#f59e0b", width=2.5, dash="dash"),
            mode="lines+markers",
            marker=dict(
                color=[STATUS_COLORS.get(reflectivity_to_status(v)["label"]) for v in fc_mean],
                size=7, symbol="circle",
            ),
        ))

        # When does it hit critical?
        breach_idx = next((i for i, v in enumerate(fc_mean) if v < ALERT_CRITICAL), None)
        if breach_idx is not None:
            fig_fc.add_vline(
                x=fc_dates[breach_idx].timestamp() * 1000,
                line_dash="dot", line_color="#ef4444", line_width=2,
                annotation_text=f"⚠️ CRITICAL in {breach_idx+1}d",
                annotation_font_color="#ef4444",
            )

        fig_fc.update_layout(
            title=f"Degradation Forecast — {fc_seg} | Horizon: {fc_days} days",
            height=400,
            **PLOTLY_THEME,
            legend=dict(
                bgcolor="#0f172a",
                bordercolor="#1e293b",
                font=dict(color="#94a3b8")
            ),
        )

        fig_fc.update_xaxes(title="Date")

        fig_fc.update_yaxes(
            title="Reflectivity Score",
            range=[0, 1.05]
        )

        st.plotly_chart(fig_fc, use_container_width=True)

        # Forecast summary table
        summary_cols = st.columns(4)
        fc_scores   = {7: fc_mean[6] if len(fc_mean) > 6 else None,
                       14: fc_mean[13] if len(fc_mean) > 13 else None,
                       30: fc_mean[29] if len(fc_mean) > 29 else None}
        for i, (days_out, pred_score) in enumerate(fc_scores.items()):
            with summary_cols[i]:
                if pred_score:
                    s = reflectivity_to_status(pred_score)
                    st.metric(f"In {days_out} Days", f"{pred_score:.3f}", delta=s['label'])

    # Multi-segment forecast comparison
    st.markdown('<div class="section-header">30-Day Forecast — All Segments</div>', unsafe_allow_html=True)

    fig_multi = go.Figure()
    for sid in seg_ids[:10]:  # Show first 10 for clarity
        fc = forecasts.get(sid, {})
        if fc and "mean" in fc:
            last_dt = hist_df[hist_df["segment_id"]==sid]["timestamp"].max()
            fd = pd.date_range(start=last_dt + timedelta(days=1), periods=30, freq="D")
            curr = summary_df[summary_df["segment_id"]==sid]["reflectivity_score"].values
            curr_score_val = float(curr[0]) if len(curr) else 0.5
            color = STATUS_COLORS.get(reflectivity_to_status(curr_score_val)["label"], "#3b82f6")
            fig_multi.add_trace(go.Scatter(
                x=fd, y=fc["mean"][:30],
                name=sid,
                line=dict(color=color, width=1.5),
                opacity=0.7,
            ))

    fig_multi.add_hline(y=ALERT_CRITICAL, line_dash="dot", line_color="#ef4444")
    fig_multi.add_hline(y=ALERT_WARNING,  line_dash="dot", line_color="#f97316")
    fig_multi.update_layout(
        height=300,
        **PLOTLY_THEME,
        legend=dict(
            bgcolor="#0f172a",
            bordercolor="#1e293b",
            font=dict(color="#94a3b8"),
            orientation="h",
            y=-0.2
        ),
    )

    fig_multi.update_yaxes(
        title="Reflectivity Score",
        range=[0, 1]
    )

    st.plotly_chart(fig_multi, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 5 — Maintenance Planner
# ══════════════════════════════════════════════════════════════════════════════
with tab5:
    st.markdown('<div class="section-header">🔧 Intelligent Maintenance Planner</div>', unsafe_allow_html=True)

    budget_input = st.number_input(
        "Annual Maintenance Budget (INR)", value=50_000_000, step=5_000_000,
        format="%d", key="budget"
    )

    seg_lengths = {
        row["segment_id"]: round(row["end_km"] - row["start_km"], 1)
        for _, row in summary_df.iterrows()
    }

    recs = pipeline.get_recommendations(summary_df, seg_lengths)
    budget_stats = pipeline.scheduler.budget_analysis(recs, budget_input)

    # Budget KPIs
    b1, b2, b3, b4 = st.columns(4)
    with b1:
        total_cost = budget_stats.get("total_recommended_cost", 0)
        st.metric("Total Recommended Spend", f"₹{total_cost:,.0f}")
    with b2:
        imm_cost = budget_stats.get("immediate_spend_required", 0)
        st.metric("Immediate Required", f"₹{imm_cost:,.0f}")
    with b3:
        util_pct = budget_stats.get("budget_utilization_pct", 0)
        st.metric("Budget Utilization", f"{util_pct:.1f}%")
    with b4:
        n_imm = budget_stats.get("segments_in_immediate", 0)
        st.metric("Segments (IMMEDIATE)", n_imm, delta="🚨 Urgent" if n_imm > 0 else "✅ None")

    st.markdown('<div class="section-header">Priority Maintenance Queue</div>', unsafe_allow_html=True)

    if len(recs):
        # Style recommendations table
        def color_urgency(val):
            colors = {"IMMEDIATE": "background:#ef44441a;color:#ef4444",
                      "HIGH":      "background:#f9731618;color:#f97316",
                      "MEDIUM":    "background:#f59e0b18;color:#f59e0b",
                      "LOW":       "background:#22c55e18;color:#22c55e"}
            return colors.get(val, "")

        display_recs = recs[[
            "segment_id","current_score","recommended_action",
            "urgency","days_to_critical","post_maintenance_r","estimated_cost_inr"
        ]].copy()
        display_recs.columns = [
            "Segment","Score","Action","Urgency",
            "Days→Critical","Post-Maint R","Est. Cost (₹)"
        ]
        display_recs["Est. Cost (₹)"] = display_recs["Est. Cost (₹)"].apply(lambda x: f"₹{x:,.0f}")

        st.dataframe(
            display_recs,
            use_container_width=True,
            hide_index=True,
            height=350,
        )

        # Cost waterfall chart
        st.markdown('<div class="section-header">Cost Breakdown by Urgency</div>', unsafe_allow_html=True)

        cost_by_urgency = recs.groupby("urgency")["estimated_cost_inr"].sum().reset_index()
        urgency_order   = {"IMMEDIATE": 0, "HIGH": 1, "MEDIUM": 2, "LOW": 3}
        cost_by_urgency["_order"] = cost_by_urgency["urgency"].map(urgency_order)
        cost_by_urgency = cost_by_urgency.sort_values("_order")

        fig_cost = go.Figure(go.Bar(
            x=cost_by_urgency["urgency"],
            y=cost_by_urgency["estimated_cost_inr"],
            marker_color=[{"IMMEDIATE":"#ef4444","HIGH":"#f97316","MEDIUM":"#f59e0b","LOW":"#22c55e"}.get(u,"#3b82f6")
                          for u in cost_by_urgency["urgency"]],
            text=[f"₹{v:,.0f}" for v in cost_by_urgency["estimated_cost_inr"]],
            textposition="outside",
        ))
        fig_cost.add_hline(y=budget_input, line_dash="dot", line_color="#3b82f6",
                           annotation_text="Annual Budget", annotation_font_color="#3b82f6")
        fig_cost.update_layout(
            height=280,
            yaxis_title="Total Cost (INR)",
            **PLOTLY_THEME,
        )
        st.plotly_chart(fig_cost, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 6 — AI Explainability
# ══════════════════════════════════════════════════════════════════════════════
with tab6:
    st.markdown('<div class="section-header">🧠 AI Model Explainability</div>', unsafe_allow_html=True)

    col_imp, col_metrics = st.columns([2, 1])

    with col_imp:
        st.markdown('<div class="section-header">Feature Importance (Gradient Boosting)</div>', unsafe_allow_html=True)

        feat_imp = predictor.get_feature_importance()

        fig_fi = go.Figure(go.Bar(
            y=feat_imp["feature"][::-1],
            x=feat_imp["importance"][::-1],
            orientation="h",
            marker=dict(
                color=feat_imp["importance"][::-1],
                colorscale=[[0,"#1e40af"],[0.5,"#f59e0b"],[1,"#ef4444"]],
                showscale=True,
            ),
        ))
        fig_fi.update_layout(
            height=420,
            xaxis_title="Relative Importance",
            **PLOTLY_THEME,
        )
        st.plotly_chart(fig_fi, use_container_width=True)

    with col_metrics:
        st.markdown('<div class="section-header">Model Metrics</div>', unsafe_allow_html=True)

        m = sys_data["predictor_metrics"]
        metrics_data = {
            "Train MAE": m.get("train_mae", "N/A"),
            "Test MAE":  m.get("test_mae",  "N/A"),
            "Train R²":  m.get("train_r2",  "N/A"),
            "Test R²":   m.get("test_r2",   "N/A"),
        }
        for name, val in metrics_data.items():
            color = "#22c55e" if isinstance(val, float) and val > 0.85 else "#f59e0b"
            st.markdown(f"""
            <div style="background:#0f172a;border:1px solid #1e293b;border-radius:8px;
                        padding:12px 16px;margin-bottom:8px;display:flex;
                        justify-content:space-between;align-items:center;">
              <span style="color:#94a3b8;font-size:0.9rem;">{name}</span>
              <span style="color:{color};font-weight:600;font-family:Rajdhani;font-size:1.3rem;">{val}</span>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("---")
        st.markdown('<div class="section-header">Architecture</div>', unsafe_allow_html=True)
        st.markdown("""
        <div style="font-size:0.8rem;color:#64748b;line-height:1.8;">
        <b style="color:#94a3b8;">Model 1</b> — GradientBoostingRegressor<br>
        • 200 estimators, lr=0.05<br>
        • 15 engineered features<br>
        • Visual + Spectral + Environmental<br><br>
        <b style="color:#94a3b8;">Model 2</b> — CNN Feature Extractor<br>
        • MobileNet-style depthwise separable<br>
        • 128-dim road image embeddings<br><br>
        <b style="color:#94a3b8;">Model 3</b> — ARIMA(2,d,2) per segment<br>
        • ADF test for auto-differencing<br>
        • 90% confidence intervals<br>
        </div>
        """, unsafe_allow_html=True)

    # Correlation heatmap of features
    st.markdown('<div class="section-header">Feature Correlation Matrix</div>', unsafe_allow_html=True)

    train_df = sys_data["train_df"]
    from modules.model import FEATURE_COLS
    feat_cols_avail = [c for c in FEATURE_COLS if c in train_df.columns]
    corr_matrix = train_df[feat_cols_avail + ["reflectivity_score"]].corr()

    fig_corr = go.Figure(go.Heatmap(
        z=corr_matrix.values,
        x=corr_matrix.columns.tolist(),
        y=corr_matrix.columns.tolist(),
        colorscale=[[0,"#1e40af"],[0.5,"#0f172a"],[1,"#ef4444"]],
        text=corr_matrix.round(2).values,
        texttemplate="%{text}",
        textfont=dict(size=9, color="white"),
        zmid=0, zmin=-1, zmax=1,
        showscale=True,
    ))
    fig_corr.update_layout(
        height=380,
        **PLOTLY_THEME,
    )

    fig_corr.update_xaxes(tickfont=dict(size=9))
    fig_corr.update_yaxes(tickfont=dict(size=9))

    st.plotly_chart(fig_corr, use_container_width=True)


# ─────────────────────────────────────────────────────────────────────────────
# Footer
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<div style="margin-top:40px;padding:20px;border-top:1px solid #1e293b;text-align:center;color:#334155;font-size:0.8rem;">
  <b style="color:#f59e0b;">Safar AI</b> · <i>Har Safar, Surakshit Safar</i><br>
  Built for NHAI Hackathon 2024 | Python + OpenCV + TensorFlow + Streamlit<br>
  🛣️ Har Safar, Surakshit Safar — Powered by Hyperspectral Vision · Digital Twin · ARIMA/LSTM · Edge AI
</div>
""", unsafe_allow_html=True)

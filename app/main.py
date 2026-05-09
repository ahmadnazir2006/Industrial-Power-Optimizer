import numpy as np
import joblib
from tensorflow.keras.models import load_model
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
import os
import streamlit as st
import plotly.graph_objects as go
import time

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ============================================================
#  PAGE CONFIG
# ============================================================
st.set_page_config(
    page_title="IPO · Industrial Power Optimizer",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
#  CUSTOM CSS  —  modern dark theme, glassmorphism, gradients
# ============================================================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }

    .stApp {
        background:
            radial-gradient(ellipse at top left, rgba(99, 102, 241, 0.15), transparent 50%),
            radial-gradient(ellipse at bottom right, rgba(168, 85, 247, 0.12), transparent 50%),
            linear-gradient(135deg, #0a0e27 0%, #0f1535 50%, #0a0e27 100%);
        background-attachment: fixed;
        color: #e2e8f0;
    }

    #MainMenu, footer, header { visibility: hidden; }

    .block-container {
        padding-top: 2rem;
        padding-bottom: 3rem;
        max-width: 1400px;
    }

    /* ---------- Hero header ---------- */
    .hero-container {
        background: linear-gradient(135deg, rgba(99, 102, 241, 0.12), rgba(168, 85, 247, 0.08));
        border: 1px solid rgba(99, 102, 241, 0.25);
        border-radius: 24px;
        padding: 2.5rem 3rem;
        margin-bottom: 2rem;
        backdrop-filter: blur(20px);
        position: relative;
        overflow: hidden;
        box-shadow: 0 8px 40px rgba(0, 0, 0, 0.3);
    }
    .hero-container::before {
        content: '';
        position: absolute;
        top: -50%;
        right: -10%;
        width: 500px;
        height: 500px;
        background: radial-gradient(circle, rgba(99, 102, 241, 0.25), transparent 70%);
        border-radius: 50%;
        pointer-events: none;
    }
    .hero-container::after {
        content: '';
        position: absolute;
        bottom: -30%;
        left: -10%;
        width: 400px;
        height: 400px;
        background: radial-gradient(circle, rgba(236, 72, 153, 0.18), transparent 70%);
        border-radius: 50%;
        pointer-events: none;
    }
    .hero-pill {
        display: inline-flex;
        align-items: center;
        padding: 0.4rem 1rem;
        background: rgba(16, 185, 129, 0.12);
        border: 1px solid rgba(16, 185, 129, 0.35);
        border-radius: 999px;
        color: #6ee7b7;
        font-size: 0.8rem;
        font-weight: 600;
        margin-bottom: 1.25rem;
        position: relative;
        z-index: 1;
        letter-spacing: 0.02em;
    }
    .hero-title {
        font-size: 3.25rem;
        font-weight: 800;
        background: linear-gradient(135deg, #60a5fa 0%, #a78bfa 50%, #f472b6 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin: 0;
        letter-spacing: -0.03em;
        line-height: 1.1;
        position: relative;
        z-index: 1;
    }
    .hero-subtitle {
        color: #94a3b8;
        font-size: 1.15rem;
        font-weight: 400;
        margin-top: 0.75rem;
        max-width: 700px;
        line-height: 1.6;
        position: relative;
        z-index: 1;
    }

    /* ---------- Pulsing dot ---------- */
    .pulse-dot {
        display: inline-block;
        width: 8px;
        height: 8px;
        background: #10b981;
        border-radius: 50%;
        margin-right: 8px;
        box-shadow: 0 0 0 0 rgba(16, 185, 129, 0.7);
        animation: pulse 2s infinite;
    }
    .pulse-dot.red { background: #ef4444; box-shadow: 0 0 0 0 rgba(239, 68, 68, 0.7); animation: pulse-red 2s infinite; }
    @keyframes pulse {
        0%   { box-shadow: 0 0 0 0 rgba(16, 185, 129, 0.7); }
        70%  { box-shadow: 0 0 0 12px rgba(16, 185, 129, 0); }
        100% { box-shadow: 0 0 0 0 rgba(16, 185, 129, 0); }
    }
    @keyframes pulse-red {
        0%   { box-shadow: 0 0 0 0 rgba(239, 68, 68, 0.7); }
        70%  { box-shadow: 0 0 0 12px rgba(239, 68, 68, 0); }
        100% { box-shadow: 0 0 0 0 rgba(239, 68, 68, 0); }
    }

    /* ---------- Metric cards ---------- */
    .metric-card {
        background: linear-gradient(135deg, rgba(30, 41, 59, 0.6), rgba(15, 23, 42, 0.6));
        border: 1px solid rgba(148, 163, 184, 0.12);
        border-radius: 16px;
        padding: 1.5rem;
        backdrop-filter: blur(12px);
        transition: all 0.3s ease;
        height: 100%;
        position: relative;
        overflow: hidden;
    }
    .metric-card::before {
        content: '';
        position: absolute;
        top: 0; left: 0;
        width: 100%;
        height: 3px;
        background: linear-gradient(90deg, #6366f1, #8b5cf6, #ec4899);
        opacity: 0.6;
    }
    .metric-card:hover {
        border-color: rgba(99, 102, 241, 0.5);
        transform: translateY(-3px);
        box-shadow: 0 12px 40px rgba(99, 102, 241, 0.2);
    }
    .metric-label {
        color: #94a3b8;
        font-size: 0.75rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        margin-bottom: 0.5rem;
    }
    .metric-value {
        color: #f8fafc;
        font-size: 2rem;
        font-weight: 700;
        font-family: 'JetBrains Mono', monospace;
        line-height: 1.1;
        margin-bottom: 0.5rem;
    }
    .metric-delta { font-size: 0.8rem; font-weight: 500; color: #64748b; }
    .metric-delta.positive { color: #10b981; }
    .metric-delta.negative { color: #f87171; }
    .metric-delta.warning  { color: #fbbf24; }

    /* ---------- Sidebar ---------- */
    [data-testid="stSidebar"] {
        background: rgba(10, 14, 39, 0.85) !important;
        border-right: 1px solid rgba(148, 163, 184, 0.1);
        backdrop-filter: blur(20px);
    }
    [data-testid="stSidebar"] h3 { color: #f1f5f9 !important; font-weight: 700; }
    [data-testid="stSidebar"] label { color: #cbd5e1 !important; font-weight: 600; }

    /* ---------- Selectbox ---------- */
    .stSelectbox div[data-baseweb="select"] > div {
        background: rgba(30, 41, 59, 0.6) !important;
        border: 1px solid rgba(148, 163, 184, 0.2) !important;
        border-radius: 10px !important;
        color: #f1f5f9 !important;
    }

    /* ---------- Tabs ---------- */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background: rgba(30, 41, 59, 0.4);
        padding: 8px;
        border-radius: 14px;
        border: 1px solid rgba(148, 163, 184, 0.12);
    }
    .stTabs [data-baseweb="tab"] {
        background: transparent;
        border-radius: 10px;
        color: #94a3b8;
        font-weight: 600;
        padding: 10px 24px;
        border: none;
    }
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #6366f1, #8b5cf6) !important;
        color: white !important;
        box-shadow: 0 4px 14px rgba(99, 102, 241, 0.4);
    }

    /* ---------- Buttons ---------- */
    .stButton > button {
        background: linear-gradient(135deg, #6366f1, #8b5cf6);
        color: white;
        border: none;
        border-radius: 12px;
        padding: 0.7rem 1.75rem;
        font-weight: 600;
        font-size: 0.95rem;
        transition: all 0.3s ease;
        box-shadow: 0 4px 14px rgba(99, 102, 241, 0.35);
        width: 100%;
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 24px rgba(99, 102, 241, 0.55);
        background: linear-gradient(135deg, #7c83f7, #a78bfa);
    }

    /* ---------- Alert cards ---------- */
    .alert-card {
        border-radius: 14px;
        padding: 1.25rem 1.5rem;
        margin: 1rem 0;
        backdrop-filter: blur(10px);
        border-left: 4px solid;
        font-size: 0.95rem;
        line-height: 1.6;
    }
    .alert-critical { background: rgba(239, 68, 68, 0.1);  border-color: #ef4444; color: #fecaca; }
    .alert-warning  { background: rgba(245, 158, 11, 0.1); border-color: #f59e0b; color: #fde68a; }
    .alert-success  { background: rgba(16, 185, 129, 0.1); border-color: #10b981; color: #a7f3d0; }
    .alert-info     { background: rgba(59, 130, 246, 0.1); border-color: #3b82f6; color: #bfdbfe; }

    /* ---------- Section header ---------- */
    .section-header {
        color: #f8fafc;
        font-size: 1.4rem;
        font-weight: 700;
        margin: 2rem 0 1rem 0;
        display: flex;
        align-items: center;
        gap: 0.75rem;
        letter-spacing: -0.01em;
    }
    .section-header::before {
        content: '';
        width: 4px;
        height: 26px;
        background: linear-gradient(180deg, #6366f1, #ec4899);
        border-radius: 4px;
    }

    /* ---------- Status panel in sidebar ---------- */
    .status-panel {
        padding: 1rem;
        background: linear-gradient(135deg, rgba(16, 185, 129, 0.1), rgba(16, 185, 129, 0.05));
        border-radius: 12px;
        border: 1px solid rgba(16, 185, 129, 0.25);
    }
    .config-list {
        font-size: 0.85rem;
        color: #94a3b8;
        line-height: 1.9;
    }
    .config-list b { color: #cbd5e1; }

    /* ---------- Date input ---------- */
    .stDateInput label { color: #cbd5e1 !important; font-weight: 600; }
    .stDateInput div[data-baseweb="input"] {
        background: rgba(30, 41, 59, 0.6) !important;
        border-radius: 10px !important;
    }

    /* ---------- DataFrame ---------- */
    .stDataFrame { border-radius: 12px; overflow: hidden; }

    h1, h2, h3, h4, h5, h6 { color: #f8fafc !important; }
    .stMarkdown p { color: #cbd5e1; }

    /* Expander */
    .streamlit-expanderHeader {
        background: rgba(30, 41, 59, 0.5) !important;
        border-radius: 10px !important;
        font-weight: 600 !important;
    }
</style>
""", unsafe_allow_html=True)

MODELS = {

    "GRU(Gated Recurrent Unit)": {
        "file": "gru.h5",
        "scaler": "gru_scaler.pkl",
        "results": "gru_results.csv"
    },

    "Stacked LSTM": {
        "file": "lstm.h5",
        "scaler": "lstm_scaler.pkl",
        "results": "lstm_results.csv"
    },

    "CNN-LSTM Hybrid": {
        "file": "cnn_lstm.h5",
        "scaler": "cnn_lstm_scaler.pkl",
        "results": "cnn_lstm_results.csv"
    }

}

# ============================================================
#  HERO HEADER
# ============================================================
st.markdown("""
<div class="hero-container">
    <div class="hero-pill"><span class="pulse-dot"></span>AI SYSTEM ONLINE</div>
    <h1 class="hero-title">⚡ Industrial Power Optimizer</h1>
    <p class="hero-subtitle">
        Deep learning–powered peak demand forecasting and intelligent load scheduling
        for manufacturing facilities. Reduce demand charges through predictive intervention.
    </p>
</div>
""", unsafe_allow_html=True)

# ============================================================
#  SIDEBAR
# ============================================================
with st.sidebar:
    st.markdown("### 🧠 AI Configuration")

    selected_model_name = st.selectbox(
        "Neural Architecture",
        list(MODELS.keys()),
        help="Pick the deep learning model used for forecasting."
    )

    st.markdown("---")
    st.markdown("### 📡 System Status")
    st.markdown("""
    <div class="status-panel">
        <div style="color: #6ee7b7; font-weight: 600; font-size: 0.9rem;">
            <span class="pulse-dot"></span>All Systems Operational
        </div>
        <div style="color: #94a3b8; font-size: 0.78rem; margin-top: 0.4rem;">
            Models loaded · Data streams ready
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### ⚙️ Runtime Parameters")
    st.markdown("""
    <div class="config-list">
        <b>Threshold:</b> 100 kWh<br>
        <b>Cost rate:</b> $10 / excess kWh<br>
        <b>History window:</b> 24 h (96 steps)<br>
        <b>Sample interval:</b> 15 minutes<br>
        <b>Forecast horizon:</b> 24 hours
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("""
    <div style="font-size: 0.72rem; color: #64748b; text-align: center; margin-top: 1rem; line-height: 1.6;">
        Built by <b style="color: #94a3b8;">Ahmad Nazir</b><br>
        FAST-NUCES, Lahore
    </div>
    """, unsafe_allow_html=True)


@st.cache_resource
def load_ai_assets(name):

    model_info = MODELS[name]

    model_path = os.path.join(
        BASE_DIR,
        "models",
        model_info["file"]
    )

    scaler_path = os.path.join(
        BASE_DIR,
        "models",
        model_info["scaler"]
    )


    m = tf.keras.models.load_model(model_path)

    s = joblib.load(scaler_path)



    return m, s

MODEL, SCALER = load_ai_assets(selected_model_name)

tab1, tab2 = st.tabs(["📊  Historical Dashboard", "🚀  Live Monitoring"])

with tab1:

    st.markdown('<div class="section-header">Historical Energy Consumption</div>', unsafe_allow_html=True)

    st.markdown(
        '<p style="color: #94a3b8; font-size: 0.95rem; line-height: 1.6;">'
        'Explore historical energy consumption for December 2018. '
        'Select a date range to analyze actual vs predicted usage and identify critical load events.'
        '</p>',
        unsafe_allow_html=True
    )

    selected_results_file = MODELS[selected_model_name]["results"]

    DATA_PATH = os.path.join(
        BASE_DIR,
        "data",
        "results",
        selected_results_file
    )

    try:

        df = pd.read_csv(
            DATA_PATH,
            parse_dates=['date'],
            index_col='date'
        )

        df.index = pd.to_datetime(df.index)

    except FileNotFoundError:

        st.error(f"Data file not found at {DATA_PATH}")

        st.stop()

    except Exception as e:

        st.error(f"Error loading project files: {e}")

        st.stop()

    Global_Threshold = 100

    start_date = pd.to_datetime('2018-12-01 00:00:00')

    end_date = pd.to_datetime('2018-12-31 23:45:00')

    final_month_data = df.loc[start_date:end_date]

    col_filter, col_spacer = st.columns([2, 3])
    with col_filter:
        start_date, end_date = st.date_input(

            "📅 Select Analysis Range",

            value=(
                final_month_data.index.min().date(),
                final_month_data.index.max().date()
            ),

            min_value=final_month_data.index.min().date(),

            max_value=final_month_data.index.max().date()

        )

    start_date = pd.to_datetime(start_date)

    end_date = pd.to_datetime(end_date)

    daily_data = final_month_data.loc[start_date:end_date]

    peaks = daily_data[
        daily_data['Predicted Usage (kWh)'] > Global_Threshold
    ]

    if not peaks.empty:

        st.markdown('<div class="section-header">⚠️ Critical Load Analysis</div>', unsafe_allow_html=True)

        excess_kwh = (
            peaks['Predicted Usage (kWh)']
            - Global_Threshold
        )

        additional_cost = excess_kwh.sum() * 10

        c1, c2, c3, c4 = st.columns(4)

        with c1:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Peak Events</div>
                <div class="metric-value">{len(peaks)}</div>
                <div class="metric-delta warning">⚠ Threshold breaches</div>
            </div>
            """, unsafe_allow_html=True)

        with c2:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Avg Excess Load</div>
                <div class="metric-value">{excess_kwh.mean():.2f}</div>
                <div class="metric-delta">kWh per event</div>
            </div>
            """, unsafe_allow_html=True)

        with c3:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Maximum Excess</div>
                <div class="metric-value">{excess_kwh.max():.2f}</div>
                <div class="metric-delta negative">↑ Highest peak</div>
            </div>
            """, unsafe_allow_html=True)

        with c4:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Estimated Cost</div>
                <div class="metric-value">${additional_cost:,.0f}</div>
                <div class="metric-delta negative">Additional charges</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("""
        <div class="alert-card alert-info" style="margin-top: 1.5rem;">
            <b>💡 AI Recommendation:</b> Shift heavy machinery loads to off-peak hours
            to reduce demand charges. Review peak timestamps below for optimal load-shift planning.
        </div>
        """, unsafe_allow_html=True)

        with st.expander("📋  View Detailed Peak Timestamps"):

            st.dataframe(
                peaks[['Predicted Usage (kWh)']],
                use_container_width=True
            )

    else:

        st.markdown("""
        <div class="alert-card alert-success">
            <b>✅ All systems stable.</b> No abnormal usage patterns detected in the selected range.
        </div>
        """, unsafe_allow_html=True)

        st.snow()

    st.markdown('<div class="section-header">📈 Consumption Forecast</div>', unsafe_allow_html=True)

    fig = go.Figure()

    fig.add_trace(

        go.Scatter(

            x=daily_data.index,

            y=daily_data['Actual Usage (kWh)'],

            mode='lines',

            name="Actual Usage",

            line=dict(
                dash='dash',
                color='#f472b6',
                width=2
            ),

            hovertemplate='<b>Actual</b>: %{y:.2f} kWh<extra></extra>'

        )

    )

    fig.add_trace(

        go.Scatter(

            x=daily_data.index,

            y=daily_data['Predicted Usage (kWh)'],

            mode='lines',

            name="AI Forecast",

            line=dict(
                dash='solid',
                color='#60a5fa',
                width=2.5
            ),

            hovertemplate='<b>Predicted</b>: %{y:.2f} kWh<extra></extra>'

        )

    )

    fig.add_hline(
        y=Global_Threshold,
        line_dash="dot",
        line_color="rgba(239, 68, 68, 0.5)",
        annotation_text=f"  Threshold: {Global_Threshold} kWh  ",
        annotation_position="top right",
        annotation_font_color="#fecaca",
        annotation_font_size=11
    )

    fig.update_layout(

        title=dict(
            text=f"Energy Consumption · {start_date.strftime('%b %d')} → {end_date.strftime('%b %d, %Y')}",
            font=dict(color='#f1f5f9', size=18, family='Inter')
        ),

        xaxis_title="Time",

        yaxis_title="Electricity Usage (kWh)",

        template="plotly_dark",

        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#cbd5e1', family='Inter'),

        hovermode='x unified',

        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            bgcolor='rgba(30, 41, 59, 0.6)',
            bordercolor='rgba(148, 163, 184, 0.2)',
            borderwidth=1,
            font=dict(color='#e2e8f0')
        ),

        margin=dict(t=80, b=40, l=40, r=40),
        height=500

    )

    fig.update_xaxes(gridcolor='rgba(148, 163, 184, 0.08)', linecolor='rgba(148, 163, 184, 0.2)')
    fig.update_yaxes(gridcolor='rgba(148, 163, 184, 0.08)', linecolor='rgba(148, 163, 184, 0.2)', autorange=True)

    st.plotly_chart(fig, use_container_width=True)


with tab2:
    st.markdown('<div class="section-header">Live Energy Stream</div>', unsafe_allow_html=True)
    st.markdown(
        '<p style="color: #94a3b8; font-size: 0.95rem; line-height: 1.6;">'
        'Real-time AI-powered energy monitoring with predictive load-shifting recommendations. '
        'Click below to begin streaming sensor data through the model.'
        '</p>',
        unsafe_allow_html=True
    )

    col_btn, col_status = st.columns([1, 3])
    with col_btn:
        start_stream = st.button("🚀  Launch Live Stream")
    with col_status:
        st.markdown("""
        <div style="padding: 0.75rem 1rem; background: rgba(99, 102, 241, 0.1); border: 1px solid rgba(99, 102, 241, 0.3); border-radius: 12px; color: #a5b4fc; font-size: 0.88rem;">
            <span class="pulse-dot"></span><b>Stream Engine:</b> Ready · 96-step rolling window · 15-min sampling
        </div>
        """, unsafe_allow_html=True)

    FULL_DATA_PATH = os.path.join(BASE_DIR, "data","processed_deployment.csv")

    try:
        df_sim = pd.read_csv(FULL_DATA_PATH, parse_dates=['date'], index_col='date')
    except Exception as e:
        st.error("Engineered data file not found. Simulation cannot run.")
        st.stop()

    container = st.container()

    with container:
        chart_placeholder = st.empty()
        metrics_placeholder = st.empty()
        alert_placeholder = st.empty()
        suggestion_placeholder = st.empty()

        if start_stream:
            st.markdown("""
            <div class="alert-card alert-success">
                <b>🟢 Live stream initiated.</b> AI is processing real-time sensor data...
            </div>
            """, unsafe_allow_html=True)

            feature_cols = [
                'Usage_kWh', 'Lagging_Current_Reactive.Power_kVarh',
                'NSM_sin', 'NSM_cos', 'Day_of_week_sin', 'Day_of_week_cos', 'WeekStatus'
            ]

            sim_source = df_sim.iloc[1000:1500]

            for i in range(96, len(sim_source)):
                current_window = sim_source.iloc[i-96:i]

                scaled_input = SCALER.transform(current_window[feature_cols])

                reshaped_input = scaled_input.reshape(1, 96, 7)

                raw_prediction = MODEL.predict(reshaped_input, verbose=0)

                dummy = np.zeros((96, 7))
                dummy[:, 0] = raw_prediction.flatten()
                inv_prediction = SCALER.inverse_transform(dummy)[:, 0]
                future_peak = inv_prediction.max()

                actual_so_far = sim_source['Usage_kWh'].iloc[:i]

                fig = go.Figure()

                fig.add_trace(go.Scatter(
                    x=actual_so_far.index[-100:],
                    y=actual_so_far.values[-100:],
                    name="Sensor (Past)",
                    line=dict(color='#fb923c', width=2.5),
                    hovertemplate='<b>Past</b>: %{y:.2f} kWh<extra></extra>'
                ))

                future_dates = pd.date_range(start=actual_so_far.index[-1], periods=96, freq='15min')
                fig.add_trace(go.Scatter(
                    x=future_dates, y=inv_prediction,
                    name="AI Forecast (Future)",
                    line=dict(color='#22d3ee', dash='dash', width=2.5),
                    hovertemplate='<b>Forecast</b>: %{y:.2f} kWh<extra></extra>'
                ))

                fig.update_layout(
                    template="plotly_dark",
                    height=520,
                    title=dict(
                        text="🔴 Real-Time Smart Grid Forecast",
                        font=dict(color='#f1f5f9', size=18, family='Inter')
                    ),
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    font=dict(color='#cbd5e1', family='Inter'),
                    hovermode='x unified',
                    legend=dict(
                        orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1,
                        bgcolor='rgba(30, 41, 59, 0.6)',
                        bordercolor='rgba(148, 163, 184, 0.2)',
                        borderwidth=1,
                        font=dict(color='#e2e8f0')
                    ),
                    margin=dict(t=80, b=40, l=40, r=40)
                )
                fig.update_xaxes(gridcolor='rgba(148, 163, 184, 0.08)')
                fig.update_yaxes(gridcolor='rgba(148, 163, 184, 0.08)')

                chart_placeholder.plotly_chart(fig, use_container_width=True)

                live_load = actual_so_far.iloc[-1]
                metrics_placeholder.markdown(f"""
                <div class="metric-card" style="margin: 1rem 0;">
                    <div class="metric-label"><span class="pulse-dot red"></span>Live Load Reading</div>
                    <div class="metric-value">{live_load:.2f} <span style="font-size: 1rem; color: #94a3b8; font-weight: 400;">kWh</span></div>
                    <div class="metric-delta">Current sensor sample</div>
                </div>
                """, unsafe_allow_html=True)

                future_peak = inv_prediction.max()
                peak_index = np.argmax(inv_prediction)

                future_peak = inv_prediction.max()

                if future_peak > 120:
                    alert_placeholder.markdown(f"""
                    <div class="alert-card alert-critical">
                        <b>🔴 CRITICAL ALERT</b> · Predicted peak <b>{future_peak:.2f} kWh</b> — immediate intervention recommended.
                    </div>
                    """, unsafe_allow_html=True)
                elif future_peak > 100:
                    alert_placeholder.markdown(f"""
                    <div class="alert-card alert-warning">
                        <b>🟠 HIGH LOAD WARNING</b> · Predicted peak <b>{future_peak:.2f} kWh</b> — consider load redistribution.
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    alert_placeholder.markdown(f"""
                    <div class="alert-card alert-success">
                        <b>🟢 NORMAL LOAD</b> · Predicted peak <b>{future_peak:.2f} kWh</b> — operations within safe range.
                    </div>
                    """, unsafe_allow_html=True)

                time_window = 12
                current_time_index = i

                peak_idx = np.argmax(inv_prediction)
                valley_idx = np.argmin(inv_prediction)

                peak_value = inv_prediction[peak_idx]
                valley_value = inv_prediction[valley_idx]

                peak_time = future_dates[peak_idx]
                valley_time = future_dates[valley_idx]

                cost_per_kwh = 10
                factor = 0.20

                savings = (peak_value - valley_value) * factor * cost_per_kwh

                date_str = peak_time.strftime('%d-%m-%Y')
                peak_str = peak_time.strftime('%H:%M')
                valley_str = valley_time.strftime('%H:%M')

                time_diff = peak_idx - current_time_index

                if time_diff < 0:
                    urgency = "⚠️ PEAK ALREADY IN PROGRESS"
                elif time_diff < time_window:
                    urgency = "🚨 TAKE ACTION NOW"
                else:
                    urgency = "📅 PLAN AHEAD"

                suggestion_placeholder.markdown(f"""
                <div class="alert-card alert-info">
                    <b>⚡ Optimizer Suggestion · {urgency}</b><br><br>
                    Move high-load activities from
                    <b style="color: #f472b6;">{peak_str}</b> to
                    <b style="color: #34d399;">{valley_str}</b> on
                    <b>{date_str}</b> to save approximately
                    <b style="color: #facc15;">${savings:.2f}</b>
                    on today's peak charge. Take action before <b>{peak_str}</b>.
                </div>
                """, unsafe_allow_html=True)

                time.sleep(1)

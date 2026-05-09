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
    page_title="Industrial Power Optimizer",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
#  STYLES — Apple-inspired: SF Pro, restrained palette,
#  subtle borders, no glows, no pulse animations.
# ============================================================
st.markdown("""
<style>
    html, body, [class*="css"], button, input, select, textarea {
        font-family: -apple-system, BlinkMacSystemFont, "SF Pro Display", "SF Pro Text",
                     "Inter", "Helvetica Neue", "Segoe UI", sans-serif;
        -webkit-font-smoothing: antialiased;
        -moz-osx-font-smoothing: grayscale;
    }

    .stApp {
        background: #000000;
        color: #f5f5f7;
    }

    #MainMenu, footer, header { visibility: hidden; }

    .block-container {
        padding-top: 2.5rem;
        padding-bottom: 4rem;
        max-width: 1280px;
    }

    /* ---------- Hero ---------- */
    .hero-container {
        padding: 3.5rem 0 3rem 0;
        margin-bottom: 1.5rem;
        border-bottom: 1px solid rgba(255, 255, 255, 0.08);
    }
    .hero-eyebrow {
        display: inline-flex;
        align-items: center;
        padding: 0.35rem 0.85rem;
        background: rgba(10, 132, 255, 0.1);
        border: 1px solid rgba(10, 132, 255, 0.22);
        border-radius: 999px;
        color: #64a8ff;
        font-size: 0.75rem;
        font-weight: 600;
        letter-spacing: 0.02em;
        margin-bottom: 1.5rem;
    }
    .hero-title {
        font-size: 3rem;
        font-weight: 600;
        color: #f5f5f7;
        margin: 0;
        letter-spacing: -0.025em;
        line-height: 1.05;
        background: linear-gradient(180deg, #f5f5f7 0%, #b8b8bd 120%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    .hero-subtitle {
        color: rgba(235, 235, 245, 0.6);
        font-size: 1.1rem;
        font-weight: 400;
        margin-top: 1rem;
        max-width: 640px;
        line-height: 1.5;
        letter-spacing: -0.005em;
    }

    /* ---------- Status indicator (no animation) ---------- */
    .dot {
        display: inline-block;
        width: 6px; height: 6px;
        border-radius: 50%;
        margin-right: 7px;
        vertical-align: middle;
        background: #30d158;
    }
    .dot.red    { background: #ff453a; }
    .dot.orange { background: #ff9f0a; }
    .dot.blue   { background: #0a84ff; }

    /* ---------- Sidebar ---------- */
    [data-testid="stSidebar"] {
        background: #0a0a0a !important;
        border-right: 1px solid rgba(255, 255, 255, 0.06);
    }
    [data-testid="stSidebar"] .block-container {
        padding-top: 2rem;
    }
    [data-testid="stSidebar"] hr {
        border-color: rgba(255, 255, 255, 0.06);
        margin: 1.25rem 0;
    }
    .sidebar-eyebrow {
        font-size: 0.72rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        color: rgba(235, 235, 245, 0.45);
        margin-bottom: 0.6rem;
    }
    [data-testid="stSidebar"] label {
        color: #f5f5f7 !important;
        font-weight: 500;
        font-size: 0.88rem;
    }

    /* ---------- Selectbox ---------- */
    .stSelectbox div[data-baseweb="select"] > div {
        background: #1c1c1e !important;
        border: 1px solid rgba(255, 255, 255, 0.08) !important;
        border-radius: 10px !important;
        color: #f5f5f7 !important;
        min-height: 42px;
    }
    .stSelectbox svg { fill: rgba(235, 235, 245, 0.5) !important; }

    /* ---------- Date input ---------- */
    .stDateInput div[data-baseweb="input"] {
        background: #1c1c1e !important;
        border: 1px solid rgba(255, 255, 255, 0.08) !important;
        border-radius: 10px !important;
    }

    /* ---------- Tabs (segmented control) ---------- */
    .stTabs [data-baseweb="tab-list"] {
        gap: 2px;
        background: rgba(255, 255, 255, 0.06);
        padding: 3px;
        border-radius: 10px;
        border: none;
        width: fit-content;
        margin-top: 0.5rem;
    }
    .stTabs [data-baseweb="tab"] {
        background: transparent;
        border-radius: 7px;
        color: rgba(235, 235, 245, 0.6);
        font-weight: 500;
        padding: 7px 18px;
        font-size: 0.9rem;
        border: none;
        min-height: 32px;
    }
    .stTabs [aria-selected="true"] {
        background: #2c2c2e !important;
        color: #f5f5f7 !important;
        box-shadow: 0 1px 2px rgba(0, 0, 0, 0.3);
    }
    .stTabs [data-baseweb="tab-highlight"] { display: none; }
    .stTabs [data-baseweb="tab-border"] { display: none; }

    /* ---------- Buttons ---------- */
    .stButton > button {
        background: #0a84ff;
        color: white;
        border: none;
        border-radius: 10px;
        padding: 0.55rem 1.3rem;
        font-weight: 500;
        font-size: 0.9rem;
        transition: background 0.15s ease;
        box-shadow: none;
        height: auto;
        min-height: 38px;
    }
    .stButton > button:hover {
        background: #339cff;
        transform: none;
        box-shadow: none;
        color: white;
    }
    .stButton > button:active {
        background: #0866cc;
    }
    .stButton > button:focus { box-shadow: none; outline: none; }

    /* ---------- Section header ---------- */
    .section-eyebrow {
        color: rgba(235, 235, 245, 0.45);
        font-size: 0.72rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        margin: 2.5rem 0 0.5rem 0;
    }
    .section-title {
        color: #f5f5f7;
        font-size: 1.5rem;
        font-weight: 600;
        margin: 0 0 1rem 0;
        letter-spacing: -0.02em;
    }

    .body-text {
        color: rgba(235, 235, 245, 0.6);
        font-size: 0.95rem;
        line-height: 1.55;
        max-width: 720px;
        font-weight: 400;
    }

    /* ---------- Metric cards ---------- */
    .metric-card {
        background: #1c1c1e;
        border: 1px solid rgba(255, 255, 255, 0.06);
        border-radius: 14px;
        padding: 1.5rem;
        height: 100%;
        transition: border-color 0.2s ease;
    }
    .metric-card:hover {
        border-color: rgba(255, 255, 255, 0.14);
    }
    .metric-label {
        color: rgba(235, 235, 245, 0.55);
        font-size: 0.82rem;
        font-weight: 500;
        margin-bottom: 0.65rem;
        letter-spacing: -0.005em;
    }
    .metric-value {
        color: #f5f5f7;
        font-size: 2rem;
        font-weight: 600;
        font-variant-numeric: tabular-nums;
        letter-spacing: -0.025em;
        line-height: 1;
        margin-bottom: 0.6rem;
    }
    .metric-unit {
        color: rgba(235, 235, 245, 0.4);
        font-size: 0.95rem;
        font-weight: 500;
        margin-left: 0.25rem;
        letter-spacing: -0.005em;
    }
    .metric-delta {
        font-size: 0.78rem;
        font-weight: 500;
        color: rgba(235, 235, 245, 0.45);
    }
    .metric-delta.positive { color: #30d158; }
    .metric-delta.negative { color: #ff453a; }
    .metric-delta.warning  { color: #ff9f0a; }

    /* ---------- Alert cards ---------- */
    .alert-card {
        border-radius: 12px;
        padding: 0.95rem 1.15rem;
        margin: 0.75rem 0;
        border: 1px solid;
        font-size: 0.9rem;
        line-height: 1.55;
        display: flex;
        align-items: flex-start;
        gap: 0.75rem;
    }
    .alert-card .dot {
        margin-top: 7px;
        margin-right: 0;
        flex-shrink: 0;
    }
    .alert-critical { background: rgba(255, 69, 58, 0.07);  border-color: rgba(255, 69, 58, 0.2);  color: #ff8b80; }
    .alert-warning  { background: rgba(255, 159, 10, 0.07); border-color: rgba(255, 159, 10, 0.22); color: #ffb547; }
    .alert-success  { background: rgba(48, 209, 88, 0.07);  border-color: rgba(48, 209, 88, 0.2);  color: #6ee190; }
    .alert-info     { background: rgba(10, 132, 255, 0.07); border-color: rgba(10, 132, 255, 0.2); color: #64a8ff; }
    .alert-card b { color: inherit; font-weight: 600; }

    /* ---------- Status panel ---------- */
    .status-panel {
        padding: 0.85rem 1rem;
        background: #1c1c1e;
        border-radius: 10px;
        border: 1px solid rgba(255, 255, 255, 0.06);
    }
    .status-panel-title {
        color: #f5f5f7; font-weight: 500; font-size: 0.88rem;
        display: flex; align-items: center;
    }
    .status-panel-sub {
        color: rgba(235, 235, 245, 0.5);
        font-size: 0.78rem;
        margin-top: 0.3rem;
        margin-left: 13px;
    }

    .config-row {
        display: flex;
        justify-content: space-between;
        padding: 0.4rem 0;
        font-size: 0.85rem;
        border-bottom: 1px solid rgba(255, 255, 255, 0.05);
    }
    .config-row:last-child { border-bottom: none; }
    .config-row .k { color: rgba(235, 235, 245, 0.55); font-weight: 500; }
    .config-row .v { color: #f5f5f7; font-weight: 500; font-variant-numeric: tabular-nums; }

    .live-badge {
        display: inline-flex;
        align-items: center;
        padding: 0.55rem 1rem;
        background: #1c1c1e;
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 10px;
        color: rgba(235, 235, 245, 0.7);
        font-size: 0.85rem;
        font-weight: 500;
    }

    .author-credit {
        font-size: 0.75rem;
        color: rgba(235, 235, 245, 0.4);
        text-align: center;
        margin-top: 1.5rem;
        line-height: 1.8;
    }
    .author-credit a {
        color: rgba(235, 235, 245, 0.7);
        font-weight: 500;
        text-decoration: none;
        border-bottom: 1px solid rgba(235, 235, 245, 0.2);
        padding-bottom: 1px;
        transition: color 0.15s ease, border-color 0.15s ease;
    }
    .author-credit a:hover {
        color: #0a84ff;
        border-color: #0a84ff;
    }

    h1, h2, h3, h4, h5, h6 { color: #f5f5f7 !important; font-weight: 600; }
    .stMarkdown p { color: rgba(235, 235, 245, 0.65); }

    /* DataFrame */
    .stDataFrame { border-radius: 12px; overflow: hidden; }
    div[data-testid="stExpander"] {
        background: #1c1c1e;
        border: 1px solid rgba(255, 255, 255, 0.06);
        border-radius: 12px;
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
#  PLOTLY THEME HELPER
# ============================================================
def apple_layout(title=None, height=460):
    return dict(
        title=dict(
            text=title or "",
            font=dict(color='#f5f5f7', size=15, family='-apple-system, SF Pro Display, Inter, sans-serif'),
            x=0, xanchor='left', pad=dict(l=10, t=8)
        ),
        template='none',
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='rgba(235, 235, 245, 0.7)', family='-apple-system, SF Pro Display, Inter, sans-serif', size=12),
        hovermode='x unified',
        hoverlabel=dict(
            bgcolor='rgba(28, 28, 30, 0.95)',
            bordercolor='rgba(255, 255, 255, 0.12)',
            font=dict(color='#f5f5f7', family='-apple-system, SF Pro Display, Inter, sans-serif', size=12)
        ),
        margin=dict(t=60, b=45, l=55, r=30),
        height=height,
        legend=dict(
            orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1,
            bgcolor='rgba(0, 0, 0, 0)', bordercolor='rgba(0, 0, 0, 0)',
            font=dict(color='rgba(235, 235, 245, 0.75)', size=11)
        )
    )


def apple_axes(fig):
    fig.update_xaxes(
        gridcolor='rgba(255, 255, 255, 0.04)',
        linecolor='rgba(255, 255, 255, 0.08)',
        zeroline=False, showspikes=False,
        tickfont=dict(color='rgba(235, 235, 245, 0.55)', size=11)
    )
    fig.update_yaxes(
        gridcolor='rgba(255, 255, 255, 0.04)',
        linecolor='rgba(255, 255, 255, 0.08)',
        zeroline=False,
        tickfont=dict(color='rgba(235, 235, 245, 0.55)', size=11)
    )
    return fig


# ============================================================
#  HERO
# ============================================================
st.markdown("""
<div class="hero-container">
    <div class="hero-eyebrow"><span class="dot"></span>System online</div>
    <h1 class="hero-title">Industrial Power Optimizer</h1>
    <p class="hero-subtitle">
        Deep learning forecasting and intelligent load scheduling for manufacturing.
        Reduce demand charges through predictive intervention.
    </p>
</div>
""", unsafe_allow_html=True)

# ============================================================
#  SIDEBAR
# ============================================================
with st.sidebar:
    st.markdown('<div class="sidebar-eyebrow">Model</div>', unsafe_allow_html=True)

    selected_model_name = st.selectbox(
        "Neural architecture",
        list(MODELS.keys()),
        label_visibility="collapsed"
    )

    st.markdown("---")
    st.markdown('<div class="sidebar-eyebrow">Status</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="status-panel">
        <div class="status-panel-title"><span class="dot"></span>Operational</div>
        <div class="status-panel-sub">Models loaded · Streams ready</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown('<div class="sidebar-eyebrow">Parameters</div>', unsafe_allow_html=True)
    st.markdown("""
    <div>
        <div class="config-row"><span class="k">Threshold</span><span class="v">100 kWh</span></div>
        <div class="config-row"><span class="k">Cost rate</span><span class="v">$10 / kWh</span></div>
        <div class="config-row"><span class="k">Window</span><span class="v">24 h</span></div>
        <div class="config-row"><span class="k">Interval</span><span class="v">15 min</span></div>
        <div class="config-row"><span class="k">Horizon</span><span class="v">24 h</span></div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("""
    <div class="author-credit">
        Built by<br>
        <a href="https://www.linkedin.com/in/ahmad-nazir-75752225b/" target="_blank">Ahmad Nazir</a>
        &amp;
        <a href="https://www.linkedin.com/in/abdur-rafay-khan/" target="_blank">Abdur Rafay Khan</a><br>
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

tab1, tab2 = st.tabs(["Historical", "Live Monitoring"])

with tab1:

    st.markdown('<div class="section-eyebrow">Overview</div>', unsafe_allow_html=True)
    st.markdown('<h2 class="section-title">Historical energy consumption</h2>', unsafe_allow_html=True)
    st.markdown(
        '<p class="body-text">'
        'Explore historical energy consumption for December 2018. '
        'Select a date range to analyze actual versus predicted usage and identify critical load events.'
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

            "Date range",

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

        st.markdown('<div class="section-eyebrow">Critical load analysis</div>', unsafe_allow_html=True)
        st.markdown('<h2 class="section-title">Demand exceeds threshold</h2>', unsafe_allow_html=True)

        excess_kwh = (
            peaks['Predicted Usage (kWh)']
            - Global_Threshold
        )

        additional_cost = excess_kwh.sum() * 10

        c1, c2, c3, c4 = st.columns(4)

        with c1:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Peak events</div>
                <div class="metric-value">{len(peaks)}</div>
                <div class="metric-delta warning">Threshold breaches</div>
            </div>
            """, unsafe_allow_html=True)

        with c2:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Average excess</div>
                <div class="metric-value">{excess_kwh.mean():.2f}<span class="metric-unit">kWh</span></div>
                <div class="metric-delta">Per event</div>
            </div>
            """, unsafe_allow_html=True)

        with c3:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Maximum excess</div>
                <div class="metric-value">{excess_kwh.max():.2f}<span class="metric-unit">kWh</span></div>
                <div class="metric-delta negative">Highest single peak</div>
            </div>
            """, unsafe_allow_html=True)

        with c4:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Estimated cost</div>
                <div class="metric-value">${additional_cost:,.0f}</div>
                <div class="metric-delta negative">Additional charges</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("""
        <div class="alert-card alert-info">
            <span class="dot blue"></span>
            <div><b>Recommendation.</b> Shift heavy machinery loads to off-peak hours to reduce demand charges. Review peak timestamps below for optimal load-shift planning.</div>
        </div>
        """, unsafe_allow_html=True)

        with st.expander("Peak timestamps"):

            st.dataframe(
                peaks[['Predicted Usage (kWh)']],
                use_container_width=True
            )

    else:

        st.markdown("""
        <div class="alert-card alert-success">
            <span class="dot"></span>
            <div><b>All systems stable.</b> No abnormal usage patterns detected in the selected range.</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown('<div class="section-eyebrow">Forecast</div>', unsafe_allow_html=True)
    st.markdown('<h2 class="section-title">Consumption over time</h2>', unsafe_allow_html=True)

    fig = go.Figure()

    fig.add_trace(

        go.Scatter(

            x=daily_data.index,

            y=daily_data['Actual Usage (kWh)'],

            mode='lines',

            name="Actual",

            line=dict(
                color='rgba(235, 235, 245, 0.4)',
                width=1.6
            ),

            hovertemplate='Actual: %{y:.2f} kWh<extra></extra>'

        )

    )

    fig.add_trace(

        go.Scatter(

            x=daily_data.index,

            y=daily_data['Predicted Usage (kWh)'],

            mode='lines',

            name="Forecast",

            line=dict(
                color='#0a84ff',
                width=2
            ),

            hovertemplate='Forecast: %{y:.2f} kWh<extra></extra>'

        )

    )

    fig.add_hline(
        y=Global_Threshold,
        line_dash="dot",
        line_color="rgba(255, 69, 58, 0.45)",
        line_width=1,
        annotation_text=f"  Threshold {Global_Threshold} kWh  ",
        annotation_position="top right",
        annotation_font_color="rgba(255, 139, 128, 0.9)",
        annotation_font_size=10
    )

    fig.update_layout(**apple_layout(
        title=f"{start_date.strftime('%b %d')} – {end_date.strftime('%b %d, %Y')}",
        height=460
    ))

    fig.update_yaxes(title_text="kWh")
    apple_axes(fig)

    st.plotly_chart(fig, use_container_width=True)


with tab2:
    st.markdown('<div class="section-eyebrow">Real-time</div>', unsafe_allow_html=True)
    st.markdown('<h2 class="section-title">Live energy stream</h2>', unsafe_allow_html=True)
    st.markdown(
        '<p class="body-text">'
        'Real-time energy monitoring with predictive load-shifting recommendations. '
        'Start the stream to begin processing sensor data through the model.'
        '</p>',
        unsafe_allow_html=True
    )

    col_btn, col_status = st.columns([1, 3])
    with col_btn:
        start_stream = st.button("Launch stream")
    with col_status:
        st.markdown("""
        <div class="live-badge">
            <span class="dot blue"></span>96-step rolling window · 15-minute sampling · ready
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
                <span class="dot"></span>
                <div><b>Stream initiated.</b> Model is processing sensor data.</div>
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
                    name="Sensor (past)",
                    line=dict(color='rgba(235, 235, 245, 0.55)', width=1.8),
                    hovertemplate='Past: %{y:.2f} kWh<extra></extra>'
                ))

                future_dates = pd.date_range(start=actual_so_far.index[-1], periods=96, freq='15min')
                fig.add_trace(go.Scatter(
                    x=future_dates, y=inv_prediction,
                    name="Forecast",
                    line=dict(color='#0a84ff', dash='dash', width=2),
                    hovertemplate='Forecast: %{y:.2f} kWh<extra></extra>'
                ))

                fig.update_layout(**apple_layout(title="Real-time forecast", height=480))
                fig.update_yaxes(title_text="kWh")
                apple_axes(fig)

                chart_placeholder.plotly_chart(fig, use_container_width=True)

                live_load = actual_so_far.iloc[-1]
                metrics_placeholder.markdown(f"""
                <div class="metric-card" style="margin: 1rem 0; max-width: 320px;">
                    <div class="metric-label">Live load</div>
                    <div class="metric-value">{live_load:.2f}<span class="metric-unit">kWh</span></div>
                    <div class="metric-delta"><span class="dot red"></span>Live sensor sample</div>
                </div>
                """, unsafe_allow_html=True)

                future_peak = inv_prediction.max()
                peak_index = np.argmax(inv_prediction)

                future_peak = inv_prediction.max()

                if future_peak > 120:
                    alert_placeholder.markdown(f"""
                    <div class="alert-card alert-critical">
                        <span class="dot red"></span>
                        <div><b>Critical alert.</b> Predicted peak {future_peak:.2f} kWh — immediate intervention recommended.</div>
                    </div>
                    """, unsafe_allow_html=True)
                elif future_peak > 100:
                    alert_placeholder.markdown(f"""
                    <div class="alert-card alert-warning">
                        <span class="dot orange"></span>
                        <div><b>High load warning.</b> Predicted peak {future_peak:.2f} kWh — consider load redistribution.</div>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    alert_placeholder.markdown(f"""
                    <div class="alert-card alert-success">
                        <span class="dot"></span>
                        <div><b>Normal load.</b> Predicted peak {future_peak:.2f} kWh — operations within safe range.</div>
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

                date_str = peak_time.strftime('%d %b %Y')
                peak_str = peak_time.strftime('%H:%M')
                valley_str = valley_time.strftime('%H:%M')

                time_diff = peak_idx - current_time_index

                if time_diff < 0:
                    urgency = "Peak in progress"
                elif time_diff < time_window:
                    urgency = "Take action now"
                else:
                    urgency = "Plan ahead"

                suggestion_placeholder.markdown(f"""
                <div class="alert-card alert-info">
                    <span class="dot blue"></span>
                    <div>
                        <b>Optimizer suggestion · {urgency}.</b>
                        Move high-load activities from <b>{peak_str}</b> to <b>{valley_str}</b> on <b>{date_str}</b>
                        to save approximately <b>${savings:.2f}</b> on today's peak charge. Action window closes at <b>{peak_str}</b>.
                    </div>
                </div>
                """, unsafe_allow_html=True)

                time.sleep(1)

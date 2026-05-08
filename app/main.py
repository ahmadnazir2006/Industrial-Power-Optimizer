import numpy as np
import joblib
from tensorflow.keras.models import load_model
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
import os
import streamlit as st
import plotly.graph_objects as go

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

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

st.sidebar.header("🧠 AI Configuration")

selected_model_name = st.sidebar.selectbox(
    "Select AI Architecture",
    list(MODELS.keys())
)

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

st.title("⚡ Energy Consumer Explorer")

start_date = pd.to_datetime('2018-12-01 00:00:00')

end_date = pd.to_datetime('2018-12-31 23:45:00')

final_month_data = df.loc[start_date:end_date]

start_date, end_date = st.date_input(

    "Select date range",

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

    st.error("⚠️ CRITICAL LOAD ALERT")

    col1, col2 = st.columns(2)

    excess_kwh = (
        peaks['Predicted Usage (kWh)']
        - Global_Threshold
    )

    with col1:

        st.metric(
            "Peak Load Detected",
            f"{len(peaks)} times"
        )

        st.metric(
            "Average Excess Load",
            f"{excess_kwh.mean():.2f} kWh"
        )

        st.metric(
            "Maximum Excess Load",
            f"{excess_kwh.max():.2f} kWh"
        )

    with col2:

        additional_cost = excess_kwh.sum() * 10

        st.metric(
            "Estimated Additional Cost",
            f"${additional_cost:.2f}"
        )

        st.warning(
            "Predicted energy usage exceeded the safety threshold."
        )

    with st.expander("View Detailed Peak Timestamps"):

        st.dataframe(
            peaks[['Predicted Usage (kWh)']]
        )

    st.info(
        "💡 Recommendation: Shift heavy machinery loads "
        "to off-peak hours."
    )

else:

    st.success(
        "All systems stable ⚡ No abnormal usage detected!"
    )

    st.snow()

fig = go.Figure()

fig.add_trace(

    go.Scatter(

        x=daily_data.index,

        y=daily_data['Actual Usage (kWh)'],

        mode='lines',

        name="Actual Usage",

        line=dict(
            dash='dash',
            color='red'
        )

    )

)

fig.add_trace(

    go.Scatter(

        x=daily_data.index,

        y=daily_data['Predicted Usage (kWh)'],

        mode='lines',

        name="AI Predicted Usage",

        line=dict(
            dash='solid',
            color='blue'
        )

    )

)

fig.update_layout(

    title=f"Energy Consumption: "
          f"{start_date.strftime('%Y-%m-%d')} "
          f"to {end_date.strftime('%Y-%m-%d')}",

    xaxis_title="Time",

    yaxis_title="Electricity Usage (kWh)",

    template="plotly_white",

    hovermode='x unified',

    legend=dict(
        orientation="h"
    )

)

fig.update_yaxes(autorange=True)

st.plotly_chart(fig, use_container_width=True)
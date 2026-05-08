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

tab1,tab2=st.tabs(["📊 Historical Dashboard",
    "🚀 Live Monitoring"])

with tab1:

    st.subheader("Historical Energy Consumption")

    st.markdown(
         "This dashboard allows you to explore the historical energy consumption "
         "data for December 2018. You can select a date range to analyze the "
         "actual vs predicted energy usage and identify any critical load events."
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


with tab2:
    start_stream = st.button("🚀 Launch Live Stream")
    st.subheader("Live Energy Consumption Monitoring")

    FULL_DATA_PATH = os.path.join(BASE_DIR, "data", "raw", "processed", "processed_deployment.csv")
    
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

    if start_stream:
        st.success("Live stream started! AI is processing sensor data...")
        
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
                name="Sensor (Past)", line=dict(color='orange')))
            
            future_dates = pd.date_range(start=actual_so_far.index[-1], periods=96, freq='15min')
            fig.add_trace(go.Scatter(
                x=future_dates, y=inv_prediction,
                name="AI Forecast (Future)", line=dict(color='cyan', dash='dash')))

            fig.update_layout(template="plotly_dark", height=500, title="Real-Time Smart Grid Forecast")

           
   
            chart_placeholder.plotly_chart(fig, use_container_width=True)
            metrics_placeholder.metric("Live Load", f"{actual_so_far.iloc[-1]:.2f} kWh")
            
            future_peak = inv_prediction.max()
            peak_index = np.argmax(inv_prediction)

            future_peak = inv_prediction.max()

            if future_peak > 120:
                alert_placeholder.error(f"🔴 CRITICAL ALERT: {future_peak:.2f} kWh predicted")
            elif future_peak > 100:
                alert_placeholder.warning(f"🟠 HIGH LOAD WARNING: {future_peak:.2f} kWh predicted")
            else:
                alert_placeholder.success(f"🟢 NORMAL LOAD: {future_peak:.2f} kWh predicted")

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

            st.info(
                f"⚡ **Optimizer Suggestion ({urgency})**: Move high-load activities from "
                f"**{peak_str}** to **{valley_str}** on **{date_str}** "
                f"to save approximately {savings:.2f}$ on today's peak charge. "
                f"Take action before {peak_str}"
            )

            time.sleep(1)
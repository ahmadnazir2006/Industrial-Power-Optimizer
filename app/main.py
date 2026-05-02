import numpy as np
import joblib
from tensorflow.keras.models import load_model
import pandas as pd
import matplotlib.pyplot as plt 
model=load_model('models/industrial_lstm_model.keras')
scaler=joblib.load('models/scaler.pkl')
import streamlit as st
import plotly.graph_objects as go


import os


BASE_DIR = os.path.dirname(__file__)

MODEL_PATH = os.path.join(BASE_DIR, "..", "models", "industrial_lstm_model.keras")

model = load_model(MODEL_PATH)
if not os.path.exists(MODEL_PATH):
    st.error("🚨 **Model File Missing in the Cloud!**")
    st.write(f"The app looked for the model here: `{MODEL_PATH}`")
    st.write("Current working directory:", os.getcwd())
    st.write("Files in root:", os.listdir(os.path.join(BASE_DIR, "..")))
    st.stop() # This halts the app cleanly instead of crashing




Global_Threshold=100

plot_data=pd.read_csv('data/raw/processed/prediction_results.csv',parse_dates=['date'],index_col='date')
plot_data.index = pd.to_datetime(plot_data.index)
st.title("Energy Consumer Explorer")

start_date=pd.to_datetime('2018-12-1 00:00:00')
end_date=pd.to_datetime('2018-12-31 23:45:00')
final_month_data=plot_data.loc[start_date:end_date]

start_date, end_date = st.date_input(
    "Select date range",
    value=(final_month_data.index.min().date(), final_month_data.index.max().date()),
    min_value=final_month_data.index.min().date(),
    max_value=final_month_data.index.max().date()
)



start_date =pd.to_datetime(start_date)
end_date=pd.to_datetime(end_date)


if start_date in final_month_data.index and end_date in final_month_data.index:
    daily_data=final_month_data.loc[start_date:end_date]
    peaks=daily_data[daily_data['Predicted Usage (KWh)']>Global_Threshold]


    if not peaks.empty:
        st.error("⚠️ CRITICAL LOAD ALERT")
        col1,col2=st.columns(2)
        excess_kwh=peaks['Predicted Usage (KWh)']-Global_Threshold
    
        with col1:
            st.metric("Peak Load Detected(total):",f"{len(peaks)} times",delta="High Demand")
            st.metric("Average Excess Load:",f"{excess_kwh.mean():.2f} KWh")
            st.metric("Maximum Excess Load:",f"{excess_kwh.max():.2f} KWh")
        
        with col2:
            additonal_cost=excess_kwh.sum()*10   # Business Logic: Estimate cost (Assuming $10 penalty per peak kWh)
            st.metric("Estimated Additional Cost:",f"${additonal_cost:.2f}",delta_color="inverse")
            st.write("⚠️ The predicted load has exceeded the critical threshold of 100 KWh multiple times during the selected date range. This indicates potential overload conditions that may require immediate attention to prevent equipment damage or operational disruptions.")
        with st.expander("View Detailed Peak Timestamps"):
            st.write("The following 15-minute intervals exceeded the safety threshold:")
            st.dataframe(peaks[['Predicted Usage (KWh)']])
        st.info(f"💡 **Recommendation:** Consider shifting heavy machinery loads from the identified peaks to off-peak hours (11 PM - 6 AM) to reduce demand charges.")
    else:
        st.success("All systems stable ⚡ No abnormal energy usage detected!")
        st.snow()
    fig=go.Figure()
    fig.add_trace(go.Scatter(x=daily_data.index,y=daily_data['Actual Usage (KWh)'],mode='lines'
                             ,name="Actual Usage (KWh)",line=dict(dash='dash',color='red')))
    fig.add_trace(go.Scatter(x=daily_data.index,y=daily_data['Predicted Usage (KWh)'],mode='lines'
                  ,name="AI Predicted Usage (KWh)",line=dict(dash='solid',color='blue')))
    fig.update_layout(title=f"Energy Consumption on {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}",xaxis_title="Time",yaxis_title="Electricity Usage (KWh)" ,
    legend_title="Legend",template="plotly_white",hovermode='x unified',
    legend=dict(orientation="h"))
    fig.update_yaxes(autorange=True)
    st.plotly_chart(fig)
else:
    st.write("No data available for the selected date. Please select a date between December 1, 2018 and December 31, 2018.")





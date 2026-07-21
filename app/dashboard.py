import streamlit as st
import pandas as pd
import xgboost as xgb
import json
import os

# Set working directory to the app folder to resolve relative paths
current_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(current_dir, '../models/xgboost_traffic_model.json')
metrics_path = os.path.join(current_dir, '../results/metrics.json')

@st.cache_resource
def load_model():
    """Loads the pre-trained XGBoost model."""
    model = xgb.XGBRegressor()
    model.load_model(model_path)
    return model

@st.cache_data
def load_metrics():
    """Loads the training metrics."""
    with open(metrics_path, 'r') as f:
        return json.load(f)

# --- Streamlit UI ---
st.set_page_config(page_title="Smart City Traffic Forecaster", layout="wide")
st.title("🏙️ Smart City Traffic Forecaster")

try:
    model = load_model()
    metrics = load_metrics()
    st.sidebar.success(f"Model loaded successfully! (Training RMSE: {metrics['RMSE']})")
except Exception as e:
    st.sidebar.error("Model not found. Please run `src/train_model.py` first.")
    st.stop()

# User Inputs
st.header("Forecast Traffic Volume")
col1, col2, col3, col4 = st.columns(4)

with col1:
    hour = st.slider("Hour of Day", 0, 23, 12)
with col2:
    day = st.slider("Day of Week (0=Mon, 6=Sun)", 0, 6, 2)
with col3:
    weather = st.selectbox("Weather Condition", options=[0, 1, 2], format_func=lambda x: ["Clear", "Rain", "Snow"][x])
with col4:
    event = st.selectbox("Special Event?", options=[0, 1], format_func=lambda x: ["No", "Yes"][x])

# Make Prediction
if st.button("Generate Forecast"):
    input_df = pd.DataFrame([[hour, day, weather, event]], 
                              columns=['hour_of_day', 'day_of_week', 'weather_condition', 'special_event'])
    
    prediction = model.predict(input_df)[0]
    st.metric(label="Predicted Traffic Volume (Vehicles per hour)", value=f"{int(prediction):,}")
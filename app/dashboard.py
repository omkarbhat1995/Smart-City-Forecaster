import streamlit as st
import pandas as pd
import xgboost as xgb
import json
import os

# Anchor the path to the script's physical location
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
st.title("🏙️ Smart City Traffic Forecaster (UCI Real Data)")
st.markdown("Predictive analytics powered by XGBoost. Trained on 48,000+ real interstate observations.")

try:
    model = load_model()
    metrics = load_metrics()
    st.sidebar.success(f"Model loaded successfully! (Training RMSE: {metrics['RMSE']})")
except Exception as e:
    st.sidebar.error("Model not found. Please run `src/train_model.py` first.")
    st.stop()

# User Inputs (Updated with 6 features)
st.header("Forecast Traffic Volume")
col1, col2, col3 = st.columns(3)
col4, col5, col6 = st.columns(3)

with col1:
    hour = st.slider("Hour of Day", 0, 23, 17) # Default to 5 PM
with col2:
    day = st.slider("Day of Week (0=Mon, 6=Sun)", 0, 6, 2)
with col3:
    month = st.slider("Month of Year", 1, 12, 6)

with col4:
    # The UCI dataset records temperature in Kelvin. 
    # 273.15K is 0°C (Freezing). 300K is ~27°C (Warm).
    temp = st.slider("Temperature (Kelvin)", 240.0, 310.0, 290.0)
with col5:
    weather = st.selectbox("Weather Condition", options=[0, 1, 2], format_func=lambda x: ["Clear / Clouds", "Rain / Storms", "Snow / Ice"][x])
with col6:
    event = st.selectbox("Holiday / Special Event?", options=[0, 1], format_func=lambda x: ["No", "Yes"][x])

# Make Prediction
if st.button("Generate Forecast"):
    # WARNING: The column names and order MUST exactly match the training data
    input_df = pd.DataFrame([[hour, day, month, temp, weather, event]], 
                              columns=['hour_of_day', 'day_of_week', 'month', 'temp', 'weather_condition', 'special_event'])
    
    prediction = model.predict(input_df)[0]
    
    # Clip prediction to 0 in case the model predicts negative traffic in extreme edge cases
    final_prediction = max(0, int(prediction))
    
    st.metric(label="Predicted Traffic Volume (Vehicles per hour)", value=f"{final_prediction:,}")
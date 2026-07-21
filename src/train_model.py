import pandas as pd
import numpy as np
import xgboost as xgb
import json
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

def train_and_evaluate():
    """Trains the XGBoost model and saves artifacts."""
    os.makedirs('../models', exist_ok=True)
    os.makedirs('../results', exist_ok=True)
    
    # Load processed data
    df = pd.read_csv('../data/processed/synthetic_traffic_processed.csv')
    
    X = df.drop('traffic_volume', axis=1)
    y = df['traffic_volume']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train Model
    model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100, max_depth=3)
    model.fit(X_train, y_train)
    
    # Evaluate
    predictions = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    
    # Save Model
    model_path = '../models/xgboost_traffic_model.json'
    model.save_model(model_path)
    print(f"Model saved to {model_path}")
    
    # Save Metrics
    metrics = {"RMSE": round(rmse, 2), "n_estimators": 100, "max_depth": 3}
    with open('../results/metrics.json', 'w') as f:
        json.dump(metrics, f, indent=4)
    print(f"Metrics saved to ../results/metrics.json")

if __name__ == "__main__":
    train_and_evaluate()
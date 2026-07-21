import pandas as pd
import numpy as np
import os
from ucimlrepo import fetch_ucirepo

# Anchor the path to the script's physical location
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.join(current_dir, '..')

def download_and_process_uci_data():
    """Fetches real traffic data using ucimlrepo, engineers features, and saves it."""
    
    raw_dir = os.path.join(project_root, 'data', 'raw')
    processed_dir = os.path.join(project_root, 'data', 'processed')
    
    os.makedirs(raw_dir, exist_ok=True)
    os.makedirs(processed_dir, exist_ok=True)
    
    print("Fetching Metro Interstate Traffic Volume dataset from UCI...")
    try:
        # Fetch dataset using the official UCI repo package
        metro_traffic = fetch_ucirepo(id=492) 
        X = metro_traffic.data.features 
        y = metro_traffic.data.targets 
        
        # Combine features and targets into one dataframe for processing
        df_raw = pd.concat([X, y], axis=1)
        
    except Exception as e:
        print(f"Failed to download data: {e}")
        return
        
    # Save Raw Data
    raw_path = os.path.join(raw_dir, 'UCI_Traffic_raw.csv')
    df_raw.to_csv(raw_path, index=False)
    print(f"Raw data saved to {os.path.abspath(raw_path)}")
    
    print("Engineering features...")
    # 1. Parse DateTime to extract time-series features
    df_raw['date_time'] = pd.to_datetime(df_raw['date_time'])
    df_raw['hour_of_day'] = df_raw['date_time'].dt.hour
    df_raw['day_of_week'] = df_raw['date_time'].dt.dayofweek
    df_raw['month'] = df_raw['date_time'].dt.month
    
    # 2. Create Binary flag for holidays/special events
    df_raw['special_event'] = (df_raw['holiday'] != 'None').astype(int)
    
    # 3. Categorize messy weather strings into our 3 numerical buckets
    def map_weather(weather_str):
        w = str(weather_str).lower()
        if 'snow' in w or 'ice' in w: 
            return 2
        elif 'rain' in w or 'drizzle' in w or 'thunderstorm' in w or 'squall' in w: 
            return 1
        else: 
            return 0 # Clear, Clouds, Fog, Mist, etc.
            
    df_raw['weather_condition'] = df_raw['weather_main'].apply(map_weather)
    
    # 4. Select the finalized features for modeling (Adding Temperature)
    # The dataset target column is named 'traffic_volume'
    features = ['hour_of_day', 'day_of_week', 'month', 'temp', 'weather_condition', 'special_event', 'traffic_volume']
    df_processed = df_raw[features]
    
    # Save Processed Data
    processed_path = os.path.join(processed_dir, 'UCI_Traffic_processed.csv')
    df_processed.to_csv(processed_path, index=False)
    print(f"Processed data saved to {os.path.abspath(processed_path)}")

if __name__ == "__main__":
    download_and_process_uci_data()
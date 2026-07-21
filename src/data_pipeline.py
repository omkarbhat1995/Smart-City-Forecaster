import pandas as pd
import numpy as np
import os

def generate_and_save_data(num_records=2500):
    """Generates synthetic data and saves it to the raw and processed directories."""
    # Ensure directories exist
    os.makedirs('../data/raw', exist_ok=True)
    os.makedirs('../data/processed', exist_ok=True)
    
    # 1. Generate Raw Data
    np.random.seed(42)
    data = {
        'hour_of_day': np.random.randint(0, 24, num_records),
        'day_of_week': np.random.randint(0, 7, num_records),
        'weather_condition': np.random.randint(0, 3, num_records), 
        'special_event': np.random.randint(0, 2, num_records), 
    }
    df_raw = pd.DataFrame(data)
    
    # Add target variable (Traffic Volume)
    df_raw['traffic_volume'] = (
        (df_raw['hour_of_day'] * 150) - 
        (df_raw['weather_condition'] * 300) + 
        (df_raw['special_event'] * 800) + 
        np.random.normal(0, 100, num_records)
    ).clip(lower=0) 
    
    # Save Raw Data
    raw_path = '../data/raw/synthetic_traffic_raw.csv'
    df_raw.to_csv(raw_path, index=False)
    print(f"Raw data saved to {raw_path}")
    
    # 2. Preprocess Data (In this case, just passing it through, but you could add scaling here)
    processed_path = '../data/processed/synthetic_traffic_processed.csv'
    df_raw.to_csv(processed_path, index=False)
    print(f"Processed data saved to {processed_path}")

if __name__ == "__main__":
    generate_and_save_data()
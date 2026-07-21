# 🏙️ Smart City Traffic Forecaster

An end-to-end Applied Machine Learning dashboard designed to predict urban traffic volume. This project demonstrates the ability to train, optimize, and serve machine learning models locally using lightweight infrastructure.

## 🚀 Overview

## 📊 Dataset & Features
Instead of relying on static synthetic data, this project ingests real-world traffic observations from the UCI Metro Interstate Traffic Volume dataset (via ucimlrepo).

Engineered Features:

hour_of_day & day_of_week: Extracted from timestamps to capture cyclical commuting patterns and weekend lulls.

month: Tracks seasonal variations in traffic volume.

temp: Ambient temperature (Kelvin), capturing environmental impact on travel behavior.

weather_condition: Categorized numerical encoding (0: Clear/Clouds, 1: Rain/Storms, 2: Snow/Ice).

special_event: Binary indicator mapped from federal and local holidays.

**Key Features:**
* **Predictive Analytics:** Implements `XGBoost` for fast, highly accurate tabular forecasting.
* **Interactive UI:** Fully functional `Streamlit` dashboard for real-time scenario testing.
* **Local Execution:** Optimized to run entirely locally without reliance on cloud compute.

## 🛠️ Technology Stack
🛠️ Technology Stack
* **Language**: Python 3.9+

* **Data Engineering & Retrieval**: Pandas, NumPy, ucimlrepo

* **Machine Learning**: XGBoost, Scikit-Learn

* **Frontend UI**: Streamlit

* **Version Control & Best Practices**: Modular structure, path-anchored scripts (os.path), and robust .gitignore rules.

## ⚙️ Installation & Usage

1. **Clone the repository:**
   ``` bash
   git clone [https://github.com/omkarbhat1995/Smart-City-Forecaster.git](https://github.com/omkarbhat1995/Smart-City-Forecaster.git)
   cd Smart-City-Forecaster
    ```
2. Create and activate a virtual environment:
```
python -m venv venv
# On Windows (PowerShell):
.\venv\Scripts\activate
# On Mac/Linux:
source venv/bin/activate
```    
3. Install dependencies:
It is recommended to use a virtual environment.
```
pip install -r requirements.txt
```
(Note: Ensure your requirements.txt includes streamlit, pandas, numpy, xgboost, and scikit-learn)
4. Run the Data Pipeline:
Fetches the raw UCI dataset and generates the cleaned processed CSV.
```
python src\data_pipeline.py
```
5. Train the Model:
Trains the XGBoost regressor and outputs model artifacts and performance logs
```
python src\train_model.py
```
6. Launch the Dashboard
Boots up the interactive local web server.
```
streamlit run app.py
```
The dashboard will automatically open in your default web browser at http://localhost:8501.

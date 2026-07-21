# 🏙️ Smart City Traffic Forecaster

An end-to-end Applied Machine Learning dashboard designed to predict urban traffic volume. This project demonstrates the ability to train, optimize, and serve machine learning models locally using lightweight infrastructure.

## 🚀 Overview

Data science goes beyond Jupyter notebooks. This repository highlights a production-ready approach to serving predictions. It leverages an XGBoost regressor for rapid CPU-bound training and utilizes Streamlit to deliver an interactive web interface for business stakeholders.

**Key Features:**
* **Predictive Analytics:** Implements `XGBoost` for fast, highly accurate tabular forecasting.
* **Interactive UI:** Fully functional `Streamlit` dashboard for real-time scenario testing.
* **Local Execution:** Optimized to run entirely locally without reliance on cloud compute.

## 🛠️ Technology Stack
* **Language:** Python 3.9+
* **Machine Learning:** XGBoost, Scikit-Learn
* **Data Processing:** Pandas, NumPy
* **Frontend:** Streamlit

## ⚙️ Installation & Usage

1. **Clone the repository:**
   ``` bash
   git clone [https://github.com/omkarbhat1995/Smart-City-Forecaster.git](https://github.com/omkarbhat1995/Smart-City-Forecaster.git)
   cd Smart-City-Forecaster
    ```
2. Install dependencies:
It is recommended to use a virtual environment.
```
pip install -r requirements.txt
```
(Note: Ensure your requirements.txt includes streamlit, pandas, numpy, xgboost, and scikit-learn)
3. Preprocess Data
```
python src\data_pipeline.py
```
4. Train Model
```
python src\train_model.py
```
5. Launch the Dashboard
```
streamlit run app.py
```
The dashboard will automatically open in your default web browser at http://localhost:8501.

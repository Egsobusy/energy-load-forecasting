# Energy Load Forecasting Dashboard

> Short-term electricity load forecasting using ensemble machine learning models

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.52.1-FF4B4B.svg)](https://streamlit.io/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

## Overview

**Energy Load Forecasting** is a short-term electricity load forecasting system using ensemble machine learning models.

Key capabilities:
- Forecast electricity load from 24 hours to 14 days ahead
- Interactive historical data analysis with visualizations
- Ensemble of 4 models: Random Forest, MLP, ARIMA, LSTM
- Uncertainty quantification with confidence intervals
- Business insights for capacity planning and peak demand management

## Features

### Forecasting
- Multiple forecast horizons (24h - 336h)
- Configurable confidence intervals (80%, 90%, 95%, 99%)
- Real-time uncertainty quantification
- Export predictions to CSV

### Data Analysis
- Summary statistics
- Time series visualization
- Distribution analysis
- Seasonal patterns
- Correlation heatmap

### Model Performance
- Ensemble predictions with weighted averaging
- Pattern-based fallback forecaster
- Uncertainty analysis
- Model comparison

### Business Insights
- Capacity planning analysis
- Peak demand identification
- Load duration curves
- Weekly load profiles
- Operational recommendations

## Installation

### Requirements
- Python 3.8 or higher
- pip or conda

### Step 1: Clone repository

```bash
git clone https://github.com/yourusername/energy-load-forecasting.git
cd energy-load-forecasting
```

### Step 2: Create virtual environment (recommended)

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Install dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Prepare data

Place PJM data files in `data/raw/`:
- `pjm_hourly_est.csv` or
- `PJME_hourly.csv`

Download from [Kaggle - Hourly Energy Consumption](https://www.kaggle.com/datasets/robikscube/hourly-energy-consumption)

### Step 5: Process data

```bash
python core/data_processing.py
```

### Step 6: Run dashboard

```bash
streamlit run app.py
```

Dashboard will open at: http://localhost:8501

## Project Structure

```
energy-load-forecasting/
├── app.py                      # Main dashboard application
├── config.py                   # Configuration constants
├── requirements.txt            # Python dependencies
├── core/
│   ├── data_processing.py     # Data processing pipeline
│   └── utils.py               # Utility functions
├── dashboard/
│   ├── data_overview.py       # Data overview tab
│   ├── forecasting.py         # Forecasting tab
│   ├── model_performance.py   # Model performance tab
│   └── business_insights.py   # Business insights tab
├── models/
│   ├── ensemble_predictor.py  # Ensemble prediction system
│   ├── simple_forecaster.py   # Pattern-based fallback
│   ├── arima_model.py        # ARIMA model
│   ├── lstm_model.py         # LSTM model
│   ├── mlp_model.py          # MLP model
│   ├── random_forest_model.py # Random Forest model
│   └── saved_models/         # Trained model files
└── data/
    ├── raw/                   # Raw data files
    └── processed/             # Processed data with features
```

## Feature Engineering

The system automatically creates features:

**Calendar Features:**
- Hour, day, week, month, quarter, year
- Weekend, holiday, business hour indicators

**Lag Features:**
- 1h, 2h, 3h, 6h, 12h, 24h, 48h, 168h

**Rolling Statistics:**
- Mean, std, min, max (multiple windows)

**Cyclical Encoding:**
- Sin/cos for periodic features

**Differences:**
- 1h, 24h, 168h differences

## Data

**Source:** PJM Interconnection
- Hourly energy load data (2002-2018)
- Region: PJM East (PJME)
- 145,000+ hourly records

**Links:**
- [PJM Data Miner 2](https://dataminer2.pjm.com/)
- [Kaggle Dataset](https://www.kaggle.com/datasets/robikscube/hourly-energy-consumption)

## Usage

### 1. Explore Data
- Navigate to "Data Overview" tab
- View summary statistics
- Analyze patterns and distributions

### 2. Generate Forecast
- Go to "Forecasting" tab
- Select forecast horizon (24h - 336h)
- Choose confidence level
- Click "Generate Forecast"
- Download results as CSV

### 3. Evaluate Performance
- Check "Model Performance" tab
- View forecast quality metrics
- Analyze uncertainty

### 4. Business Insights
- Open "Business Insights" tab
- Capacity planning analysis
- Peak demand patterns
- Operational recommendations

## License

[MIT License](LICENSE)

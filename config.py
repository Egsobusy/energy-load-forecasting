"""Configuration constants"""

from pathlib import Path
PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
PROCESSED_DIR = DATA_DIR / "processed"

DATETIME_COL = "Datetime"
LOAD_COL = "PJME"
RAW_DATA_FILE = "pjm_hourly_est.csv"
PROCESSED_DATA_FILE = "pjm_processed.csv"

MIN_DATA_POINTS = 8760
MISSING_THRESHOLD = 0.05
STREAMLIT_PAGE_ICON = "⚡"
STREAMLIT_PAGE_TITLE = "Energy Load Forecasting"
STREAMLIT_LAYOUT = "wide"
STREAMLIT_INITIAL_SIDEBAR_STATE = "expanded"

COLOR_PRIMARY = "#667eea"
COLOR_SECONDARY = "#764ba2"
COLOR_HISTORICAL = "#95a5a6"
COLOR_SUCCESS = "#2ecc71"
COLOR_WARNING = "#f39c12"
COLOR_ERROR = "#e74c3c"
COLOR_NEUTRAL = "#7f8c8d"
COLOR_BACKGROUND = "#f5f7fa"
COLOR_TEXT = "#2c3e50"
COLOR_ACCENT = "#9b59b6"
CHART_HEIGHT_MAIN = 500
CHART_HEIGHT_MEDIUM = 400
CHART_HEIGHT_SMALL = 300

PLOTLY_TEMPLATE = "plotly"
PLOTLY_FONT_FAMILY = "Arial, sans-serif"

DEFAULT_FORECAST_HORIZON = 7
DEFAULT_CONFIDENCE_LEVEL = 0.95
CONFIDENCE_LEVELS = [0.80, 0.90, 0.95, 0.99]
MAPE_EXCELLENT = 2.0
MAPE_GOOD = 5.0
MAPE_ACCEPTABLE = 10.0
ENSEMBLE_MODELS = {
    "statistical": ["Prophet"],
    "ml": ["XGBoost", "LightGBM"]
}

MODEL_WEIGHTS = {
    "XGBoost": 0.40,
    "LightGBM": 0.35,
    "Prophet": 0.25
}

ALL_MODELS = ["Ensemble", "XGBoost", "LightGBM", "Prophet"]
PRICE_PEAK = 80
PRICE_OFFPEAK = 35
PRICE_NORMAL = 50
PEAK_HOUR_START = 6
PEAK_HOUR_END = 22

DEFAULT_CAPACITY_MW = 50000
RECOMMENDED_RESERVE_MARGIN = 0.15
APP_TITLE = "Energy Load Forecasting Dashboard"
APP_SUBTITLE = "Short-term load forecasting with ensemble machine learning models"

TAB_NAMES = {
    "overview": "Data Overview",
    "forecasting": "Forecasting",
    "performance": "Model Performance",
    "insights": "Business Insights"
}

RANDOM_SEED = 42

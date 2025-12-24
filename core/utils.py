"""Utility Functions"""

import pandas as pd
import numpy as np
import streamlit as st
from datetime import datetime, timedelta
import os
import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))
import config


@st.cache_data
def load_processed_data():
    """
    Load processed PJM data with caching.
    
    Returns:
        pd.DataFrame: Processed data with datetime index
    """
    try:
        df = pd.read_csv('data/processed/pjm_processed.csv', index_col=0)
        df.index = pd.to_datetime(df.index)
        return df
    except FileNotFoundError:
        st.error("Processed data not found. Please run data processing first.")
        return None


@st.cache_data
def load_raw_data():
    """
    Load raw PJM data for simple visualizations.
    
    Returns:
        pd.DataFrame: Raw PJME hourly data
    """
    try:
        # Try multiple possible locations
        possible_paths = [
            'data/raw/pjm_hourly_est.csv',
            'data/raw/PJME_hourly.csv',
            'data/PJME_hourly.csv'
        ]
        
        for path in possible_paths:
            if os.path.exists(path):
                df = pd.read_csv(path)
                df['Datetime'] = pd.to_datetime(df['Datetime'])
                df = df.set_index('Datetime').sort_index()
                return df
        
        return None
    except Exception as e:
        st.error(f"Error loading raw data: {str(e)}")
        return None


def filter_data_by_date(df, start_date, end_date):
    """
    Filter dataframe by date range.
    
    Args:
        df: DataFrame with datetime index
        start_date: Start date
        end_date: End date
        
    Returns:
        Filtered DataFrame
    """
    mask = (df.index >= pd.to_datetime(start_date)) & (df.index <= pd.to_datetime(end_date))
    return df[mask]


def calculate_metrics(y_true, y_pred):
    """
    Calculate forecasting accuracy metrics.
    
    Args:
        y_true: Actual values
        y_pred: Predicted values
        
    Returns:
        dict: Dictionary of metrics
    """
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    r2 = r2_score(y_true, y_pred)
    
    return {
        'MAE': mae,
        'RMSE': rmse,
        'MAPE': mape,
        'R²': r2
    }


def get_summary_stats(df, target_col=None):
    """
    Get summary statistics for the dataset.
    
    Args:
        df: DataFrame
        target_col: Name of target column
        
    Returns:
        dict: Summary statistics
    """
    if target_col is None:
        target_col = config.LOAD_COL
    return {
        'Total Records': len(df),
        'Date Range': f"{df.index.min().strftime('%Y-%m-%d')} to {df.index.max().strftime('%Y-%m-%d')}",
        'Total Consumption (MWh)': df[target_col].sum(),
        'Average Load (MW)': df[target_col].mean(),
        'Peak Load (MW)': df[target_col].max(),
        'Min Load (MW)': df[target_col].min(),
        'Std Dev (MW)': df[target_col].std(),
        'Peak Date': df[target_col].idxmax().strftime('%Y-%m-%d %H:%M'),
        'Min Date': df[target_col].idxmin().strftime('%Y-%m-%d %H:%M')
    }


def get_hourly_pattern(df, target_col=None):
    """
    Get average load by hour of day.
    
    Args:
        df: DataFrame with datetime index
        target_col: Target column name
        
    Returns:
        pd.Series: Average load by hour
    """
    if target_col is None:
        target_col = config.LOAD_COL
    return df.groupby(df.index.hour)[target_col].mean()


def get_daily_pattern(df, target_col=None):
    """
    Get average load by day of week.
    
    Args:
        df: DataFrame with datetime index
        target_col: Target column name
        
    Returns:
        pd.Series: Average load by day of week
    """
    if target_col is None:
        target_col = config.LOAD_COL
    day_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    daily = df.groupby(df.index.dayofweek)[target_col].mean()
    daily.index = day_names
    return daily


def get_monthly_pattern(df, target_col=None):
    """
    Get average load by month.
    
    Args:
        df: DataFrame with datetime index
        target_col: Target column name
        
    Returns:
        pd.Series: Average load by month
    """
    if target_col is None:
        target_col = config.LOAD_COL
    month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                   'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    monthly = df.groupby(df.index.month)[target_col].mean()
    monthly.index = month_names
    return monthly


def compare_weekday_weekend(df, target_col=None):
    """
    Compare weekday vs weekend consumption.
    
    Args:
        df: DataFrame with datetime index
        target_col: Target column name
        
    Returns:
        dict: Comparison statistics
    """
    if target_col is None:
        target_col = config.LOAD_COL
    weekday = df[df.index.dayofweek < 5][target_col].mean()
    weekend = df[df.index.dayofweek >= 5][target_col].mean()
    
    return {
        'Weekday Average': weekday,
        'Weekend Average': weekend,
        'Difference': weekday - weekend,
        'Difference (%)': ((weekday - weekend) / weekday) * 100
    }


def analyze_peak_demand(df, target_col=None, top_n=10):
    """
    Analyze peak demand periods.
    
    Args:
        df: DataFrame
        target_col: Target column
        top_n: Number of top peaks to return
        
    Returns:
        dict: Peak demand analysis
    """
    if target_col is None:
        target_col = config.LOAD_COL
    # Get top peaks
    top_peaks = df.nlargest(top_n, target_col)[[target_col]].copy()
    top_peaks['Hour'] = top_peaks.index.hour
    top_peaks['Day'] = top_peaks.index.day_name()
    top_peaks['Month'] = top_peaks.index.month_name()
    
    # Peak statistics
    percentile_95 = df[target_col].quantile(0.95)
    percentile_99 = df[target_col].quantile(0.99)
    
    high_load_hours = (df[target_col] > percentile_95).sum()
    extreme_load_hours = (df[target_col] > percentile_99).sum()
    
    # Most common peak hours
    peak_hours = df[df[target_col] > percentile_95].index.hour.value_counts().head(5)
    
    return {
        'top_peaks': top_peaks,
        'percentile_95': percentile_95,
        'percentile_99': percentile_99,
        'high_load_hours': high_load_hours,
        'extreme_load_hours': extreme_load_hours,
        'common_peak_hours': peak_hours,
        'peak_percentage': (high_load_hours / len(df)) * 100
    }


def calculate_cost_analysis(df, target_col=None):
    """
    Calculate cost analysis with different pricing for peak/off-peak.
    
    Args:
        df: DataFrame
        target_col: Target column
        
    Returns:
        dict: Cost analysis
    """
    if target_col is None:
        target_col = config.LOAD_COL
    # Pricing structure ($/MWh)
    peak_price = 80      # 6 AM - 10 PM
    offpeak_price = 35   # 10 PM - 6 AM
    
    # Classify hours
    df_cost = df.copy()
    df_cost['is_peak'] = ((df_cost.index.hour >= 6) & (df_cost.index.hour < 22)).astype(int)
    
    # Calculate costs
    peak_consumption = df_cost[df_cost['is_peak'] == 1][target_col].sum()
    offpeak_consumption = df_cost[df_cost['is_peak'] == 0][target_col].sum()
    
    peak_cost = peak_consumption * peak_price
    offpeak_cost = offpeak_consumption * offpeak_price
    total_cost = peak_cost + offpeak_cost
    
    # Hours breakdown
    peak_hours = (df_cost['is_peak'] == 1).sum()
    offpeak_hours = (df_cost['is_peak'] == 0).sum()
    
    return {
        'total_cost': total_cost,
        'peak_cost': peak_cost,
        'offpeak_cost': offpeak_cost,
        'peak_consumption': peak_consumption,
        'offpeak_consumption': offpeak_consumption,
        'peak_hours': peak_hours,
        'offpeak_hours': offpeak_hours,
        'avg_cost_per_mwh': total_cost / (peak_consumption + offpeak_consumption),
        'peak_cost_percentage': (peak_cost / total_cost) * 100
    }


def capacity_analysis(df, target_col=None, assumed_capacity=50000):
    """
    Analyze capacity utilization and adequacy.
    
    Args:
        df: DataFrame
        target_col: Target column
        assumed_capacity: Assumed system capacity in MW
        
    Returns:
        dict: Capacity analysis
    """
    if target_col is None:
        target_col = config.LOAD_COL
    max_load = df[target_col].max()
    avg_load = df[target_col].mean()
    
    # Utilization
    avg_utilization = (avg_load / assumed_capacity) * 100
    peak_utilization = (max_load / assumed_capacity) * 100
    
    # Reserve margin
    reserve_margin = ((assumed_capacity - max_load) / max_load) * 100
    
    # Hours near capacity (>90%)
    near_capacity_threshold = assumed_capacity * 0.9
    hours_near_capacity = (df[target_col] > near_capacity_threshold).sum()
    
    # Recommended capacity (with 15% safety margin)
    recommended_capacity = max_load * 1.15
    
    return {
        'current_capacity': assumed_capacity,
        'max_load': max_load,
        'avg_load': avg_load,
        'avg_utilization': avg_utilization,
        'peak_utilization': peak_utilization,
        'reserve_margin': reserve_margin,
        'hours_near_capacity': hours_near_capacity,
        'recommended_capacity': recommended_capacity,
        'capacity_gap': max(0, recommended_capacity - assumed_capacity)
    }


def generate_recommendations(df, target_col=None):
    """
    Generate actionable business recommendations.
    
    Args:
        df: DataFrame
        target_col: Target column
        
    Returns:
        dict: Recommendations by priority
    """
    if target_col is None:
        target_col = config.LOAD_COL
    
    # Get patterns
    hourly = get_hourly_pattern(df, target_col)
    peak_analysis = analyze_peak_demand(df, target_col)
    
    # Find best maintenance window (lowest average load for 3 consecutive hours)
    rolling_3h = hourly.rolling(3).mean()
    best_maintenance_start = rolling_3h.idxmin()
    
    recommendations = {
        'urgent': [],
        'short_term': [],
        'long_term': []
    }
    
    # Urgent recommendations
    if peak_analysis['percentile_99'] > 50000:
        recommendations['urgent'].append(
            f"Peak loads exceeding 50,000 MW detected. Ensure all generation units operational."
        )
    
    recommendations['urgent'].append(
        f"Monitor hours {peak_analysis['common_peak_hours'].index[0]}-{peak_analysis['common_peak_hours'].index[0]+2} closely - highest peak frequency."
    )
    
    # Short-term
    recommendations['short_term'].append(
        f"Optimal maintenance window: {int(best_maintenance_start)}-{int(best_maintenance_start)+3} hours (avg load: {rolling_3h.min():.0f} MW)"
    )
    
    recommendations['short_term'].append(
        "Implement demand response programs during peak hours to reduce load by 2-5%."
    )
    
    # Long-term
    recommendations['long_term'].append(
        f"Plan capacity expansion: Current peak {peak_analysis['top_peaks'][target_col].max():.0f} MW, recommended capacity {peak_analysis['top_peaks'][target_col].max() * 1.15:.0f} MW."
    )
    
    recommendations['long_term'].append(
        "Invest in energy storage systems (500-1000 MWh) for peak shaving."
    )
    
    recommendations['long_term'].append(
        "Develop renewable integration strategy to reduce costs and emissions."
    )
    
    return recommendations


def create_forecast_dates(start_date, horizon_days):
    """
    Create future dates for forecasting.
    
    Args:
        start_date: Starting date
        horizon_days: Number of days to forecast
        
    Returns:
        pd.DatetimeIndex: Future dates
    """
    start = pd.to_datetime(start_date)
    hours = horizon_days * 24
    return pd.date_range(start=start, periods=hours, freq='h')


def generate_mock_forecast(historical_data, forecast_dates, model_name='Ensemble'):
    """
    Generate mock forecast for demonstration purposes.
    This will be replaced with actual model predictions.
    
    Args:
        historical_data: Historical DataFrame
        forecast_dates: Future dates to forecast
        model_name: Name of model (affects variance)
        
    Returns:
        dict: Forecast results with predictions and confidence intervals
    """
    # Get recent patterns
    load_col = config.LOAD_COL
    recent_mean = historical_data[load_col].tail(168).mean()  # Last week
    recent_std = historical_data[load_col].tail(168).std()
    
    # Model-specific error levels (for demo)
    error_multipliers = {
        'XGBoost': 1.0,
        'LightGBM': 1.1,
        'Random Forest': 1.3,
        'Prophet': 1.5,
        'Ensemble': 0.9
    }
    
    error_mult = error_multipliers.get(model_name, 1.0)
    
    # Generate predictions with patterns
    predictions = []
    for date in forecast_dates:
        hour = date.hour
        day_of_week = date.dayofweek
        
        # Hourly pattern
        if 0 <= hour < 6:
            hour_factor = 0.75
        elif 6 <= hour < 12:
            hour_factor = 0.95
        elif 12 <= hour < 18:
            hour_factor = 1.05
        else:
            hour_factor = 1.15
        
        # Day of week pattern
        dow_factor = 0.88 if day_of_week >= 5 else 1.0
        
        # Generate prediction
        base = recent_mean * hour_factor * dow_factor
        noise = np.random.normal(0, recent_std * 0.1)
        pred = base + noise
        
        predictions.append(pred)
    
    predictions = np.array(predictions)
    
    # Confidence intervals (95%)
    ci_width = recent_std * 1.96 * error_mult
    lower = predictions - ci_width
    upper = predictions + ci_width
    
    return {
        'dates': forecast_dates,
        'predictions': predictions,
        'lower_bound': lower,
        'upper_bound': upper,
        'model_name': model_name
    }


def calculate_scenario_impact(base_forecast, scenario_type='base'):
    """
    Calculate different scenario impacts on forecast.
    
    Args:
        base_forecast: Base forecast values
        scenario_type: 'base', 'high_growth', 'efficiency', 'heat_wave'
        
    Returns:
        np.array: Adjusted forecast
    """
    scenarios = {
        'base': 1.0,
        'high_growth': 1.10,      # +10% growth
        'efficiency': 0.90,        # -10% through efficiency
        'heat_wave': 1.20,         # +20% during extreme weather
        'recession': 0.85          # -15% economic downturn
    }
    
    multiplier = scenarios.get(scenario_type, 1.0)
    return base_forecast * multiplier


# Format helpers
def format_large_number(num):
    """Format large numbers with commas."""
    return f"{num:,.0f}"


def format_currency(num):
    """Format as currency."""
    if num >= 1e9:
        return f"${num/1e9:.2f}B"
    elif num >= 1e6:
        return f"${num/1e6:.2f}M"
    elif num >= 1e3:
        return f"${num/1e3:.2f}K"
    else:
        return f"${num:.2f}"


def format_percentage(num):
    """Format as percentage."""
    return f"{num:.1f}%"


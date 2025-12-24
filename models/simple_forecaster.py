"""Simple Pattern-Based Forecaster"""

import numpy as np
import pandas as pd


class SimpleForecaster:
    """Pattern-based forecasting using historical patterns"""
    
    def __init__(self):
        self.hourly_pattern = None
        self.weekly_pattern = None
        self.recent_mean = None
        self.recent_std = None
        
    def fit(self, data, load_col='PJME'):
        """Learn patterns from historical data"""
        # Hourly pattern (average by hour)
        data_copy = data.copy()
        data_copy['hour'] = data_copy.index.hour
        self.hourly_pattern = data_copy.groupby('hour')[load_col].mean()
        
        # Weekly pattern (average by day of week)
        data_copy['dayofweek'] = data_copy.index.dayofweek
        self.weekly_pattern = data_copy.groupby('dayofweek')[load_col].mean()
        
        # Recent statistics (last 30 days)
        recent = data.tail(30*24)
        self.recent_mean = recent[load_col].mean()
        self.recent_std = recent[load_col].std()
        
    def predict(self, future_dates):
        """Generate predictions for future dates"""
        predictions = []
        
        for date in future_dates:
            hour = date.hour
            dow = date.dayofweek
            
            # Combine hourly and weekly patterns
            hour_factor = self.hourly_pattern[hour] / self.recent_mean
            dow_factor = self.weekly_pattern[dow] / self.recent_mean
            
            # Average the factors
            combined_factor = (hour_factor + dow_factor) / 2
            
            # Base prediction
            pred = self.recent_mean * combined_factor
            
            # Add small random variation
            noise = np.random.normal(0, self.recent_std * 0.05)
            pred += noise
            
            predictions.append(pred)
        
        return np.array(predictions)
    
    def predict_with_interval(self, future_dates, confidence=0.95):
        """Generate predictions with confidence intervals"""
        predictions = self.predict(future_dates)
        
        # Uncertainty based on recent variability
        from scipy import stats
        z_score = stats.norm.ppf((1 + confidence) / 2)
        margin = z_score * self.recent_std
        
        lower = predictions - margin
        upper = predictions + margin
        uncertainty = np.full(len(predictions), margin)
        
        return predictions, lower, upper, uncertainty


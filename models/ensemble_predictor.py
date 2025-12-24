"""Ensemble Model Predictor"""

import pandas as pd
import numpy as np
import pickle
import joblib
from pathlib import Path
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

try:
    from tensorflow.keras.models import load_model
    KERAS_AVAILABLE = True
except:
    KERAS_AVAILABLE = False

try:
    import torch
    TORCH_AVAILABLE = True
except:
    TORCH_AVAILABLE = False


class EnsemblePredictor:
    """
    Ensemble predictor that combines multiple models for robust forecasting
    """
    
    def __init__(self, models_dir='models/saved_models'):
        self.models_dir = Path(models_dir)
        self.models = {}
        self.model_weights = {
            'random_forest': 0.35,
            'mlp': 0.30,
            'arima': 0.20,
            'lstm': 0.15
        }
        
    def load_models(self):
        """Load all available trained models with error handling"""
        loaded = []
        
        # Random Forest
        try:
            rf_path = self.models_dir / 'random_forest_model.pkl'
            if rf_path.exists():
                self.models['random_forest'] = joblib.load(rf_path)
                loaded.append('Random Forest')
        except Exception as e:
            print(f"Could not load Random Forest: {e}")
        
        # MLP
        try:
            mlp_path = self.models_dir / 'mlp_model.pkl'
            if mlp_path.exists():
                self.models['mlp'] = joblib.load(mlp_path)
                loaded.append('MLP')
        except Exception as e:
            print(f"Could not load MLP: {e}")
        
        # ARIMA
        try:
            arima_path = self.models_dir / 'arima_model.pkl'
            if arima_path.exists():
                with open(arima_path, 'rb') as f:
                    self.models['arima'] = pickle.load(f)
                loaded.append('ARIMA')
        except Exception as e:
            print(f"Could not load ARIMA: {e}")
        
        # LSTM
        if KERAS_AVAILABLE:
            try:
                lstm_path = self.models_dir / 'lstm_model.h5'
                if lstm_path.exists():
                    self.models['lstm'] = load_model(lstm_path)
                    loaded.append('LSTM')
            except Exception as e:
                print(f"Could not load LSTM: {e}")
        
        return loaded
    
    def predict_single(self, model_name, X):
        """Get prediction from a single model"""
        if model_name not in self.models:
            return None
        
        model = self.models[model_name]
        
        if model_name == 'arima':
            pred = model.forecast(steps=len(X))
            return np.array(pred)
        else:
            pred = model.predict(X)
            return pred.flatten() if len(pred.shape) > 1 else pred
    
    def predict_ensemble(self, X):
        """Generate ensemble prediction with uncertainty estimates"""
        predictions = {}
        
        for model_name in self.models.keys():
            pred = self.predict_single(model_name, X)
            if pred is not None:
                predictions[model_name] = pred
        
        if not predictions:
            return None, None, None
        
        # Weighted average
        weighted_pred = np.zeros_like(list(predictions.values())[0])
        total_weight = 0
        
        for model_name, pred in predictions.items():
            weight = self.model_weights.get(model_name, 0)
            weighted_pred += weight * pred
            total_weight += weight
        
        if total_weight > 0:
            weighted_pred /= total_weight
        
        # Calculate uncertainty (std across models)
        pred_array = np.array(list(predictions.values()))
        uncertainty = np.std(pred_array, axis=0)
        
        return weighted_pred, uncertainty, predictions
    
    def calculate_confidence_interval(self, predictions, uncertainty, confidence=0.95):
        """Calculate prediction intervals based on uncertainty"""
        from scipy import stats
        
        z_score = stats.norm.ppf((1 + confidence) / 2)
        margin = z_score * uncertainty
        
        lower = predictions - margin
        upper = predictions + margin
        
        return lower, upper
    
    def evaluate_models(self, X_test, y_test):
        """Evaluate all models on test set"""
        results = {}
        
        for model_name in self.models.keys():
            pred = self.predict_single(model_name, X_test)
            
            if pred is not None and len(pred) == len(y_test):
                mae = mean_absolute_error(y_test, pred)
                rmse = np.sqrt(mean_squared_error(y_test, pred))
                r2 = r2_score(y_test, pred)
                mape = np.mean(np.abs((y_test - pred) / y_test)) * 100
                
                results[model_name] = {
                    'MAE': mae,
                    'RMSE': rmse,
                    'R2': r2,
                    'MAPE': mape
                }
        
        # Ensemble performance
        ensemble_pred, uncertainty, _ = self.predict_ensemble(X_test)
        
        if ensemble_pred is not None:
            mae = mean_absolute_error(y_test, ensemble_pred)
            rmse = np.sqrt(mean_squared_error(y_test, ensemble_pred))
            r2 = r2_score(y_test, ensemble_pred)
            mape = np.mean(np.abs((y_test - ensemble_pred) / y_test)) * 100
            
            results['ensemble'] = {
                'MAE': mae,
                'RMSE': rmse,
                'R2': r2,
                'MAPE': mape
            }
        
        return results


def generate_forecast(historical_data, feature_columns, horizon_hours=168, confidence=0.95):
    """
    Generate forecast for specified horizon
    
    Args:
        historical_data: DataFrame with features
        feature_columns: List of feature column names
        horizon_hours: Number of hours to forecast
        confidence: Confidence level for prediction intervals
    
    Returns:
        Dictionary with forecasts and metadata
    """
    predictor = EnsemblePredictor()
    loaded_models = predictor.load_models()
    
    if not loaded_models:
        return None
    
    # Use last available data point features for forecasting
    X_forecast = historical_data[feature_columns].tail(horizon_hours)
    
    # Generate predictions
    predictions, uncertainty, individual_preds = predictor.predict_ensemble(X_forecast)
    
    if predictions is None:
        return None
    
    # Calculate confidence intervals
    lower, upper = predictor.calculate_confidence_interval(predictions, uncertainty, confidence)
    
    # Create forecast dates
    last_date = historical_data.index[-1]
    forecast_dates = pd.date_range(start=last_date + pd.Timedelta(hours=1), periods=horizon_hours, freq='H')
    
    return {
        'dates': forecast_dates,
        'predictions': predictions,
        'lower_bound': lower,
        'upper_bound': upper,
        'uncertainty': uncertainty,
        'individual_predictions': individual_preds,
        'loaded_models': loaded_models,
        'confidence': confidence
    }


if __name__ == '__main__':
    # Test ensemble predictor
    predictor = EnsemblePredictor()
    loaded = predictor.load_models()
    print(f"Loaded models: {loaded}")


"""Forecasting Tab"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

import config
from models.ensemble_predictor import EnsemblePredictor
from models.simple_forecaster import SimpleForecaster


def render_forecasting_tab(data: pd.DataFrame):
    """
    Render forecasting interface with ensemble predictions
    
    Args:
        data: Historical load data with features
    """
    st.markdown("## Forecasting")
    
    render_current_status(data)
    
    st.markdown("---")
    
    forecast_config = render_forecast_controls()
    
    if st.button("Generate Forecast", type="primary", width='stretch'):
        generate_and_display_forecast(data, forecast_config)
    
    if 'forecast_result' in st.session_state:
        st.markdown("---")
        render_forecast_visualization(data, st.session_state['forecast_result'])
        
        st.markdown("---")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            render_forecast_table(st.session_state['forecast_result'])
        
        with col2:
            render_forecast_statistics(st.session_state['forecast_result'])


def render_current_status(data: pd.DataFrame):
    """Display current load status"""
    st.markdown("### Current Status")
    
    load_col = config.LOAD_COL
    
    # Get recent data
    latest = data.iloc[-1]
    last_24h = data.tail(24)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        current_load = latest[load_col]
        st.metric("Current Load", f"{current_load:.0f} MW")
    
    with col2:
        avg_24h = last_24h[load_col].mean()
        st.metric("24h Average", f"{avg_24h:.0f} MW")
    
    with col3:
        peak_24h = last_24h[load_col].max()
        st.metric("24h Peak", f"{peak_24h:.0f} MW")
    
    with col4:
        last_time = latest.name
        st.metric("Last Update", last_time.strftime('%Y-%m-%d %H:%M'))


def render_forecast_controls():
    """Render forecast configuration controls"""
    st.markdown("### Forecast Configuration")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        horizon = st.selectbox(
            "Forecast Horizon",
            options=[24, 48, 72, 168, 336],
            format_func=lambda x: f"{x} hours ({x//24} days)",
            index=3
        )
    
    with col2:
        confidence = st.select_slider(
            "Confidence Level",
            options=[0.80, 0.90, 0.95, 0.99],
            value=0.95,
            format_func=lambda x: f"{x*100:.0f}%"
        )
    
    with col3:
        model_type = st.selectbox(
            "Model Type",
            options=["Ensemble", "Individual Models"],
            index=0
        )
    
    return {
        'horizon': horizon,
        'confidence': confidence,
        'model_type': model_type
    }


def generate_and_display_forecast(data: pd.DataFrame, config_dict: dict):
    """Generate forecast using available method"""
    
    with st.spinner("Generating forecast..."):
        try:
            # Try ensemble predictor first
            predictor = EnsemblePredictor()
            loaded_models = predictor.load_models()
            
            if loaded_models:
                # Use ensemble predictor
                st.info(f"Using ensemble: {', '.join(loaded_models)}")
                forecast_result = generate_ensemble_forecast(
                    data, predictor, config_dict, loaded_models
                )
            else:
                # Fallback to simple pattern-based forecaster
                st.info("Using pattern-based forecasting (models not available)")
                forecast_result = generate_simple_forecast(data, config_dict)
            
            if forecast_result:
                st.session_state['forecast_result'] = forecast_result
                st.success("Forecast complete")
                st.rerun()
            
        except Exception as e:
            st.error(f"Error: {str(e)}")


def generate_ensemble_forecast(data, predictor, config_dict, loaded_models):
    """Generate forecast using ensemble models"""
    feature_cols = [col for col in data.columns if col != config.LOAD_COL]
    X_forecast = data[feature_cols].tail(config_dict['horizon'])
    
    predictions, uncertainty, individual_preds = predictor.predict_ensemble(X_forecast)
    
    if predictions is None:
        return None
    
    lower, upper = predictor.calculate_confidence_interval(
        predictions, uncertainty, config_dict['confidence']
    )
    
    last_date = data.index[-1]
    forecast_dates = pd.date_range(
        start=last_date + pd.Timedelta(hours=1),
        periods=config_dict['horizon'],
        freq='H'
    )
    
    return {
        'dates': forecast_dates,
        'predictions': predictions,
        'lower_bound': lower,
        'upper_bound': upper,
        'uncertainty': uncertainty,
        'method': 'ensemble',
        'models': loaded_models,
        'confidence': config_dict['confidence']
    }


def generate_simple_forecast(data, config_dict):
    """Generate forecast using simple pattern-based method"""
    forecaster = SimpleForecaster()
    forecaster.fit(data, config.LOAD_COL)
    
    last_date = data.index[-1]
    forecast_dates = pd.date_range(
        start=last_date + pd.Timedelta(hours=1),
        periods=config_dict['horizon'],
        freq='H'
    )
    
    predictions, lower, upper, uncertainty = forecaster.predict_with_interval(
        forecast_dates, config_dict['confidence']
    )
    
    return {
        'dates': forecast_dates,
        'predictions': predictions,
        'lower_bound': lower,
        'upper_bound': upper,
        'uncertainty': uncertainty,
        'method': 'pattern-based',
        'models': ['Historical Patterns'],
        'confidence': config_dict['confidence']
    }


def render_forecast_visualization(historical_data: pd.DataFrame, forecast_result: dict):
    """Render interactive forecast chart"""
    st.markdown("### Forecast Visualization")
    
    # Get recent historical context
    context_hours = min(168, len(historical_data))
    historical_context = historical_data.tail(context_hours)
    
    fig = go.Figure()
    
    # Historical data
    fig.add_trace(go.Scatter(
        x=historical_context.index,
        y=historical_context[config.LOAD_COL],
        mode='lines',
        name='Historical',
        line=dict(color=config.COLOR_HISTORICAL, width=1.5),
        hovertemplate='%{x}<br>Load: %{y:.0f} MW<extra></extra>'
    ))
    
    # Forecast
    fig.add_trace(go.Scatter(
        x=forecast_result['dates'],
        y=forecast_result['predictions'],
        mode='lines',
        name='Forecast',
        line=dict(color=config.COLOR_PRIMARY, width=2.5),
        hovertemplate='%{x}<br>Forecast: %{y:.0f} MW<extra></extra>'
    ))
    
    # Confidence interval
    fig.add_trace(go.Scatter(
        x=forecast_result['dates'],
        y=forecast_result['upper_bound'],
        mode='lines',
        line=dict(width=0),
        showlegend=False,
        hoverinfo='skip'
    ))
    
    fig.add_trace(go.Scatter(
        x=forecast_result['dates'],
        y=forecast_result['lower_bound'],
        mode='lines',
        line=dict(width=0),
        fill='tonexty',
        fillcolor='rgba(31, 119, 180, 0.2)',
        name=f'{forecast_result["confidence"]*100:.0f}% Confidence',
        hoverinfo='skip'
    ))
    
    fig.update_layout(
        xaxis_title="Date & Time",
        yaxis_title="Load (MW)",
        height=500,
        template=config.PLOTLY_TEMPLATE,
        hovermode='x unified',
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1)
    )
    
    st.plotly_chart(fig, width='stretch')


def render_forecast_table(forecast_result: dict):
    """Render forecast data table"""
    st.markdown("### Forecast Data")
    
    # Create DataFrame
    forecast_df = pd.DataFrame({
        'Timestamp': forecast_result['dates'].strftime('%Y-%m-%d %H:%M'),
        'Forecast': forecast_result['predictions'].astype(int),
        'Lower Bound': forecast_result['lower_bound'].astype(int),
        'Upper Bound': forecast_result['upper_bound'].astype(int),
        'Uncertainty': forecast_result['uncertainty'].astype(int)
    })
    
    # Display table
    st.dataframe(
        forecast_df.head(48),
        width='stretch',
        height=400,
        hide_index=True
    )
    
    # Download button
    csv = forecast_df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="Download Full Forecast (CSV)",
        data=csv,
        file_name=f"forecast_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
        mime="text/csv",
        width='stretch'
    )


def render_forecast_statistics(forecast_result: dict):
    """Render forecast summary statistics"""
    st.markdown("### Forecast Summary")
    
    predictions = forecast_result['predictions']
    
    st.metric("Average Load", f"{predictions.mean():.0f} MW")
    st.metric("Peak Load", f"{predictions.max():.0f} MW")
    st.metric("Min Load", f"{predictions.min():.0f} MW")
    
    # Find peak time
    peak_idx = predictions.argmax()
    peak_time = forecast_result['dates'][peak_idx]
    st.metric("Peak Time", peak_time.strftime('%Y-%m-%d %H:%M'))
    
    # Uncertainty statistics
    uncertainty = forecast_result['uncertainty']
    st.metric("Avg Uncertainty", f"±{uncertainty.mean():.0f} MW")
    
    # Total energy
    total_energy = predictions.sum()
    st.metric("Total Energy", f"{total_energy:,.0f} MWh")
    
    # Alert if high load
    if predictions.max() > 45000:
        st.warning(f"High load alert: Peak forecast exceeds 45,000 MW")


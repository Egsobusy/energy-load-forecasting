"""Model Performance Tab"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))
import config


def render_model_performance_tab(data: pd.DataFrame):
    """Render model performance analysis"""
    st.markdown("## Model Performance")
    
    if 'forecast_result' not in st.session_state:
        st.info("Generate a forecast to see performance analysis")
        render_model_info()
        return
    
    forecast_result = st.session_state['forecast_result']
    
    render_forecast_quality_metrics(forecast_result)
    st.markdown("---")
    render_prediction_analysis(forecast_result)
    st.markdown("---")
    render_model_info()


def render_forecast_quality_metrics(forecast_result: dict):
    """Display forecast quality metrics"""
    st.markdown("### Forecast Quality")
    
    method = forecast_result.get('method', 'unknown')
    models = forecast_result.get('models', [])
    confidence = forecast_result.get('confidence', 0.95)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Method", method.title())
    
    with col2:
        st.metric("Models", len(models))
    
    with col3:
        st.metric("Confidence", f"{confidence*100:.0f}%")
    
    with col4:
        horizon_hours = len(forecast_result['predictions'])
        st.metric("Horizon", f"{horizon_hours}h")
    
    if models:
        st.info(f"**Active Models:** {', '.join(models)}")






def render_prediction_analysis(forecast_result: dict):
    """Analyze prediction uncertainty"""
    st.markdown("### Uncertainty Analysis")
    
    col1, col2, col3 = st.columns(3)
    
    predictions = forecast_result['predictions']
    lower = forecast_result['lower_bound']
    upper = forecast_result['upper_bound']
    
    with col1:
        avg_interval = np.mean(upper - lower)
        st.metric("Avg Interval Width", f"{avg_interval:.0f} MW")
    
    with col2:
        max_interval = np.max(upper - lower)
        st.metric("Max Interval Width", f"{max_interval:.0f} MW")
    
    with col3:
        interval_pct = (avg_interval / np.mean(predictions)) * 100
        st.metric("Interval Width %", f"{interval_pct:.1f}%")
    
    # Uncertainty over time
    fig = go.Figure()
    
    dates = forecast_result['dates']
    uncertainty = forecast_result.get('uncertainty', upper - lower)
    
    fig.add_trace(go.Scatter(
        x=dates,
        y=uncertainty,
        mode='lines',
        name='Uncertainty',
        line=dict(color=config.COLOR_WARNING, width=2),
        fill='tozeroy',
        fillcolor=f'rgba(255, 127, 14, 0.2)'
    ))
    
    fig.update_layout(
        xaxis_title="Date",
        yaxis_title="Uncertainty (MW)",
        height=300,
        template=config.PLOTLY_TEMPLATE,
        showlegend=False
    )
    
    st.plotly_chart(fig, width='stretch')


def render_model_info():
    """Display model information"""
    st.markdown("### Model Information")
    
    with st.expander("Random Forest"):
        st.markdown("""
        **Type:** Ensemble Tree-based Model
        
        **Configuration:**
        - Multiple decision trees
        - Bootstrap aggregating (bagging)
        - Feature importance analysis
        
        **Strengths:** Robust to outliers, captures non-linear patterns, handles many features
        """)
    
    with st.expander("MLP (Multi-Layer Perceptron)"):
        st.markdown("""
        **Type:** Neural Network
        
        **Configuration:**
        - Hidden layers: (100, 50)
        - Activation: ReLU
        - Optimizer: Adam
        
        **Strengths:** Learns complex non-linear relationships, adaptable
        """)
    
    with st.expander("ARIMA"):
        st.markdown("""
        **Type:** Statistical Time Series Model
        
        **Configuration:**
        - AutoRegressive (AR): Past values
        - Integrated (I): Differencing
        - Moving Average (MA): Past errors
        
        **Strengths:** Captures temporal dependencies, interpretable
        """)
    
    with st.expander("LSTM (Long Short-Term Memory)"):
        st.markdown("""
        **Type:** Recurrent Neural Network
        
        **Configuration:**
        - LSTM units: 50
        - Timesteps: 10
        - Optimizer: Adam
        
        **Strengths:** Captures long-term dependencies, handles sequences
        """)
    
    with st.expander("Ensemble Model"):
        st.markdown("""
        **Type:** Weighted Combination
        
        **Weights:**
        - Random Forest: 35%
        - MLP: 30%
        - ARIMA: 20%
        - LSTM: 15%
        
        **Strengths:** Combines strengths of all models, more robust predictions
        """)


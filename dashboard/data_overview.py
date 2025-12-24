"""Data Overview Tab"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))
import config


def render_data_overview_tab(data: pd.DataFrame):
    """
    Render data overview with statistical analysis
    
    Args:
        data: Historical load data with datetime index
    """
    st.markdown("## Data Overview")
    
    render_summary_statistics(data)
    
    st.markdown("---")
    
    render_time_series_visualization(data)
    
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    
    with col1:
        render_distribution_analysis(data)
    
    with col2:
        render_seasonal_patterns(data)
    
    st.markdown("---")
    
    render_correlation_heatmap(data)


def render_summary_statistics(data: pd.DataFrame):
    """Display summary statistics metrics"""
    st.markdown("### Summary Statistics")
    
    load_col = config.LOAD_COL
    load_data = data[load_col]
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric("Total Records", f"{len(data):,}")
    
    with col2:
        st.metric("Mean Load", f"{load_data.mean():.0f} MW")
    
    with col3:
        st.metric("Peak Load", f"{load_data.max():.0f} MW")
    
    with col4:
        st.metric("Min Load", f"{load_data.min():.0f} MW")
    
    with col5:
        std_dev = load_data.std()
        cv = (std_dev / load_data.mean()) * 100
        st.metric("Std Dev", f"{std_dev:.0f} MW", f"CV: {cv:.1f}%")
    
    col6, col7, col8, col9 = st.columns(4)
    
    with col6:
        date_range = (data.index.max() - data.index.min()).days
        st.metric("Date Range", f"{date_range} days")
    
    with col7:
        st.metric("Start Date", data.index.min().strftime('%Y-%m-%d'))
    
    with col8:
        st.metric("End Date", data.index.max().strftime('%Y-%m-%d'))
    
    with col9:
        missing_pct = (load_data.isna().sum() / len(load_data)) * 100
        st.metric("Missing Data", f"{missing_pct:.2f}%")


def render_time_series_visualization(data: pd.DataFrame):
    """Render time series plot"""
    st.markdown("### Time Series Visualization")
    
    time_range = st.radio(
        "Select time range:",
        options=['Last 7 Days', 'Last 30 Days', 'Last 90 Days', 'Last Year', 'All Data'],
        horizontal=True,
        index=1
    )
    
    # Filter data based on selection
    if time_range == 'Last 7 Days':
        plot_data = data.tail(24*7)
    elif time_range == 'Last 30 Days':
        plot_data = data.tail(24*30)
    elif time_range == 'Last 90 Days':
        plot_data = data.tail(24*90)
    elif time_range == 'Last Year':
        plot_data = data.tail(24*365)
    else:
        plot_data = data
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=plot_data.index,
        y=plot_data[config.LOAD_COL],
        mode='lines',
        name='Load',
        line=dict(color=config.COLOR_PRIMARY, width=1),
        hovertemplate='%{x}<br>Load: %{y:.0f} MW<extra></extra>'
    ))
    
    # Add rolling average
    if len(plot_data) > 24:
        rolling_avg = plot_data[config.LOAD_COL].rolling(window=24).mean()
        fig.add_trace(go.Scatter(
            x=plot_data.index,
            y=rolling_avg,
            mode='lines',
            name='24h Moving Avg',
            line=dict(color=config.COLOR_SUCCESS, width=2),
            hovertemplate='%{x}<br>Avg: %{y:.0f} MW<extra></extra>'
        ))
    
    fig.update_layout(
        xaxis_title="Date",
        yaxis_title="Load (MW)",
        height=450,
        template=config.PLOTLY_TEMPLATE,
        hovermode='x unified',
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1)
    )
    
    st.plotly_chart(fig, width='stretch')


def render_distribution_analysis(data: pd.DataFrame):
    """Render load distribution histogram"""
    st.markdown("### Load Distribution")
    
    fig = go.Figure()
    
    fig.add_trace(go.Histogram(
        x=data[config.LOAD_COL],
        nbinsx=50,
        name='Load',
        marker_color=config.COLOR_PRIMARY,
        opacity=0.75
    ))
    
    # Add mean line
    mean_load = data[config.LOAD_COL].mean()
    fig.add_vline(
        x=mean_load,
        line_dash="dash",
        line_color=config.COLOR_ERROR,
        annotation_text=f"Mean: {mean_load:.0f} MW",
        annotation_position="top"
    )
    
    fig.update_layout(
        xaxis_title="Load (MW)",
        yaxis_title="Frequency",
        height=350,
        template=config.PLOTLY_TEMPLATE,
        showlegend=False
    )
    
    st.plotly_chart(fig, width='stretch')
    
    # Statistics
    q25 = data[config.LOAD_COL].quantile(0.25)
    q50 = data[config.LOAD_COL].quantile(0.50)
    q75 = data[config.LOAD_COL].quantile(0.75)
    
    st.markdown(f"""
    **Quartiles:**
    - Q1 (25%): {q25:.0f} MW
    - Q2 (50%): {q50:.0f} MW  
    - Q3 (75%): {q75:.0f} MW
    """)


def render_seasonal_patterns(data: pd.DataFrame):
    """Render seasonal pattern analysis"""
    st.markdown("### Hourly Pattern")
    
    data_copy = data.copy()
    data_copy['hour'] = data_copy.index.hour
    hourly_avg = data_copy.groupby('hour')[config.LOAD_COL].mean()
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=hourly_avg.index,
        y=hourly_avg.values,
        mode='lines+markers',
        name='Hourly Average',
        line=dict(color=config.COLOR_PRIMARY, width=2),
        marker=dict(size=8),
        fill='tozeroy',
        fillcolor=f'rgba(31, 119, 180, 0.1)'
    ))
    
    fig.update_layout(
        xaxis_title="Hour of Day",
        yaxis_title="Average Load (MW)",
        height=350,
        template=config.PLOTLY_TEMPLATE,
        showlegend=False
    )
    
    st.plotly_chart(fig, width='stretch')
    
    # Peak hours
    peak_hour = hourly_avg.idxmax()
    low_hour = hourly_avg.idxmin()
    
    st.markdown(f"""
    **Pattern Insights:**
    - Peak hour: {peak_hour}:00 ({hourly_avg.max():.0f} MW)
    - Low hour: {low_hour}:00 ({hourly_avg.min():.0f} MW)
    - Peak/Low ratio: {hourly_avg.max()/hourly_avg.min():.2f}x
    """)


def render_correlation_heatmap(data: pd.DataFrame):
    """Render feature correlation heatmap"""
    st.markdown("### Feature Correlations")
    
    # Select numeric columns
    numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
    
    if len(numeric_cols) < 2:
        st.info("Not enough numeric features for correlation analysis")
        return
    
    # Limit to most relevant features
    relevant_features = [col for col in numeric_cols if any(x in col for x in 
                        ['lag', 'rolling', 'hour', 'dayofweek', 'month', config.LOAD_COL])][:15]
    
    if config.LOAD_COL not in relevant_features:
        relevant_features = [config.LOAD_COL] + relevant_features[:14]
    
    corr_data = data[relevant_features].corr()
    
    fig = go.Figure(data=go.Heatmap(
        z=corr_data.values,
        x=corr_data.columns,
        y=corr_data.columns,
        colorscale='RdBu',
        zmid=0,
        text=corr_data.values,
        texttemplate='%{text:.2f}',
        textfont={"size": 8},
        colorbar=dict(title="Correlation")
    ))
    
    fig.update_layout(
        height=500,
        template=config.PLOTLY_TEMPLATE
    )
    
    st.plotly_chart(fig, width='stretch')


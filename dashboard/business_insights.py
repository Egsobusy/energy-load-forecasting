"""Business Insights Tab"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))
import config


def render_business_insights_tab(data: pd.DataFrame):
    """
    Render business insights and planning recommendations
    
    Args:
        data: Historical load data
    """
    st.markdown("## Business Insights")
    
    render_capacity_planning(data)
    
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    
    with col1:
        render_peak_demand_analysis(data)
    
    with col2:
        render_load_profile_analysis(data)
    
    st.markdown("---")
    
    render_operational_recommendations(data)


def render_capacity_planning(data: pd.DataFrame):
    """Capacity planning analysis"""
    st.markdown("### Capacity Planning")
    
    load_col = config.LOAD_COL
    
    # Calculate statistics
    peak_load = data[load_col].max()
    avg_load = data[load_col].mean()
    p95_load = data[load_col].quantile(0.95)
    p99_load = data[load_col].quantile(0.99)
    
    # Recommended capacity with reserve margin
    recommended_capacity = p99_load * (1 + config.RECOMMENDED_RESERVE_MARGIN)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Peak Load", f"{peak_load:.0f} MW")
    
    with col2:
        st.metric("95th Percentile", f"{p95_load:.0f} MW")
    
    with col3:
        st.metric("99th Percentile", f"{p99_load:.0f} MW")
    
    with col4:
        st.metric("Recommended Capacity", f"{recommended_capacity:.0f} MW",
                 f"+{config.RECOMMENDED_RESERVE_MARGIN*100:.0f}% margin")
    
    # Capacity planning chart
    fig = go.Figure()
    
    # Load duration curve
    sorted_load = np.sort(data[load_col].values)[::-1]
    percentiles = np.linspace(0, 100, len(sorted_load))
    
    fig.add_trace(go.Scatter(
        x=percentiles,
        y=sorted_load,
        mode='lines',
        name='Load Duration Curve',
        line=dict(color=config.COLOR_PRIMARY, width=2),
        fill='tozeroy',
        fillcolor=f'rgba(31, 119, 180, 0.2)'
    ))
    
    # Add capacity lines
    fig.add_hline(y=recommended_capacity, line_dash="dash", line_color=config.COLOR_SUCCESS,
                 annotation_text=f"Recommended: {recommended_capacity:.0f} MW")
    
    fig.add_hline(y=peak_load, line_dash="dot", line_color=config.COLOR_ERROR,
                 annotation_text=f"Historical Peak: {peak_load:.0f} MW")
    
    fig.update_layout(
        xaxis_title="Percentile (%)",
        yaxis_title="Load (MW)",
        height=400,
        template=config.PLOTLY_TEMPLATE
    )
    
    st.plotly_chart(fig, width='stretch')
    
    # Capacity utilization
    if 'forecast_result' in st.session_state:
        forecast = st.session_state['forecast_result']
        forecast_peak = np.max(forecast['predictions'])
        utilization = (forecast_peak / recommended_capacity) * 100
        
        if utilization > 90:
            st.warning(f"High capacity utilization forecast: {utilization:.1f}%")
        else:
            st.success(f"Forecast peak utilization: {utilization:.1f}%")


def render_peak_demand_analysis(data: pd.DataFrame):
    """Analyze peak demand patterns"""
    st.markdown("### Peak Demand Patterns")
    
    load_col = config.LOAD_COL
    
    # Identify top 1% peaks
    threshold = data[load_col].quantile(0.99)
    peak_hours = data[data[load_col] >= threshold].copy()
    
    # Peak by hour of day
    peak_hours['hour'] = peak_hours.index.hour
    peak_by_hour = peak_hours.groupby('hour').size()
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=peak_by_hour.index,
        y=peak_by_hour.values,
        marker_color=config.COLOR_ERROR,
        text=peak_by_hour.values,
        textposition='outside'
    ))
    
    fig.update_layout(
        xaxis_title="Hour of Day",
        yaxis_title="Number of Peak Events",
        height=300,
        template=config.PLOTLY_TEMPLATE,
        showlegend=False
    )
    
    st.plotly_chart(fig, width='stretch')
    
    # Peak statistics
    most_common_hour = peak_by_hour.idxmax()
    peak_hours['dayofweek'] = peak_hours.index.dayofweek
    most_common_day = peak_hours['dayofweek'].mode()[0]
    
    day_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    
    st.markdown(f"""
    **Peak Event Analysis:**
    - Total peak events (>99th percentile): {len(peak_hours)}
    - Most common hour: {most_common_hour}:00
    - Most common day: {day_names[most_common_day]}
    - Peak threshold: {threshold:.0f} MW
    """)


def render_load_profile_analysis(data: pd.DataFrame):
    """Analyze typical load profiles"""
    st.markdown("### Weekly Load Profile")
    
    load_col = config.LOAD_COL
    
    # Average by day of week and hour
    data_copy = data.copy()
    data_copy['hour'] = data_copy.index.hour
    data_copy['dayofweek'] = data_copy.index.dayofweek
    
    pivot = data_copy.pivot_table(
        values=load_col,
        index='hour',
        columns='dayofweek',
        aggfunc='mean'
    )
    
    day_names = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
    
    fig = go.Figure(data=go.Heatmap(
        z=pivot.values,
        x=day_names,
        y=list(range(24)),
        colorscale='YlOrRd',
        hovertemplate='%{x}, Hour %{y}<br>Avg: %{z:.0f} MW<extra></extra>',
        colorbar=dict(title="MW")
    ))
    
    fig.update_layout(
        xaxis_title="Day of Week",
        yaxis_title="Hour of Day",
        height=400,
        template=config.PLOTLY_TEMPLATE
    )
    
    st.plotly_chart(fig, width='stretch')
    
    # Weekday vs weekend comparison
    weekday_avg = data_copy[data_copy['dayofweek'] < 5][load_col].mean()
    weekend_avg = data_copy[data_copy['dayofweek'] >= 5][load_col].mean()
    difference = ((weekday_avg - weekend_avg) / weekend_avg) * 100
    
    st.markdown(f"""
    **Weekly Patterns:**
    - Weekday average: {weekday_avg:.0f} MW
    - Weekend average: {weekend_avg:.0f} MW
    - Difference: {difference:+.1f}%
    """)


def render_operational_recommendations(data: pd.DataFrame):
    """Provide operational recommendations"""
    st.markdown("### Operational Recommendations")
    
    load_col = config.LOAD_COL
    
    # Calculate key metrics
    recent_data = data.tail(24*30)
    recent_peak = recent_data[load_col].max()
    historical_peak = data[load_col].max()
    trend = recent_data[load_col].rolling(window=24*7).mean().iloc[-1]
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Short-term Actions")
        
        recommendations = []
        
        # Check if recent peaks are concerning
        if recent_peak > historical_peak * 0.95:
            recommendations.append("High recent demand - monitor capacity closely")
        
        # Check if forecast available
        if 'forecast_result' in st.session_state:
            forecast = st.session_state['forecast_result']
            forecast_peak = np.max(forecast['predictions'])
            
            if forecast_peak > data[load_col].quantile(0.99):
                recommendations.append(f"High load forecast: {forecast_peak:.0f} MW expected")
            
            # Check uncertainty
            uncertainty = forecast.get('uncertainty')
            if uncertainty is not None and np.max(uncertainty) > np.mean(uncertainty) * 1.5:
                recommendations.append("High forecast uncertainty - prepare backup capacity")
        
        if not recommendations:
            recommendations.append("Load patterns normal - maintain current operations")
        
        for rec in recommendations:
            st.markdown(f"- {rec}")
    
    with col2:
        st.markdown("#### Long-term Planning")
        
        # Calculate growth rate
        if len(data) > 24*365:
            recent_year = data.tail(24*365)[load_col].mean()
            previous_year = data.iloc[-(24*365*2):-(24*365)][load_col].mean()
            growth_rate = ((recent_year - previous_year) / previous_year) * 100
            
            st.markdown(f"- Annual load growth: {growth_rate:+.1f}%")
            
            if growth_rate > 3:
                st.markdown("- Consider capacity expansion within 2-3 years")
            elif growth_rate > 1:
                st.markdown("- Monitor demand trends for capacity planning")
            else:
                st.markdown("- Current capacity adequate for near-term")
        
        # Variability assessment
        cv = (data[load_col].std() / data[load_col].mean()) * 100
        st.markdown(f"- Load variability (CV): {cv:.1f}%")
        
        if cv > 20:
            st.markdown("- High variability - invest in flexible capacity")
        else:
            st.markdown("- Stable load patterns - optimize baseload capacity")
    
    # Export recommendations
    st.markdown("---")
    
    if st.button("Generate Full Report"):
        report_content = generate_report(data)
        st.download_button(
            label="Download Report",
            data=report_content,
            file_name=f"energy_forecast_report_{pd.Timestamp.now().strftime('%Y%m%d')}.txt",
            mime="text/plain"
        )


def generate_report(data: pd.DataFrame) -> str:
    """Generate text report with key insights"""
    load_col = config.LOAD_COL
    
    report = f"""
ENERGY LOAD FORECASTING REPORT
Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}

DATA SUMMARY
============
Total Records: {len(data):,}
Date Range: {data.index.min()} to {data.index.max()}
Mean Load: {data[load_col].mean():.0f} MW
Peak Load: {data[load_col].max():.0f} MW
Min Load: {data[load_col].min():.0f} MW
Std Dev: {data[load_col].std():.0f} MW

CAPACITY PLANNING
=================
95th Percentile: {data[load_col].quantile(0.95):.0f} MW
99th Percentile: {data[load_col].quantile(0.99):.0f} MW
Recommended Capacity: {data[load_col].quantile(0.99) * 1.15:.0f} MW (with 15% margin)

OPERATIONAL INSIGHTS
====================
- Peak demand typically occurs during business hours (9 AM - 6 PM)
- Weekend demand is lower than weekday demand
- Seasonal variations should be monitored for capacity planning
- Forecast accuracy metrics available in dashboard

RECOMMENDATIONS
===============
- Monitor capacity utilization regularly
- Plan maintenance during low-demand periods
- Maintain reserve margin for unexpected peaks
- Review forecasts weekly for planning updates
"""
    
    return report


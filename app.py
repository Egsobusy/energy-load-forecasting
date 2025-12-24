"""Energy Load Forecasting Dashboard"""

import streamlit as st
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent))

import config
from core.utils import load_raw_data, load_processed_data
from dashboard.data_overview import render_data_overview_tab
from dashboard.forecasting import render_forecasting_tab
from dashboard.model_performance import render_model_performance_tab
from dashboard.business_insights import render_business_insights_tab

st.set_page_config(
    page_title=config.STREAMLIT_PAGE_TITLE,
    page_icon=config.STREAMLIT_PAGE_ICON,
    layout=config.STREAMLIT_LAYOUT,
    initial_sidebar_state=config.STREAMLIT_INITIAL_SIDEBAR_STATE
)

st.markdown("""
<style>
    .stApp {
        background: linear-gradient(135deg, #1a237e 0%, #0d47a1 50%, #01579b 100%);
    }
    
    .block-container {
        background-color: #ffffff;
        border-radius: 12px;
        padding: 2.5rem;
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.3);
        margin-top: 1rem;
    }
    
    .main-header {
        font-size: 2.8rem;
        color: #0d47a1;
        font-weight: 800;
        text-align: center;
        margin-bottom: 0.5rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.15);
    }
    
    .subtitle {
        font-size: 1.15rem;
        color: #212121;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: 600;
    }
    
    div[data-testid="stMetricValue"] {
        font-size: 2rem;
        font-weight: 800;
        color: #0d47a1;
    }
    
    div[data-testid="stMetricLabel"] {
        color: #424242;
        font-weight: 700;
        font-size: 0.95rem;
    }
    
    div[data-testid="stMetricDelta"] {
        font-weight: 600;
    }
    
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0d47a1 0%, #01579b 100%);
    }
    
    section[data-testid="stSidebar"] .stMarkdown {
        color: #ffffff;
    }
    
    section[data-testid="stSidebar"] h2,
    section[data-testid="stSidebar"] h3 {
        color: #ffffff;
        font-weight: 800;
        text-shadow: 1px 1px 2px rgba(0,0,0,0.3);
    }
    
    section[data-testid="stSidebar"] label {
        color: #ffffff;
        font-weight: 600;
    }
    
    section[data-testid="stSidebar"] p {
        color: #e3f2fd;
    }
    
    section[data-testid="stSidebar"] div[data-testid="stMetricValue"] {
        color: #ffffff;
        font-size: 1.8rem;
        font-weight: 800;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    }
    
    section[data-testid="stSidebar"] div[data-testid="stMetricLabel"] {
        color: #bbdefb;
        font-weight: 600;
    }
    
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: #e8eaf6;
        border-radius: 10px;
        padding: 8px;
        border: 2px solid #c5cae9;
    }
    
    .stTabs [data-baseweb="tab"] {
        background-color: #ffffff;
        border-radius: 8px;
        color: #424242;
        font-weight: 700;
        padding: 12px 24px;
        border: 2px solid transparent;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #1565c0 0%, #0d47a1 100%);
        color: #ffffff;
        border: 2px solid #0d47a1;
        box-shadow: 0 4px 8px rgba(13, 71, 161, 0.3);
    }
    
    .stButton > button {
        background: linear-gradient(135deg, #1565c0 0%, #0d47a1 100%);
        color: #ffffff;
        border: none;
        border-radius: 8px;
        font-weight: 700;
        font-size: 1rem;
        padding: 12px 28px;
        transition: all 0.3s;
        box-shadow: 0 4px 12px rgba(13, 71, 161, 0.3);
    }
    
    .stButton > button:hover {
        transform: translateY(-3px);
        box-shadow: 0 8px 20px rgba(13, 71, 161, 0.5);
        background: linear-gradient(135deg, #0d47a1 0%, #01579b 100%);
    }
    
    .dataframe {
        border: 2px solid #e0e0e0;
        border-radius: 8px;
        overflow: hidden;
    }
    
    .stAlert {
        border-radius: 8px;
        border-left: 5px solid #1565c0;
        background-color: #e3f2fd;
        color: #0d47a1;
        font-weight: 600;
    }
    
    h1, h2, h3 {
        color: #0d47a1;
        font-weight: 700;
    }
    
    p {
        color: #212121;
    }
    
    .stMarkdown {
        color: #212121;
    }
    
    .stRadio > label,
    .stSelectbox > label,
    .stSlider > label {
        color: #424242;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

st.markdown(f"<h1 class='main-header'>{config.APP_TITLE}</h1>", unsafe_allow_html=True)
st.markdown(f"<p class='subtitle'>{config.APP_SUBTITLE}</p>", unsafe_allow_html=True)

with st.sidebar:
    st.markdown("## Settings")
    st.markdown("---")
    
    st.markdown("### Data Source")
    data_source = st.radio(
        "Select data",
        options=["Processed Data", "Raw Data"],
        help="Choose data source for analysis"
    )
    
    st.markdown("---")
    
    st.markdown("### Forecast Configuration")
    
    default_horizon = st.selectbox(
        "Default Horizon",
        options=[24, 48, 72, 168, 336],
        index=3,
        format_func=lambda x: f"{x}h ({x//24}d)"
    )
    
    default_confidence = st.select_slider(
        "Confidence Level",
        options=[0.80, 0.90, 0.95, 0.99],
        value=0.95,
        format_func=lambda x: f"{x*100:.0f}%"
    )
    
    st.session_state['default_horizon'] = default_horizon
    st.session_state['default_confidence'] = default_confidence
    
    st.markdown("---")
    
    st.markdown("### Model Information")
    st.markdown("""
    **Ensemble Models:**
    - Random Forest (35%)
    - MLP Neural Network (30%)
    - ARIMA (20%)
    - LSTM (15%)
    """)
    
    st.markdown("---")
    
    st.markdown("### Data Summary")
    if 'data_loaded' in st.session_state and st.session_state['data_loaded']:
        data_info = st.session_state.get('data_summary', {})
        st.metric("Records", f"{data_info.get('records', 0):,}")
        st.metric("Date Range", f"{data_info.get('years', 0):.1f} years")
        missing_pct = data_info.get('missing_pct', 0)
        st.metric("Data Quality", f"{100-missing_pct:.1f}%")
    else:
        st.info("Load data to see summary")


@st.cache_data
def load_data_cached(source='processed'):
    """Load and cache data"""
    try:
        if source == 'processed':
            data = load_processed_data()
            if data is None:
                st.warning("Processed data not found, loading raw data")
                data = load_raw_data()
        else:
            data = load_raw_data()
        
        if data is not None:
            summary = {
                'records': len(data),
                'years': round((data.index.max() - data.index.min()).days / 365, 1),
                'missing_pct': (data[config.LOAD_COL].isna().sum() / len(data)) * 100
            }
            return data, summary
        
        return None, None
        
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None, None


data, data_summary = load_data_cached('processed' if data_source == 'Processed Data' else 'raw')

if data is None:
    st.error("""
    ### Data Not Available
    
    Please ensure data files exist:
    - `data/raw/pjm_hourly_est.csv` (raw data)
    - `data/processed/pjm_processed.csv` (processed data)
    
    To generate processed data, run:
    ```
    python core/data_processing.py
    ```
    """)
    st.stop()

st.session_state['data_loaded'] = True
st.session_state['data_summary'] = data_summary

tab1, tab2, tab3, tab4 = st.tabs([
    config.TAB_NAMES['overview'],
    config.TAB_NAMES['forecasting'],
    config.TAB_NAMES['performance'],
    config.TAB_NAMES['insights']
])

with tab1:
    render_data_overview_tab(data)

with tab2:
    render_forecasting_tab(data)

with tab3:
    render_model_performance_tab(data)

with tab4:
    render_business_insights_tab(data)

st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; padding: 15px; font-size: 0.9rem;'>
    <p><b>Energy Load Forecasting Dashboard</b> | Introduction to Business Analytics</p>
    <p>PJM Interconnection Data | Ensemble Machine Learning Framework</p>
</div>
""", unsafe_allow_html=True)


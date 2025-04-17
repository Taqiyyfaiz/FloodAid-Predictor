import streamlit as st
import os
import sys
import pandas as pd
import numpy as np

# Add the current directory to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import components
from components.dashboard_component import dashboard_tab
from components.map_component import map_tab
from components.analytics_component import analytics_tab

# Import utilities
from utils.database import create_tables
from utils.geo_utils import create_grid, generate_elevation_data, calculate_flow_accumulation
from utils.weather_api import get_grid_weather_data
from models.flood_prediction import train_model
from config.config import DEFAULT_LATITUDE, DEFAULT_LONGITUDE

# Page configuration
st.set_page_config(
    page_title="FloodAid Predictor",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS to customize the app appearance
st.markdown(
    """
    <style>
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    h1, h2, h3 {
        color: #2c3e50;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 2rem;
    }
    .stTabs [data-baseweb="tab"] {
        height: 4rem;
        white-space: pre-wrap;
        font-size: 1rem;
    }
    </style>
    """,
    unsafe_allow_html=True
)

def initialize_app():
    """Initialize the app by setting up database and models."""
    # Set up database tables
    create_tables()
    
    # Initialize the flood prediction model if needed
    if 'model_initialized' not in st.session_state:
        with st.spinner("Setting up the prediction model..."):
            # Check if the model exists, if not, train it
            from models.flood_prediction import load_or_train_model
            model = load_or_train_model()
            st.session_state.model_initialized = True

def sidebar():
    """Create the sidebar."""
    # Remove the image
    # st.sidebar.image("https://img.freepik.com/free-vector/natural-disaster-concept-illustrated_23-2148536131.jpg?w=900&t=st=1682290772~exp=1682291372~hmac=9f8f6e26b8e84d3f9f7547d06b72cea4ecd1a9bde16cd4c3010d1d1e6af00411", use_column_width=True)
    
    st.sidebar.title("FloodAid Predictor")
    st.sidebar.markdown("### Flood Prediction & Response Tool")
    
    st.sidebar.markdown("""
    This tool helps predict flood risk zones and plan the fastest, safest routes for aid delivery during flood events.
    """)
    
    st.sidebar.markdown("---")
    
    # About the project
    with st.sidebar.expander("About the Project"):
        st.markdown("""
        **FloodAid Predictor** combines machine learning for flood prediction with route optimization algorithms to:
        
        1. Predict areas at risk of flooding
        2. Optimize aid delivery routes
        3. Monitor flood events in real-time
        
        This prototype demonstrates how technology can be used to optimize humanitarian aid during natural disasters.
        """)
    
    # Demo controls
    st.sidebar.markdown("### Demo Controls")
    
    # Region selection
    region = st.sidebar.selectbox(
        "Select Region",
        ["Mumbai, India", "Chennai, India", "Dhaka, Bangladesh", "Jakarta, Indonesia"],
        key="sidebar_region"
    )
    
    # Rainfall intensity
    rainfall = st.sidebar.slider(
        "Rainfall Intensity (mm)",
        min_value=0,
        max_value=50,
        value=15,
        step=5,
        key="sidebar_rainfall"
    )
    
    # Generate data button
    if st.sidebar.button("Generate New Scenario"):
        # Set coordinates based on selected region
        if region == "Mumbai, India":
            center_lat, center_lon = 19.0760, 72.8777
        elif region == "Chennai, India":
            center_lat, center_lon = 13.0827, 80.2707
        elif region == "Dhaka, Bangladesh":
            center_lat, center_lon = 23.8103, 90.4125
        elif region == "Jakarta, Indonesia":
            center_lat, center_lon = -6.2088, 106.8456
        else:
            center_lat, center_lon = DEFAULT_LATITUDE, DEFAULT_LONGITUDE
        
        # Clear session state
        if 'flood_data' in st.session_state:
            del st.session_state.flood_data
        if 'routes' in st.session_state:
            del st.session_state.routes
        
        # Generate new data
        with st.spinner("Generating new flood risk data..."):
            # Create grid
            grid = create_grid(center_lat, center_lon, 20, spacing=0.01)
            grid = generate_elevation_data(grid)
            grid = calculate_flow_accumulation(grid)
            
            # Get rainfall data
            rainfall_data = get_grid_weather_data(center_lat, center_lon, grid_size=20)
            
            # Adjust rainfall by the intensity factor
            intensity_factor = rainfall / 15  # Normalize to the default of 15mm
            rainfall_data['rainfall'] = rainfall_data['rainfall'] * intensity_factor
            
            # Calculate flood risk
            from utils.geo_utils import calculate_flood_risk
            flood_data = calculate_flood_risk(grid, rainfall_data)
            
            # Store in session state
            st.session_state.flood_data = flood_data
            st.session_state.center_lat = center_lat
            st.session_state.center_lon = center_lon
        
        # Success message
        st.sidebar.success("New scenario generated!")
    
    st.sidebar.markdown("---")
    
    # Add development status notification to sidebar
    st.sidebar.warning("""
    **🚧 BETA VERSION 🚧**
    
    This tool is in active development. Features may change or have limited functionality.
    
    [Learn more about our roadmap](https://github.com/yourusername/FloodAid-Predictor)
    """)
    
    # Footer
    st.sidebar.markdown("""
    **Made with ❤️ by CODEX 2.0** &nbsp; [<img src="https://github.githubassets.com/images/modules/logos_page/GitHub-Mark.png" width="25">](https://github.com/codex-team/floodaid)
    """, unsafe_allow_html=True)

def main():
    """Main application function."""
    # Initialize the app
    initialize_app()
    
    # Create sidebar
    sidebar()
    
    # Remove the header image
    # st.image("https://img.freepik.com/free-vector/flood-disaster-concept_23-2148533782.jpg?w=1480&t=st=1682290603~exp=1682291203~hmac=ccbb74acccfed91cfc24c5c3adf7c00f5da5597731a40839816fc5fd22da1b33", use_column_width=True)
    
    # Create tabs for different sections
    tab1, tab2, tab3 = st.tabs(["📊 Dashboard", "🗺️ Map & Routes", "📈 Analytics"])
    
    # Add development status notification
    st.info("""
    ### 🚧 Development in Progress 🚧
    
    **Note:** FloodAid Predictor is currently under active development and is not a complete product. We are working on several improvements including:
    
    - Integration with real-time weather and flood monitoring systems
    - Enhanced machine learning models for more accurate predictions
    - Improved route optimization for different vehicle types
    - Mobile compatibility for field workers
    - Integration with actual road network data
    
    We welcome collaborators with expertise in machine learning, geospatial analysis, disaster management, and development. If you're interested in contributing, please visit our GitHub repository.
    """)
    
    with tab1:
        dashboard_tab()
    
    with tab2:
        map_tab()
    
    with tab3:
        analytics_tab()

if __name__ == "__main__":
    main() 
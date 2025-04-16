import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
import os

# Add parent directory to path to import utils and config
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.weather_api import get_current_weather
from config.config import DEFAULT_LATITUDE, DEFAULT_LONGITUDE

def dashboard_tab():
    """Create the main dashboard tab for the Streamlit app."""
    st.header("FloodAid Dashboard")
    
    # Date and time
    current_time = datetime.now().strftime("%B %d, %Y %H:%M:%S")
    st.subheader(f"Status as of {current_time}")
    
    # Create three columns for key metrics
    col1, col2, col3 = st.columns(3)
    
    # Get current weather for the default location
    weather = get_current_weather(DEFAULT_LATITUDE, DEFAULT_LONGITUDE)
    
    with col1:
        st.markdown(
            """
            <div style="border:1px solid #ddd; border-radius:5px; padding:1rem; text-align:center;">
                <h3>Current Weather</h3>
            </div>
            """,
            unsafe_allow_html=True
        )
        st.metric(
            "Rainfall",
            f"{weather['rainfall']:.1f} mm",
            f"{weather['rainfall'] - 5:.1f} mm" if weather['rainfall'] > 5 else f"{weather['rainfall'] - 5:.1f} mm"
        )
        st.metric(
            "Temperature",
            f"{weather['temperature']:.1f}°C",
            f"{weather['temperature'] - 28:.1f}°C"
        )
        st.metric(
            "Humidity",
            f"{weather['humidity']:.0f}%",
            f"{weather['humidity'] - 70:.0f}%"
        )
    
    with col2:
        # Flood risk summary
        st.markdown(
            """
            <div style="border:1px solid #ddd; border-radius:5px; padding:1rem; text-align:center;">
                <h3>Flood Risk Summary</h3>
            </div>
            """,
            unsafe_allow_html=True
        )
        
        # Get flood risk data from session state or use defaults
        if 'flood_data' in st.session_state:
            flood_data = st.session_state.flood_data
            risk_counts = flood_data['risk_level'].value_counts().to_dict()
            total_points = len(flood_data)
            
            # Calculate percentages
            low_pct = risk_counts.get('low', 0) / total_points * 100 if total_points > 0 else 0
            medium_pct = risk_counts.get('medium', 0) / total_points * 100 if total_points > 0 else 0
            high_pct = risk_counts.get('high', 0) / total_points * 100 if total_points > 0 else 0
            severe_pct = risk_counts.get('severe', 0) / total_points * 100 if total_points > 0 else 0
            
            high_severe_pct = high_pct + severe_pct
        else:
            # Default values
            low_pct = 40
            medium_pct = 30
            high_pct = 20
            severe_pct = 10
            high_severe_pct = high_pct + severe_pct
        
        st.metric(
            "High & Severe Risk Areas",
            f"{high_severe_pct:.1f}%",
            f"{high_severe_pct - 25:.1f}%" if high_severe_pct > 25 else f"{high_severe_pct - 25:.1f}%"
        )
        
        # Create a progress bar to visualize risk distribution
        st.markdown("<p style='margin-bottom:0.2rem;'>Risk Distribution:</p>", unsafe_allow_html=True)
        
        # Create bars for each risk level
        st.markdown(
            f"""
            <div style="display:flex; width:100%; height:24px; border-radius:3px; overflow:hidden;">
                <div style="width:{low_pct}%; background-color:#fef0d9;"></div>
                <div style="width:{medium_pct}%; background-color:#fdcc8a;"></div>
                <div style="width:{high_pct}%; background-color:#fc8d59;"></div>
                <div style="width:{severe_pct}%; background-color:#d7301f;"></div>
            </div>
            <div style="display:flex; justify-content:space-between; font-size:0.8rem; margin-top:0.2rem;">
                <div>Low</div>
                <div>Medium</div>
                <div>High</div>
                <div>Severe</div>
            </div>
            """,
            unsafe_allow_html=True
        )
        
        # Affected population estimate
        st.metric(
            "Est. Affected Population",
            f"{int(high_severe_pct * 20000):,}",
            f"{int((high_severe_pct - 25) * 20000):,}" 
        )
    
    with col3:
        # Aid delivery status
        st.markdown(
            """
            <div style="border:1px solid #ddd; border-radius:5px; padding:1rem; text-align:center;">
                <h3>Aid Delivery Status</h3>
            </div>
            """,
            unsafe_allow_html=True
        )
        
        # Get route data from session state or use defaults
        if 'routes' in st.session_state and st.session_state.routes:
            routes = st.session_state.routes
            total_routes = len(routes)
            total_areas_served = sum(len(route['affected_areas']) for route in routes)
            avg_delivery_time = np.mean([route['time'] for route in routes]) if routes else 0
        else:
            # Default values
            total_routes = 3
            total_areas_served = 8
            avg_delivery_time = 45
        
        st.metric(
            "Active Aid Routes",
            total_routes,
            1
        )
        st.metric(
            "Areas Being Served",
            total_areas_served,
            2
        )
        st.metric(
            "Avg Delivery Time",
            f"{avg_delivery_time:.0f} min",
            f"{45 - avg_delivery_time:.0f} min" if avg_delivery_time < 45 else f"{avg_delivery_time - 45:.0f} min",
            delta_color="inverse"
        )
    
    # Create a status timeline
    st.subheader("Recent Activity")
    
    # Generate some synthetic timeline events
    now = datetime.now()
    
    timeline_events = [
        {
            "time": (now - timedelta(minutes=5)).strftime("%H:%M"),
            "event": "New high-risk area identified in Thane district",
            "type": "alert"
        },
        {
            "time": (now - timedelta(minutes=15)).strftime("%H:%M"),
            "event": "Aid route from Mumbai Central to Dadar optimized",
            "type": "route"
        },
        {
            "time": (now - timedelta(minutes=32)).strftime("%H:%M"),
            "event": "Weather update: Rainfall intensity increased to 18mm/hr",
            "type": "weather"
        },
        {
            "time": (now - timedelta(hours=1)).strftime("%H:%M"),
            "event": "5 affected areas have received aid supplies",
            "type": "delivery"
        },
        {
            "time": (now - timedelta(hours=2)).strftime("%H:%M"),
            "event": "Flood prediction model updated with new rainfall data",
            "type": "system"
        }
    ]
    
    # Create a timeline display
    for event in timeline_events:
        event_type = event["type"]
        
        # Define icon and color based on event type
        if event_type == "alert":
            icon = "🚨"
            color = "#d7301f"
        elif event_type == "route":
            icon = "🚚"
            color = "#2171b5"
        elif event_type == "weather":
            icon = "🌧️"
            color = "#6baed6"
        elif event_type == "delivery":
            icon = "📦"
            color = "#238b45"
        else:
            icon = "🔄"
            color = "#525252"
        
        # Display event with styling
        st.markdown(
            f"""
            <div style="display:flex; align-items:start; margin-bottom:0.8rem;">
                <div style="background-color:{color}; color:white; border-radius:50%; width:28px; height:28px; display:flex; justify-content:center; align-items:center; margin-right:10px; flex-shrink:0;">
                    {icon}
                </div>
                <div style="flex-grow:1;">
                    <div style="font-size:0.8rem; color:#777;">{event["time"]}</div>
                    <div>{event["event"]}</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )
    
    # System status
    st.subheader("System Status")
    
    # Create columns for status indicators
    status_col1, status_col2, status_col3, status_col4 = st.columns(4)
    
    with status_col1:
        st.markdown(
            """
            <div style="border:1px solid #ddd; border-radius:5px; padding:1rem; text-align:center;">
                <div style="font-size:0.9rem; color:#777;">Weather API</div>
                <div style="font-size:1.1rem; color:#238b45; margin-top:0.5rem;">● Online</div>
            </div>
            """,
            unsafe_allow_html=True
        )
    
    with status_col2:
        st.markdown(
            """
            <div style="border:1px solid #ddd; border-radius:5px; padding:1rem; text-align:center;">
                <div style="font-size:0.9rem; color:#777;">Prediction Model</div>
                <div style="font-size:1.1rem; color:#238b45; margin-top:0.5rem;">● Active</div>
            </div>
            """,
            unsafe_allow_html=True
        )
    
    with status_col3:
        st.markdown(
            """
            <div style="border:1px solid #ddd; border-radius:5px; padding:1rem; text-align:center;">
                <div style="font-size:0.9rem; color:#777;">Route Optimizer</div>
                <div style="font-size:1.1rem; color:#238b45; margin-top:0.5rem;">● Running</div>
            </div>
            """,
            unsafe_allow_html=True
        )
    
    with status_col4:
        st.markdown(
            """
            <div style="border:1px solid #ddd; border-radius:5px; padding:1rem; text-align:center;">
                <div style="font-size:0.9rem; color:#777;">Database</div>
                <div style="font-size:1.1rem; color:#238b45; margin-top:0.5rem;">● Connected</div>
            </div>
            """,
            unsafe_allow_html=True
        ) 
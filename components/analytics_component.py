import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import os
from datetime import datetime, timedelta

# Add parent directory to path to import utils and models
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.weather_api import get_historical_rainfall, get_forecast
from utils.database import get_flood_predictions, get_routes
from models.flood_prediction import load_or_train_model
from config.config import DEFAULT_LATITUDE, DEFAULT_LONGITUDE

def plot_rainfall_history(lat=DEFAULT_LATITUDE, lon=DEFAULT_LONGITUDE, days=30):
    """
    Plot historical rainfall data.
    
    Args:
        lat (float): Latitude
        lon (float): Longitude
        days (int): Number of days
    """
    # Get historical rainfall data
    df = get_historical_rainfall(lat, lon)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 4))
    
    # Plot rainfall
    ax.bar(df['date'], df['rainfall'], color='skyblue', alpha=0.7)
    ax.set_title('Historical Rainfall (Last 30 Days)')
    ax.set_xlabel('Date')
    ax.set_ylabel('Rainfall (mm)')
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Format x-axis ticks
    plt.xticks(rotation=45)
    
    # Show only some x-axis ticks to prevent crowding
    ax.set_xticks(df['date'][::5])
    
    plt.tight_layout()
    
    return fig

def plot_rainfall_forecast(lat=DEFAULT_LATITUDE, lon=DEFAULT_LONGITUDE, days=5):
    """
    Plot rainfall forecast.
    
    Args:
        lat (float): Latitude
        lon (float): Longitude
        days (int): Number of days
    """
    # Get forecast data
    df = get_forecast(lat, lon, days)
    
    # Group by date and calculate daily rainfall
    daily_rainfall = df.groupby('date')['rainfall'].sum().reset_index()
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 4))
    
    # Plot rainfall forecast
    ax.bar(daily_rainfall['date'], daily_rainfall['rainfall'], color='skyblue', alpha=0.7)
    ax.set_title(f'Rainfall Forecast (Next {days} Days)')
    ax.set_xlabel('Date')
    ax.set_ylabel('Rainfall (mm)')
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Format x-axis ticks
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    
    return fig

def plot_risk_distribution(flood_data=None):
    """
    Plot flood risk distribution.
    
    Args:
        flood_data (GeoDataFrame): Flood risk data
    """
    if flood_data is None:
        # Use data from database as fallback
        flood_data = get_flood_predictions()
        
        if flood_data.empty:
            # Generate synthetic data
            risk_levels = ['low', 'medium', 'high', 'severe']
            risk_counts = [40, 30, 20, 10]  # Example distribution
            
            risk_df = pd.DataFrame({
                'risk_level': risk_levels,
                'count': risk_counts
            })
        else:
            # Use actual data
            risk_counts = flood_data['risk_level'].value_counts().reset_index()
            # Update column names to handle newer pandas versions
            risk_df = pd.DataFrame({
                'risk_level': risk_counts.iloc[:, 0],
                'count': risk_counts.iloc[:, 1]
            })
    else:
        # Use provided data
        risk_counts = flood_data['risk_level'].value_counts().reset_index()
        # Update column names to handle newer pandas versions
        risk_df = pd.DataFrame({
            'risk_level': risk_counts.iloc[:, 0],
            'count': risk_counts.iloc[:, 1]
        })
    
    # Create figure
    fig, ax = plt.subplots(figsize=(8, 4))
    
    # Set colors according to risk level
    colors = ['#fef0d9', '#fdcc8a', '#fc8d59', '#d7301f']
    
    # Plot bar chart
    sns.barplot(x='risk_level', y='count', data=risk_df, palette=colors, ax=ax)
    
    ax.set_title('Flood Risk Distribution')
    ax.set_xlabel('Risk Level')
    ax.set_ylabel('Number of Areas')
    
    plt.tight_layout()
    
    return fig

def plot_route_metrics(routes=None):
    """
    Plot route metrics.
    
    Args:
        routes (list): List of route dictionaries
    """
    if routes is None or not routes:
        # Use data from database as fallback
        route_data = get_routes()
        
        if route_data.empty:
            # Generate synthetic data
            aid_centers = ['Mumbai Central', 'Thane Center', 'Navi Mumbai Hub']
            distances = [15.2, 18.7, 12.3]
            times = [45, 56, 37]
            
            route_df = pd.DataFrame({
                'aid_center': aid_centers,
                'distance': distances,
                'time': times
            })
        else:
            # Use actual data
            route_df = route_data
    else:
        # Use provided data
        route_df = pd.DataFrame([
            {
                'aid_center': route['aid_center']['name'],
                'distance': route['distance'],
                'time': route['time'],
                'affected_areas': len(route['affected_areas']),
                'risk_level': route['risk_level']
            }
            for route in routes
        ])
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Plot distances
    sns.barplot(x='aid_center', y='distance', data=route_df, palette='Blues_d', ax=ax1)
    ax1.set_title('Route Distances by Aid Center')
    ax1.set_xlabel('Aid Center')
    ax1.set_ylabel('Distance (km)')
    ax1.tick_params(axis='x', rotation=45)
    
    # Plot times
    sns.barplot(x='aid_center', y='time', data=route_df, palette='Oranges_d', ax=ax2)
    ax2.set_title('Route Times by Aid Center')
    ax2.set_xlabel('Aid Center')
    ax2.set_ylabel('Time (minutes)')
    ax2.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    
    return fig

def predict_flood_risk(lat, lon, rainfall, elevation=None, flow_acc=None):
    """
    Predict flood risk for a specific location.
    
    Args:
        lat (float): Latitude
        lon (float): Longitude
        rainfall (float): Rainfall in mm
        elevation (float, optional): Elevation in meters
        flow_acc (float, optional): Flow accumulation
        
    Returns:
        dict: Risk prediction
    """
    # Load the flood prediction model
    model = load_or_train_model()
    
    # If elevation or flow accumulation are not provided, estimate them
    if elevation is None or flow_acc is None:
        import geopandas as gpd
        from shapely.geometry import Point
        from utils.geo_utils import create_grid, generate_elevation_data, calculate_flow_accumulation
        
        # Create a small grid around the point
        grid = create_grid(lat, lon, 3, spacing=0.001)
        grid = generate_elevation_data(grid)
        grid = calculate_flow_accumulation(grid)
        
        # Find the center point
        center_point = grid.iloc[4]  # Center of a 3x3 grid
        
        if elevation is None:
            elevation = center_point['elevation']
        
        if flow_acc is None:
            flow_acc = center_point['flow_acc']
    
    # Create a feature dataframe
    features = pd.DataFrame({
        'elevation': [elevation],
        'flow_acc': [flow_acc],
        'nearest_rainfall': [rainfall]
    })
    
    # Make prediction
    risk_score = model.predict(features)[0]
    
    # Classify risk level
    if risk_score < 0.25:
        risk_level = 'low'
    elif risk_score < 0.5:
        risk_level = 'medium'
    elif risk_score < 0.75:
        risk_level = 'high'
    else:
        risk_level = 'severe'
    
    return {
        'risk_score': risk_score,
        'risk_level': risk_level,
        'elevation': elevation,
        'flow_acc': flow_acc,
        'rainfall': rainfall
    }

def analytics_tab():
    """
    Create the Analytics tab for the Streamlit app.
    """
    st.header("Flood Risk Analytics & Predictions")
    
    # Create tabs for different analysis views
    tab1, tab2, tab3 = st.tabs(["Historical Analysis", "Forecast", "Risk Prediction"])
    
    with tab1:
        st.subheader("Historical Rainfall Analysis")
        
        # Region selection
        region = st.selectbox(
            "Select Region",
            ["Mumbai, India", "Chennai, India", "Dhaka, Bangladesh", "Jakarta, Indonesia"],
            key="region_hist"
        )
        
        # Set coordinates based on selected region
        if region == "Mumbai, India":
            lat, lon = 19.0760, 72.8777
        elif region == "Chennai, India":
            lat, lon = 13.0827, 80.2707
        elif region == "Dhaka, Bangladesh":
            lat, lon = 23.8103, 90.4125
        elif region == "Jakarta, Indonesia":
            lat, lon = -6.2088, 106.8456
        else:
            lat, lon = DEFAULT_LATITUDE, DEFAULT_LONGITUDE
        
        # Historical rainfall plot
        fig1 = plot_rainfall_history(lat, lon)
        st.pyplot(fig1)
        
        # Risk distribution
        if 'flood_data' in st.session_state:
            st.subheader("Flood Risk Distribution")
            fig2 = plot_risk_distribution(st.session_state.flood_data)
            st.pyplot(fig2)
        
        # Route metrics
        if 'routes' in st.session_state and st.session_state.routes:
            st.subheader("Aid Delivery Route Analysis")
            fig3 = plot_route_metrics(st.session_state.routes)
            st.pyplot(fig3)
    
    with tab2:
        st.subheader("Rainfall Forecast")
        
        # Region selection
        region = st.selectbox(
            "Select Region",
            ["Mumbai, India", "Chennai, India", "Dhaka, Bangladesh", "Jakarta, Indonesia"],
            key="region_forecast"
        )
        
        # Forecast days
        days = st.slider("Forecast Days", min_value=1, max_value=7, value=5)
        
        # Set coordinates based on selected region
        if region == "Mumbai, India":
            lat, lon = 19.0760, 72.8777
        elif region == "Chennai, India":
            lat, lon = 13.0827, 80.2707
        elif region == "Dhaka, Bangladesh":
            lat, lon = 23.8103, 90.4125
        elif region == "Jakarta, Indonesia":
            lat, lon = -6.2088, 106.8456
        else:
            lat, lon = DEFAULT_LATITUDE, DEFAULT_LONGITUDE
        
        # Forecast plot
        fig = plot_rainfall_forecast(lat, lon, days)
        st.pyplot(fig)
        
        # Daily forecast table
        st.subheader("Daily Rainfall Forecast")
        
        # Get forecast data
        df = get_forecast(lat, lon, days)
        
        # Group by date and calculate daily rainfall
        daily_rainfall = df.groupby('date')['rainfall'].sum().reset_index()
        
        # Add date formatting and future risk estimation
        daily_rainfall['date_formatted'] = pd.to_datetime(daily_rainfall['date']).dt.strftime('%a, %b %d')
        
        # Estimate risk levels based on rainfall
        def estimate_risk(rainfall):
            if rainfall < 5:
                return 'Low'
            elif rainfall < 15:
                return 'Medium'
            elif rainfall < 30:
                return 'High'
            else:
                return 'Severe'
        
        daily_rainfall['estimated_risk'] = daily_rainfall['rainfall'].apply(estimate_risk)
        
        # Display table
        st.dataframe(
            daily_rainfall[['date_formatted', 'rainfall', 'estimated_risk']].rename(
                columns={
                    'date_formatted': 'Date',
                    'rainfall': 'Rainfall (mm)',
                    'estimated_risk': 'Estimated Risk'
                }
            ),
            use_container_width=True
        )
    
    with tab3:
        st.subheader("Flood Risk Prediction")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Input form for prediction
            st.write("Enter location and weather details:")
            
            # Region selection with coordinates
            region = st.selectbox(
                "Select Region",
                ["Mumbai, India", "Chennai, India", "Dhaka, Bangladesh", "Jakarta, Indonesia", "Custom Location"],
                key="region_predict"
            )
            
            # Set coordinates based on selected region
            if region == "Mumbai, India":
                default_lat, default_lon = 19.0760, 72.8777
            elif region == "Chennai, India":
                default_lat, default_lon = 13.0827, 80.2707
            elif region == "Dhaka, Bangladesh":
                default_lat, default_lon = 23.8103, 90.4125
            elif region == "Jakarta, Indonesia":
                default_lat, default_lon = -6.2088, 106.8456
            else:
                default_lat, default_lon = DEFAULT_LATITUDE, DEFAULT_LONGITUDE
            
            # Custom coordinates if selected
            if region == "Custom Location":
                lat = st.number_input("Latitude", value=default_lat)
                lon = st.number_input("Longitude", value=default_lon)
            else:
                lat, lon = default_lat, default_lon
            
            # Rainfall input
            rainfall = st.slider("Rainfall (mm/day)", min_value=0.0, max_value=100.0, value=15.0, step=1.0)
            
            # Advanced options
            with st.expander("Advanced Options"):
                elevation = st.slider("Elevation (m)", min_value=0.0, max_value=500.0, value=50.0, step=10.0)
                flow_acc = st.slider("Flow Accumulation", min_value=0.0, max_value=1.0, value=0.5, step=0.05)
            
            # Predict button
            predict_btn = st.button("Predict Flood Risk")
        
        with col2:
            # Show prediction results
            if predict_btn:
                with st.spinner("Calculating flood risk..."):
                    # Get prediction
                    prediction = predict_flood_risk(lat, lon, rainfall, elevation, flow_acc)
                    
                    # Store in session state
                    st.session_state.prediction = prediction
            
            # Display prediction if available
            if 'prediction' in st.session_state:
                prediction = st.session_state.prediction
                
                # Create a card-like display
                st.write("### Flood Risk Assessment")
                
                # Risk level with color
                risk_level = prediction['risk_level']
                risk_colors = {
                    'low': '#fef0d9',
                    'medium': '#fdcc8a',
                    'high': '#fc8d59',
                    'severe': '#d7301f'
                }
                
                st.markdown(
                    f"""
                    <div style="padding:1.5rem; border-radius:0.5rem; background-color: {risk_colors.get(risk_level, '#f0f0f0')}; margin-bottom:1rem;">
                        <h3 style="margin:0; color: {'#000' if risk_level in ['low', 'medium'] else '#fff'}">
                            {risk_level.capitalize()} Risk
                        </h3>
                        <p style="font-size:1.8rem; margin:0.5rem 0; color: {'#000' if risk_level in ['low', 'medium'] else '#fff'}">
                            {prediction['risk_score']*100:.1f}%
                        </p>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
                
                # Additional details
                st.write("#### Details")
                st.write(f"Location: {lat:.4f}, {lon:.4f}")
                st.write(f"Rainfall: {prediction['rainfall']:.1f} mm/day")
                st.write(f"Elevation: {prediction['elevation']:.1f} m")
                st.write(f"Flow Accumulation: {prediction['flow_acc']:.2f}")
                
                # Recommendations based on risk level
                st.write("#### Recommendations")
                
                if risk_level == 'low':
                    st.write("- Monitor weather updates")
                    st.write("- No immediate action needed")
                elif risk_level == 'medium':
                    st.write("- Prepare emergency supplies")
                    st.write("- Stay informed about weather changes")
                    st.write("- Check evacuation routes")
                elif risk_level == 'high':
                    st.write("- Alert local authorities")
                    st.write("- Prepare for possible evacuation")
                    st.write("- Move valuables to higher ground")
                    st.write("- Set up aid delivery standby")
                else:  # severe
                    st.write("- Immediate evacuation recommended")
                    st.write("- Deploy emergency response teams")
                    st.write("- Activate aid delivery routes")
                    st.write("- Establish emergency shelters") 
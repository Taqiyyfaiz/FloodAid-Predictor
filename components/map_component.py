import streamlit as st
import folium
from streamlit_folium import folium_static
import geopandas as gpd
import pandas as pd
import numpy as np
from shapely.geometry import Point, LineString
import sys
import os

# Add parent directory to path to import utils and config
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.geo_utils import create_grid, generate_elevation_data, calculate_flow_accumulation, calculate_flood_risk, plot_flood_map, plot_routes
from utils.weather_api import get_grid_weather_data
from config.config import DEFAULT_LATITUDE, DEFAULT_LONGITUDE, DEFAULT_ZOOM, FLOOD_COLORS, AID_CENTERS

def create_flood_map(flood_data=None, center_lat=DEFAULT_LATITUDE, center_lon=DEFAULT_LONGITUDE, zoom=DEFAULT_ZOOM):
    """
    Create a Folium map with flood risk data.
    
    Args:
        flood_data (GeoDataFrame, optional): Flood risk data
        center_lat (float): Center latitude for the map
        center_lon (float): Center longitude for the map
        zoom (int): Initial zoom level
        
    Returns:
        folium.Map: Folium map object
    """
    if flood_data is None:
        # Generate synthetic flood data
        grid = create_grid(center_lat, center_lon, 20, spacing=0.01)
        grid = generate_elevation_data(grid)
        grid = calculate_flow_accumulation(grid)
        rainfall_data = get_grid_weather_data(center_lat, center_lon, grid_size=20)
        flood_data = calculate_flood_risk(grid, rainfall_data)
    
    # Create map
    m = plot_flood_map(flood_data, center_lat, center_lon, zoom)
    
    return m

def create_route_map(routes, flood_data=None, center_lat=DEFAULT_LATITUDE, center_lon=DEFAULT_LONGITUDE, zoom=DEFAULT_ZOOM):
    """
    Create a Folium map with route data.
    
    Args:
        routes (list): List of route dictionaries
        flood_data (GeoDataFrame, optional): Flood risk data
        center_lat (float): Center latitude for the map
        center_lon (float): Center longitude for the map
        zoom (int): Initial zoom level
        
    Returns:
        folium.Map: Folium map object
    """
    # Create base map with flood data if available
    if flood_data is not None:
        m = create_flood_map(flood_data, center_lat, center_lon, zoom)
    else:
        m = folium.Map(
            location=[center_lat, center_lon],
            zoom_start=zoom,
            tiles='CartoDB positron'
        )
    
    # Add aid centers from config
    aid_centers_df = pd.DataFrame(AID_CENTERS)
    
    # Add routes to the map
    m = plot_routes(m, routes, aid_centers_df)
    
    return m

def display_map(m):
    """
    Display a Folium map in Streamlit.
    
    Args:
        m (folium.Map): Folium map object
    """
    folium_static(m)

def map_tab():
    """
    Create the Map tab for the Streamlit app.
    """
    st.header("Flood Risk & Aid Delivery Routes")
    
    # Create columns for the map controls
    col1, col2 = st.columns([3, 1])
    
    with col2:
        st.subheader("Map Controls")
        
        # Region selection
        region = st.selectbox(
            "Select Region",
            ["Mumbai, India", "Chennai, India", "Dhaka, Bangladesh", "Jakarta, Indonesia"]
        )
        
        # Map type selection
        map_type = st.radio(
            "Map Display",
            ["Flood Risk", "Delivery Routes", "Combined View"]
        )
        
        # Rainfall intensity slider
        rainfall_intensity = st.slider(
            "Rainfall Intensity (mm)",
            min_value=0,
            max_value=50,
            value=15,
            step=5
        )
        
        # Generate map button
        generate_button = st.button("Generate Map")
    
    with col1:
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
        
        # Generate synthetic data with the specified rainfall intensity
        if generate_button or 'flood_data' not in st.session_state:
            # Create grid
            grid = create_grid(center_lat, center_lon, 20, spacing=0.01)
            grid = generate_elevation_data(grid)
            grid = calculate_flow_accumulation(grid)
            
            # Get rainfall data
            rainfall_data = get_grid_weather_data(center_lat, center_lon, grid_size=20)
            
            # Adjust rainfall by the intensity factor
            intensity_factor = rainfall_intensity / 15  # Normalize to the default of 15mm
            rainfall_data['rainfall'] = rainfall_data['rainfall'] * intensity_factor
            
            # Calculate flood risk
            flood_data = calculate_flood_risk(grid, rainfall_data)
            
            # Store in session state
            st.session_state.flood_data = flood_data
            st.session_state.center_lat = center_lat
            st.session_state.center_lon = center_lon
        
        # Get data from session state
        flood_data = st.session_state.flood_data
        center_lat = st.session_state.center_lat
        center_lon = st.session_state.center_lon
        
        # Display appropriate map based on selection
        if map_type == "Flood Risk":
            m = create_flood_map(flood_data, center_lat, center_lon, DEFAULT_ZOOM)
            display_map(m)
            
            # Show risk statistics
            risk_counts = flood_data['risk_level'].value_counts().to_dict()
            total_points = len(flood_data)
            
            st.subheader("Flood Risk Statistics")
            stat_cols = st.columns(4)
            
            risk_levels = ['low', 'medium', 'high', 'severe']
            for i, risk in enumerate(risk_levels):
                count = risk_counts.get(risk, 0)
                percentage = (count / total_points) * 100 if total_points > 0 else 0
                
                with stat_cols[i]:
                    st.metric(
                        f"{risk.capitalize()} Risk", 
                        f"{count} points", 
                        f"{percentage:.1f}%"
                    )
            
        elif map_type == "Delivery Routes":
            # Generate routes if not already in session state
            if 'routes' not in st.session_state:
                # Import route optimization
                from models.route_optimization import optimize_routes
                
                # Convert aid centers format
                aid_centers = [
                    {"id": i, "name": center["name"], "lat": center["lat"], "lon": center["lon"], 
                    "capacity": center["capacity"]}
                    for i, center in enumerate(AID_CENTERS)
                ]
                
                # Generate affected areas in high risk zones
                high_risk = flood_data[flood_data['risk_level'].isin(['high', 'severe'])]
                
                if len(high_risk) > 0:
                    # Sample up to 10 points in high-risk areas
                    sample_size = min(10, len(high_risk))
                    sampled_points = high_risk.sample(sample_size)
                    
                    affected_areas = []
                    for i, point in enumerate(sampled_points.itertuples()):
                        affected_areas.append({
                            "id": i + 1000,
                            "lat": point.lat,
                            "lon": point.lon,
                            "severity": point.risk_level,
                            "population": np.random.randint(100, 1000)
                        })
                    
                    # Optimize routes
                    solution = optimize_routes(aid_centers, affected_areas, flood_data)
                    
                    if solution['status'] == 'success':
                        st.session_state.routes = solution['detailed_routes']
                    else:
                        st.error("Route optimization failed. Please try again.")
                        st.session_state.routes = []
                else:
                    st.warning("No high-risk areas found. Using default routes.")
                    # Generate some default routes
                    st.session_state.routes = []
            
            # Get routes from session state
            routes = st.session_state.routes
            
            # Create route map
            m = create_route_map(routes, None, center_lat, center_lon, DEFAULT_ZOOM)
            display_map(m)
            
            # Show route statistics
            if routes:
                st.subheader("Aid Delivery Routes")
                
                for i, route in enumerate(routes):
                    with st.expander(f"Route {i+1}: {route['aid_center']['name']} ({len(route['affected_areas'])} affected areas)", expanded=i==0):
                        st.write(f"Total Distance: {route['distance']:.2f} km")
                        st.write(f"Estimated Time: {route['time']:.0f} minutes")
                        st.write(f"Risk Level: {route['risk_level']}")
                        
                        # Show affected areas table
                        if route['affected_areas']:
                            areas_df = pd.DataFrame(route['affected_areas'])
                            if 'severity' in areas_df.columns and 'population' in areas_df.columns:
                                st.write("Affected Areas:")
                                st.dataframe(areas_df[['lat', 'lon', 'severity', 'population']])
            else:
                st.info("No routes have been generated. Please try adjusting the rainfall intensity.")
                
        else:  # Combined View
            # Generate routes if not already in session state
            if 'routes' not in st.session_state:
                # Same logic as in the "Delivery Routes" case
                from models.route_optimization import optimize_routes
                
                aid_centers = [
                    {"id": i, "name": center["name"], "lat": center["lat"], "lon": center["lon"], 
                    "capacity": center["capacity"]}
                    for i, center in enumerate(AID_CENTERS)
                ]
                
                high_risk = flood_data[flood_data['risk_level'].isin(['high', 'severe'])]
                
                if len(high_risk) > 0:
                    sample_size = min(10, len(high_risk))
                    sampled_points = high_risk.sample(sample_size)
                    
                    affected_areas = []
                    for i, point in enumerate(sampled_points.itertuples()):
                        affected_areas.append({
                            "id": i + 1000,
                            "lat": point.lat,
                            "lon": point.lon,
                            "severity": point.risk_level,
                            "population": np.random.randint(100, 1000)
                        })
                    
                    solution = optimize_routes(aid_centers, affected_areas, flood_data)
                    
                    if solution['status'] == 'success':
                        st.session_state.routes = solution['detailed_routes']
                    else:
                        st.error("Route optimization failed. Please try again.")
                        st.session_state.routes = []
                else:
                    st.warning("No high-risk areas found. Using default routes.")
                    st.session_state.routes = []
            
            # Get routes from session state
            routes = st.session_state.routes
            
            # Create combined map
            m = create_route_map(routes, flood_data, center_lat, center_lon, DEFAULT_ZOOM)
            display_map(m)
            
            # Combined statistics
            st.subheader("Flood & Route Summary")
            
            # Create columns for stats
            col1, col2 = st.columns(2)
            
            with col1:
                # Flood stats
                risk_counts = flood_data['risk_level'].value_counts().to_dict()
                total_points = len(flood_data)
                
                st.write("Flood Risk Statistics")
                for risk in ['low', 'medium', 'high', 'severe']:
                    count = risk_counts.get(risk, 0)
                    percentage = (count / total_points) * 100 if total_points > 0 else 0
                    st.write(f"- {risk.capitalize()}: {count} points ({percentage:.1f}%)")
            
            with col2:
                # Route stats
                if routes:
                    total_distance = sum(route['distance'] for route in routes)
                    total_areas = sum(len(route['affected_areas']) for route in routes)
                    
                    st.write("Aid Delivery Statistics")
                    st.write(f"- Total Routes: {len(routes)}")
                    st.write(f"- Total Distance: {total_distance:.2f} km")
                    st.write(f"- Affected Areas Served: {total_areas}")
                else:
                    st.info("No routes have been generated.")
    
    # Clear session state button
    if st.button("Reset Map"):
        if 'flood_data' in st.session_state:
            del st.session_state.flood_data
        if 'routes' in st.session_state:
            del st.session_state.routes 
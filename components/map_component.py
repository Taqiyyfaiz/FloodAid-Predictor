import streamlit as st
import folium
from streamlit_folium import folium_static
import geopandas as gpd
import pandas as pd
import numpy as np
from shapely.geometry import Point, LineString
import sys
import os
import matplotlib.pyplot as plt

# Add parent directory to path to import utils and config
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.geo_utils import create_grid, generate_elevation_data, calculate_flow_accumulation, calculate_flood_risk, plot_flood_map, plot_routes
from utils.weather_api import get_grid_weather_data
from config.config import DEFAULT_LATITUDE, DEFAULT_LONGITUDE, DEFAULT_ZOOM, FLOOD_COLORS, AID_CENTERS, OPENWEATHER_API_KEY

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
            
            # Show info about synthetic data use
            if OPENWEATHER_API_KEY == "your_api_key_here":
                st.info("Using synthetic weather data for demonstration. To use real data, add your OpenWeather API key to the .env file.")
            
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
    
    # Add report generation and download section
    if 'routes' in st.session_state and st.session_state.routes:
        st.markdown("---")
        st.subheader("📊 Detailed Dashboard & Reports")
        
        # Dashboard metrics
        dashboard_cols = st.columns(4)
        
        # Get data
        routes = st.session_state.routes
        total_routes = len(routes)
        total_distance = sum(route['distance'] for route in routes)
        total_areas = sum(len(route['affected_areas']) for route in routes)
        
        # Display metrics in dashboard style
        with dashboard_cols[0]:
            st.metric("Total Routes", f"{total_routes}")
        
        with dashboard_cols[1]:
            st.metric("Total Distance", f"{total_distance:.1f} km")
        
        with dashboard_cols[2]:
            st.metric("Areas Served", f"{total_areas}")
        
        with dashboard_cols[3]:
            # Calculate average risk level
            risk_values = {'low': 1, 'medium': 2, 'high': 3, 'severe': 4}
            risk_levels = [route['risk_level'] for route in routes]
            if risk_levels:
                avg_risk = sum(risk_values.get(risk, 1) for risk in risk_levels) / len(risk_levels)
                risk_text = "Low" if avg_risk < 1.5 else "Medium" if avg_risk < 2.5 else "High" if avg_risk < 3.5 else "Severe"
                st.metric("Avg Risk Level", risk_text)
            else:
                st.metric("Avg Risk Level", "N/A")
        
        # Create expander for detailed reports
        with st.expander("View Detailed Route Analysis", expanded=False):
            # Create a DataFrame for all routes
            route_data = []
            for i, route in enumerate(routes):
                route_data.append({
                    'Route': i+1,
                    'Aid Center': route['aid_center']['name'],
                    'Distance (km)': round(route['distance'], 2),
                    'Time (min)': round(route['time'], 0),
                    'Risk Level': route['risk_level'].capitalize(),
                    'Areas Served': len(route['affected_areas']),
                    'Affected Population': sum(area.get('population', 0) for area in route['affected_areas'])
                })
            
            route_df = pd.DataFrame(route_data)
            
            # Display the route analysis table
            st.dataframe(route_df, use_container_width=True)
            
            # Create a bar chart for route distances
            st.subheader("Route Distances")
            fig, ax = plt.subplots(figsize=(10, 4))
            bars = ax.bar(
                range(1, len(route_data) + 1),
                [r['Distance (km)'] for r in route_data],
                color=[FLOOD_COLORS.get(r['Risk Level'].lower(), '#3388ff') for r in route_data]
            )
            
            # Add labels
            ax.set_xlabel('Route Number')
            ax.set_ylabel('Distance (km)')
            ax.set_xticks(range(1, len(route_data) + 1))
            
            # Add a legend
            from matplotlib.patches import Patch
            legend_elements = [
                Patch(facecolor=FLOOD_COLORS['low'], label='Low Risk'),
                Patch(facecolor=FLOOD_COLORS['medium'], label='Medium Risk'),
                Patch(facecolor=FLOOD_COLORS['high'], label='High Risk'),
                Patch(facecolor=FLOOD_COLORS['severe'], label='Severe Risk')
            ]
            ax.legend(handles=legend_elements, loc='upper right')
            
            st.pyplot(fig)
        
        # Download section
        st.subheader("Download Reports")
        download_cols = st.columns(3)
        
        with download_cols[0]:
            # Generate CSV report
            if st.button("📄 Download CSV Report"):
                # Create a detailed CSV report
                route_details = []
                for i, route in enumerate(routes):
                    for j, area in enumerate(route['affected_areas']):
                        route_details.append({
                            'Route': i+1,
                            'Aid Center': route['aid_center']['name'],
                            'Area ID': area.get('id', f"Area {j+1}"),
                            'Latitude': area['lat'],
                            'Longitude': area['lon'],
                            'Severity': area.get('severity', 'unknown'),
                            'Population': area.get('population', 0),
                            'Distance (km)': route['distance'] / len(route['affected_areas']) if len(route['affected_areas']) > 0 else 0,
                            'Risk Level': route['risk_level']
                        })
                
                detail_df = pd.DataFrame(route_details)
                
                # Convert to CSV
                csv = detail_df.to_csv(index=False)
                
                # Create download link
                st.download_button(
                    label="Download CSV",
                    data=csv,
                    file_name="floodaid_report.csv",
                    mime="text/csv"
                )
        
        with download_cols[1]:
            # Generate Excel report
            if st.button("📊 Generate Excel Report"):
                # Check if we have the necessary libraries
                try:
                    import io
                    from xlsxwriter import Workbook
                    
                    # Create Excel file in memory
                    output = io.BytesIO()
                    workbook = Workbook(output, {'in_memory': True})
                    
                    # Route summary sheet
                    summary_sheet = workbook.add_worksheet("Route Summary")
                    
                    # Add headers
                    headers = ['Route', 'Aid Center', 'Distance (km)', 'Time (min)', 'Risk Level', 'Areas Served', 'Affected Population']
                    for col, header in enumerate(headers):
                        summary_sheet.write(0, col, header)
                    
                    # Add data
                    for row, data in enumerate(route_data):
                        for col, key in enumerate(headers):
                            if key in data:
                                summary_sheet.write(row + 1, col, data[key])
                    
                    # Affected areas sheet
                    areas_sheet = workbook.add_worksheet("Affected Areas")
                    
                    # Add headers for areas
                    area_headers = ['Route', 'Aid Center', 'Area ID', 'Latitude', 'Longitude', 'Severity', 'Population', 'Distance (km)', 'Risk Level']
                    for col, header in enumerate(area_headers):
                        areas_sheet.write(0, col, header)
                    
                    # Add area data
                    row = 1
                    for i, route in enumerate(routes):
                        for j, area in enumerate(route['affected_areas']):
                            areas_sheet.write(row, 0, i+1)
                            areas_sheet.write(row, 1, route['aid_center']['name'])
                            areas_sheet.write(row, 2, area.get('id', f"Area {j+1}"))
                            areas_sheet.write(row, 3, area['lat'])
                            areas_sheet.write(row, 4, area['lon'])
                            areas_sheet.write(row, 5, area.get('severity', 'unknown'))
                            areas_sheet.write(row, 6, area.get('population', 0))
                            areas_sheet.write(row, 7, route['distance'] / len(route['affected_areas']) if len(route['affected_areas']) > 0 else 0)
                            areas_sheet.write(row, 8, route['risk_level'])
                            row += 1
                    
                    # Close the workbook
                    workbook.close()
                    
                    # Seek to the beginning of the stream
                    output.seek(0)
                    
                    # Create download button
                    st.download_button(
                        label="Download Excel",
                        data=output,
                        file_name="floodaid_detailed_report.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                    )
                    
                except ImportError:
                    st.error("XlsxWriter package is required for Excel export. Please install it with 'pip install xlsxwriter'.")
        
        with download_cols[2]:
            # Generate PDF report (placeholder - would require additional libraries)
            if st.button("📑 Generate PDF Report"):
                st.info("PDF report generation requires additional setup. In a production environment, this would generate a PDF with route maps and detailed statistics.")
                
                # Display a markdown report instead
                markdown_report = f"""
                # FloodAid Route Optimization Report
                
                **Date:** {pd.Timestamp.now().strftime("%Y-%m-%d %H:%M")}
                
                ## Summary
                - **Total Routes:** {total_routes}
                - **Total Distance:** {total_distance:.2f} km
                - **Affected Areas Served:** {total_areas}
                
                ## Route Details
                
                """
                
                for i, route in enumerate(routes):
                    markdown_report += f"""
                    ### Route {i+1}: {route['aid_center']['name']}
                    - **Distance:** {route['distance']:.2f} km
                    - **Time:** {route['time']:.0f} minutes
                    - **Risk Level:** {route['risk_level'].capitalize()}
                    - **Affected Areas:** {len(route['affected_areas'])}
                    - **Affected Population:** {sum(area.get('population', 0) for area in route['affected_areas'])}
                    """
                
                st.markdown(markdown_report)
                
                # Offer markdown download
                st.download_button(
                    label="Download Markdown Report",
                    data=markdown_report,
                    file_name="floodaid_report.md",
                    mime="text/markdown"
                )
    
    # Clear session state button
    if st.button("Reset Map"):
        if 'flood_data' in st.session_state:
            del st.session_state.flood_data
        if 'routes' in st.session_state:
            del st.session_state.routes 
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point, LineString, Polygon
import folium
from math import radians, cos, sin, asin, sqrt
import sys
import os

# Add parent directory to path to import config
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.config import FLOOD_COLORS

def haversine(lon1, lat1, lon2, lat2):
    """
    Calculate the great circle distance between two points 
    on the earth (specified in decimal degrees)
    """
    # Convert decimal degrees to radians
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])

    # Haversine formula
    dlon = lon2 - lon1 
    dlat = lat2 - lat1 
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a)) 
    r = 6371  # Radius of earth in kilometers
    return c * r

def create_grid(center_lat, center_lon, grid_size=10, spacing=0.01):
    """
    Create a grid of points around a center location.
    
    Args:
        center_lat (float): Center latitude
        center_lon (float): Center longitude
        grid_size (int): Grid dimensions (grid_size x grid_size)
        spacing (float): Spacing between grid points in degrees
    
    Returns:
        GeoDataFrame: Grid points with geometry
    """
    # Calculate the starting point for the grid
    start_lat = center_lat - (grid_size // 2) * spacing
    start_lon = center_lon - (grid_size // 2) * spacing
    
    grid_points = []
    
    for i in range(grid_size):
        for j in range(grid_size):
            lat = start_lat + i * spacing
            lon = start_lon + j * spacing
            grid_points.append({
                'geometry': Point(lon, lat),
                'grid_id': i * grid_size + j,
                'lat': lat,
                'lon': lon
            })
    
    return gpd.GeoDataFrame(grid_points, crs="EPSG:4326")

def generate_elevation_data(gdf):
    """
    Generate synthetic elevation data for testing.
    
    Args:
        gdf (GeoDataFrame): GeoDataFrame with grid points
    
    Returns:
        GeoDataFrame: Updated GeoDataFrame with elevation data
    """
    # Create a copy to avoid modifying the original
    result = gdf.copy()
    
    # Calculate center point
    center_lat = gdf['lat'].mean()
    center_lon = gdf['lon'].mean()
    
    # Generate synthetic elevation data (higher in the center, lower on edges)
    result['elevation'] = result.apply(
        lambda row: 100 - 50 * sqrt((row['lat'] - center_lat)**2 + (row['lon'] - center_lon)**2) * 100,
        axis=1
    )
    
    return result

def calculate_flow_accumulation(gdf):
    """
    Calculate synthetic flow accumulation based on elevation.
    
    Args:
        gdf (GeoDataFrame): GeoDataFrame with elevation data
    
    Returns:
        GeoDataFrame: Updated GeoDataFrame with flow accumulation data
    """
    # Create a copy to avoid modifying the original
    result = gdf.copy()
    
    # Get min and max elevation for normalization
    min_elev = result['elevation'].min()
    max_elev = result['elevation'].max()
    
    # Calculate normalized elevation
    result['norm_elev'] = (result['elevation'] - min_elev) / (max_elev - min_elev)
    
    # Invert normalized elevation (lower elevation = higher flow accumulation)
    result['flow_acc'] = 1 - result['norm_elev']
    
    # Add some random variations
    result['flow_acc'] = result['flow_acc'] + np.random.normal(0, 0.05, len(result))
    result['flow_acc'] = result['flow_acc'].clip(0, 1)  # Clip between 0 and 1
    
    return result

def calculate_flood_risk(gdf, rainfall_data):
    """
    Calculate flood risk based on elevation, flow accumulation, and rainfall.
    
    Args:
        gdf (GeoDataFrame): GeoDataFrame with elevation and flow accumulation data
        rainfall_data (DataFrame): DataFrame with rainfall data
    
    Returns:
        GeoDataFrame: Updated GeoDataFrame with flood risk data
    """
    # Create a copy to avoid modifying the original
    result = gdf.copy()
    
    # Merge rainfall data with grid points
    rainfall_gdf = gpd.GeoDataFrame(
        rainfall_data,
        geometry=[Point(lon, lat) for lon, lat in zip(rainfall_data['longitude'], rainfall_data['latitude'])],
        crs="EPSG:4326"
    )
    
    # Find nearest rainfall point for each grid point
    result['nearest_rainfall'] = result.apply(
        lambda row: find_nearest_point(row.geometry, rainfall_gdf)['rainfall'],
        axis=1
    )
    
    # Calculate flood risk score
    # 50% weight to flow accumulation, 30% to rainfall, 20% to inverse elevation
    result['flood_risk'] = (
        0.5 * result['flow_acc'] +
        0.3 * (result['nearest_rainfall'] / rainfall_data['rainfall'].max()) +
        0.2 * (1 - result['norm_elev'])
    )
    
    # Normalize flood risk score to 0-1
    max_risk = result['flood_risk'].max()
    min_risk = result['flood_risk'].min()
    result['flood_risk'] = (result['flood_risk'] - min_risk) / (max_risk - min_risk)
    
    # Classify flood risk
    result['risk_level'] = pd.cut(
        result['flood_risk'],
        bins=[0, 0.25, 0.5, 0.75, 1.0],
        labels=['low', 'medium', 'high', 'severe']
    )
    
    return result

def find_nearest_point(point, gdf):
    """Find the nearest point in a GeoDataFrame to the given point."""
    distances = gdf.distance(point)
    nearest_idx = distances.argmin()
    return gdf.iloc[nearest_idx]

def calculate_route(start_lat, start_lon, end_lat, end_lon, flood_gdf=None, waypoints=10):
    """
    Calculate a route between two points, avoiding high flood risk areas if provided.
    For the prototype, this generates a synthetic route.
    
    Args:
        start_lat (float): Start latitude
        start_lon (float): Start longitude
        end_lat (float): End latitude
        end_lon (float): End longitude
        flood_gdf (GeoDataFrame, optional): GeoDataFrame with flood risk data
        waypoints (int): Number of waypoints in the route
    
    Returns:
        LineString: Geometry representing the route
    """
    # Direct line between start and end
    direct_line = LineString([(start_lon, start_lat), (end_lon, end_lat)])
    
    if flood_gdf is None:
        # Without flood data, just return a slightly curved route
        # Create a curved line with some randomness
        coords = []
        for i in range(waypoints):
            t = i / (waypoints - 1)
            # Interpolate between start and end
            lat = start_lat + t * (end_lat - start_lat)
            lon = start_lon + t * (end_lon - start_lon)
            
            # Add some curvature
            if 0 < i < waypoints - 1:
                # Create a slight curve
                perpendicular_x = -(end_lat - start_lat)
                perpendicular_y = end_lon - start_lon
                length = sqrt(perpendicular_x**2 + perpendicular_y**2)
                
                if length > 0:
                    perpendicular_x /= length
                    perpendicular_y /= length
                    
                    # Adjust the magnitude of the curve
                    curve_strength = sin(t * 3.14) * 0.005
                    lon += perpendicular_y * curve_strength
                    lat += perpendicular_x * curve_strength
            
            coords.append((lon, lat))
        
        return LineString(coords)
    
    # With flood data, try to avoid high risk areas
    # For the prototype, this is a simplified approach
    
    # Buffer the direct line to create a corridor for finding the route
    buffer_distance = 0.02  # Degrees
    corridor = direct_line.buffer(buffer_distance)
    
    # Extract flood risk within the corridor
    corridor_gdf = gpd.GeoDataFrame(geometry=[corridor], crs="EPSG:4326")
    risk_in_corridor = gpd.sjoin(flood_gdf, corridor_gdf)
    
    # Create waypoints along the route
    coords = []
    for i in range(waypoints):
        t = i / (waypoints - 1)
        # Base point on the direct line
        base_lat = start_lat + t * (end_lat - start_lat)
        base_lon = start_lon + t * (end_lon - start_lon)
        
        if i == 0:
            # Start point
            lat, lon = start_lat, start_lon
        elif i == waypoints - 1:
            # End point
            lat, lon = end_lat, end_lon
        else:
            # For intermediate points, try to avoid high risk areas
            # Create a point
            point = Point(base_lon, base_lat)
            
            # Find nearby risk points
            distance_threshold = 0.01  # Degrees
            nearby_risk = risk_in_corridor[risk_in_corridor.distance(point) < distance_threshold]
            
            if len(nearby_risk) > 0:
                # Calculate average risk direction to move away from
                high_risk_points = nearby_risk[nearby_risk['risk_level'].isin(['high', 'severe'])]
                
                if len(high_risk_points) > 0:
                    # Calculate vector away from high risk
                    risk_lon = high_risk_points['geometry'].x.mean()
                    risk_lat = high_risk_points['geometry'].y.mean()
                    
                    # Move away from risk
                    vector_lon = base_lon - risk_lon
                    vector_lat = base_lat - risk_lat
                    
                    # Normalize
                    length = sqrt(vector_lon**2 + vector_lat**2)
                    if length > 0:
                        vector_lon /= length
                        vector_lat /= length
                        
                        # Adjust the point location
                        adjustment_strength = 0.005  # Degrees
                        lon = base_lon + vector_lon * adjustment_strength
                        lat = base_lat + vector_lat * adjustment_strength
                    else:
                        lon, lat = base_lon, base_lat
                else:
                    lon, lat = base_lon, base_lat
            else:
                lon, lat = base_lon, base_lat
        
        coords.append((lon, lat))
    
    return LineString(coords)

def plot_flood_map(flood_gdf, center_lat, center_lon, zoom=12):
    """
    Create a Folium map with flood risk visualization.
    
    Args:
        flood_gdf (GeoDataFrame): GeoDataFrame with flood risk data
        center_lat (float): Center latitude for the map
        center_lon (float): Center longitude for the map
        zoom (int): Initial zoom level
    
    Returns:
        folium.Map: Folium map object
    """
    # Create a map centered at the specified location
    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=zoom,
        tiles='CartoDB positron'
    )
    
    # Add flood risk points
    for risk_level in ['low', 'medium', 'high', 'severe']:
        risk_points = flood_gdf[flood_gdf['risk_level'] == risk_level]
        
        for _, row in risk_points.iterrows():
            folium.CircleMarker(
                location=[row.geometry.y, row.geometry.x],
                radius=5,
                color=FLOOD_COLORS[risk_level],
                fill=True,
                fill_color=FLOOD_COLORS[risk_level],
                fill_opacity=0.7,
                tooltip=f"Risk: {risk_level}<br>Score: {row['flood_risk']:.2f}"
            ).add_to(m)
    
    # Add legend
    legend_html = '''
    <div style="position: fixed; bottom: 50px; left: 50px; z-index: 1000; background-color: white; 
    padding: 10px; border: 2px solid grey; border-radius: 5px;">
    <p><strong>Flood Risk Levels</strong></p>
    <p><i class="fa fa-circle" style="color:{}"></i> Low</p>
    <p><i class="fa fa-circle" style="color:{}"></i> Medium</p>
    <p><i class="fa fa-circle" style="color:{}"></i> High</p>
    <p><i class="fa fa-circle" style="color:{}"></i> Severe</p>
    </div>
    '''.format(
        FLOOD_COLORS['low'], 
        FLOOD_COLORS['medium'], 
        FLOOD_COLORS['high'], 
        FLOOD_COLORS['severe']
    )
    
    m.get_root().html.add_child(folium.Element(legend_html))
    
    return m

def plot_routes(m, routes, aid_centers=None):
    """
    Add routes to a Folium map.
    
    Args:
        m (folium.Map): Folium map object
        routes (list): List of route dictionaries with LineString geometries
        aid_centers (GeoDataFrame, optional): GeoDataFrame with aid center locations
    
    Returns:
        folium.Map: Updated Folium map object
    """
    # Add routes
    for i, route in enumerate(routes):
        line = route['geometry']
        risk_level = route.get('risk_level', 'low')
        
        # Get color based on risk level
        color = FLOOD_COLORS.get(risk_level, '#3388ff')
        
        # Plot the route
        folium.GeoJson(
            line,
            style_function=lambda x, color=color: {
                'color': color,
                'weight': 4,
                'opacity': 0.8
            },
            tooltip=f"Route {i+1}: {route.get('distance', 0):.2f} km"
        ).add_to(m)
    
    # Add aid centers
    if aid_centers is not None:
        for _, center in aid_centers.iterrows():
            folium.Marker(
                location=[center['latitude'], center['longitude']],
                tooltip=f"Aid Center: {center['name']}<br>Capacity: {center['capacity']}",
                icon=folium.Icon(color='green', icon='hospital', prefix='fa')
            ).add_to(m)
    
    return m 
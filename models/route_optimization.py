import numpy as np
import pandas as pd
from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp
import sys
import os
import geopandas as gpd
from shapely.geometry import Point

# Add parent directory to path to import utils
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.geo_utils import haversine, calculate_route
from config.config import AID_CENTERS, MAX_ROUTE_TIME, VEHICLE_SPEED

class RouteOptimizer:
    """Route optimization for aid delivery using Google OR-Tools."""
    
    def __init__(self):
        """Initialize the route optimizer."""
        self.manager = None
        self.routing = None
        self.solution = None
        self.distance_matrix = None
        self.risk_matrix = None
        self.aid_centers = None
        self.affected_areas = None
    
    def create_distance_matrix(self, locations):
        """
        Create a distance matrix between all locations.
        
        Args:
            locations (list): List of dictionaries with lat/lon coordinates
            
        Returns:
            np.array: Distance matrix in kilometers
        """
        num_locations = len(locations)
        matrix = np.zeros((num_locations, num_locations))
        
        for from_idx in range(num_locations):
            for to_idx in range(num_locations):
                if from_idx == to_idx:
                    matrix[from_idx, to_idx] = 0  # Same location
                else:
                    from_loc = locations[from_idx]
                    to_loc = locations[to_idx]
                    
                    # Calculate haversine distance
                    distance = haversine(
                        from_loc['lon'], from_loc['lat'],
                        to_loc['lon'], to_loc['lat']
                    )
                    
                    matrix[from_idx, to_idx] = distance
        
        return matrix
    
    def create_risk_matrix(self, locations, flood_gdf=None):
        """
        Create a risk matrix between locations based on flood risk.
        
        Args:
            locations (list): List of dictionaries with lat/lon coordinates
            flood_gdf (GeoDataFrame, optional): GeoDataFrame with flood risk data
            
        Returns:
            np.array: Risk matrix (higher values = higher risk)
        """
        num_locations = len(locations)
        matrix = np.ones((num_locations, num_locations))  # Default risk = 1
        
        if flood_gdf is None:
            return matrix  # No flood data, return default risk
        
        # Risk levels to numerical values
        risk_values = {
            'low': 1.0,
            'medium': 2.0,
            'high': 5.0,
            'severe': 10.0
        }
        
        for from_idx in range(num_locations):
            for to_idx in range(num_locations):
                if from_idx == to_idx:
                    matrix[from_idx, to_idx] = 0  # Same location, no risk
                else:
                    from_loc = locations[from_idx]
                    to_loc = locations[to_idx]
                    
                    # Create a route between the locations
                    route = calculate_route(
                        from_loc['lat'], from_loc['lon'],
                        to_loc['lat'], to_loc['lon'],
                        flood_gdf
                    )
                    
                    # Find the highest risk level along the route
                    route_gdf = gpd.GeoDataFrame(geometry=[route], crs="EPSG:4326")
                    route_buffer = route.buffer(0.002)  # Buffer around route (about 200m)
                    route_buffer_gdf = gpd.GeoDataFrame(geometry=[route_buffer], crs="EPSG:4326")
                    
                    # Spatial join to find risk points along the route
                    risk_along_route = gpd.sjoin(flood_gdf, route_buffer_gdf)
                    
                    if len(risk_along_route) > 0:
                        # Get the highest risk level along the route
                        worst_risk = risk_along_route['risk_level'].max()
                        risk_factor = risk_values.get(worst_risk, 1.0)
                    else:
                        risk_factor = 1.0  # Default risk
                    
                    matrix[from_idx, to_idx] = risk_factor
        
        return matrix
    
    def prepare_data(self, aid_centers, affected_areas, flood_gdf=None):
        """
        Prepare data for route optimization.
        
        Args:
            aid_centers (list): List of aid centers with lat/lon coordinates
            affected_areas (list): List of affected areas with lat/lon coordinates
            flood_gdf (GeoDataFrame, optional): GeoDataFrame with flood risk data
            
        Returns:
            dict: Data model for OR-Tools
        """
        self.aid_centers = aid_centers
        self.affected_areas = affected_areas
        
        # Combine aid centers and affected areas
        all_locations = aid_centers + affected_areas
        
        # Create distance matrix
        self.distance_matrix = self.create_distance_matrix(all_locations)
        
        # Create risk matrix
        self.risk_matrix = self.create_risk_matrix(all_locations, flood_gdf)
        
        # Adjust distances based on risk (higher risk = longer effective distance)
        risk_adjusted_matrix = self.distance_matrix * self.risk_matrix
        
        # Convert to integer for OR-Tools (multiply by 100 to preserve 2 decimal places)
        distance_matrix_int = (risk_adjusted_matrix * 100).astype(int)
        
        # Prepare data for OR-Tools
        data = {}
        data['distance_matrix'] = distance_matrix_int.tolist()
        data['num_vehicles'] = len(aid_centers)
        data['depot'] = 0  # Start from the first aid center
        data['demands'] = [0] * len(aid_centers) + [1] * len(affected_areas)  # Each affected area has demand of 1
        data['vehicle_capacities'] = [center.get('capacity', 10) for center in aid_centers]
        
        # Time windows (in minutes)
        data['time_matrix'] = (self.distance_matrix / VEHICLE_SPEED * 60).astype(int).tolist()  # Convert to minutes
        data['time_windows'] = [(0, MAX_ROUTE_TIME)] * len(all_locations)  # All locations have same time window
        
        return data
    
    def solve(self, aid_centers=None, affected_areas=None, flood_gdf=None, use_cached=False):
        """
        Solve the route optimization problem.
        
        Args:
            aid_centers (list, optional): List of aid centers
            affected_areas (list, optional): List of affected areas
            flood_gdf (GeoDataFrame, optional): GeoDataFrame with flood risk data
            use_cached (bool): Whether to use cached data
            
        Returns:
            dict: Solution with routes and metrics
        """
        # Use default aid centers if not provided
        if aid_centers is None and not use_cached:
            aid_centers = [
                {"id": i, "name": center["name"], "lat": center["lat"], "lon": center["lon"], 
                "capacity": center["capacity"]}
                for i, center in enumerate(AID_CENTERS)
            ]
        
        # If no affected areas are provided, create synthetic ones
        if affected_areas is None and not use_cached:
            # Create some synthetic affected areas around Mumbai
            import random
            affected_areas = []
            
            # Base coordinates (Mumbai)
            base_lat, base_lon = 19.0760, 72.8777
            
            # Generate 10 random affected areas
            for i in range(10):
                # Random offsets (within ~10km)
                lat_offset = random.uniform(-0.1, 0.1)
                lon_offset = random.uniform(-0.1, 0.1)
                
                affected_areas.append({
                    "id": i + 1000,  # Starting from 1000 to differentiate from aid centers
                    "lat": base_lat + lat_offset,
                    "lon": base_lon + lon_offset,
                    "severity": random.choice(["low", "medium", "high", "severe"]),
                    "population": random.randint(100, 1000)
                })
        
        # Skip data preparation if using cached data
        if not use_cached:
            data = self.prepare_data(aid_centers, affected_areas, flood_gdf)
        else:
            if self.distance_matrix is None:
                raise ValueError("No cached data available. Set use_cached=False.")
            data = {
                'distance_matrix': (self.distance_matrix * 100).astype(int).tolist(),
                'num_vehicles': len(self.aid_centers),
                'depot': 0,
                'demands': [0] * len(self.aid_centers) + [1] * len(self.affected_areas),
                'vehicle_capacities': [center.get('capacity', 10) for center in self.aid_centers],
                'time_matrix': (self.distance_matrix / VEHICLE_SPEED * 60).astype(int).tolist(),
                'time_windows': [(0, MAX_ROUTE_TIME)] * (len(self.aid_centers) + len(self.affected_areas))
            }
        
        # Create routing index manager
        self.manager = pywrapcp.RoutingIndexManager(
            len(data['distance_matrix']),
            data['num_vehicles'],
            data['depot']
        )
        
        # Create routing model
        self.routing = pywrapcp.RoutingModel(self.manager)
        
        # Define callbacks for distance and time
        def distance_callback(from_index, to_index):
            from_node = self.manager.IndexToNode(from_index)
            to_node = self.manager.IndexToNode(to_index)
            return data['distance_matrix'][from_node][to_node]
        
        def time_callback(from_index, to_index):
            from_node = self.manager.IndexToNode(from_index)
            to_node = self.manager.IndexToNode(to_index)
            return data['time_matrix'][from_node][to_node]
        
        # Register callbacks
        transit_callback_index = self.routing.RegisterTransitCallback(distance_callback)
        time_callback_index = self.routing.RegisterTransitCallback(time_callback)
        
        # Set arc costs (distances)
        self.routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)
        
        # Add capacity constraints
        def demand_callback(from_index):
            from_node = self.manager.IndexToNode(from_index)
            return data['demands'][from_node]
        
        demand_callback_index = self.routing.RegisterUnaryTransitCallback(demand_callback)
        self.routing.AddDimensionWithVehicleCapacity(
            demand_callback_index,
            0,  # null capacity slack
            data['vehicle_capacities'],  # vehicle capacities
            True,  # start cumul to zero
            'Capacity'
        )
        
        # Add time window constraints
        self.routing.AddDimension(
            time_callback_index,
            30,  # Allow 30 minutes slack
            MAX_ROUTE_TIME,  # Maximum time per vehicle
            False,  # Don't force start cumul to zero
            'Time'
        )
        
        time_dimension = self.routing.GetDimensionOrDie('Time')
        
        # Add time window constraints for each location
        for location_idx, time_window in enumerate(data['time_windows']):
            if location_idx == data['depot']:
                continue  # Skip depot
            index = self.manager.NodeToIndex(location_idx)
            time_dimension.CumulVar(index).SetRange(time_window[0], time_window[1])
        
        # Set objective: minimize total distance and time
        # The following line has been removed because it's redundant and not supported in newer versions
        # self.routing.SetPrimaryArcCostEvaluatorOfAllVehicles(transit_callback_index)
        
        # Set solution parameters
        search_parameters = pywrapcp.DefaultRoutingSearchParameters()
        search_parameters.first_solution_strategy = (
            routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
        )
        search_parameters.local_search_metaheuristic = (
            routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
        )
        search_parameters.time_limit.seconds = 10  # 10 seconds time limit for the search
        
        # Solve the problem
        self.solution = self.routing.SolveWithParameters(search_parameters)
        
        # Process the solution
        if self.solution:
            # Extract routes
            routes = []
            
            for vehicle_id in range(data['num_vehicles']):
                if use_cached:
                    aid_center = self.aid_centers[vehicle_id]
                else:
                    aid_center = aid_centers[vehicle_id]
                
                index = self.routing.Start(vehicle_id)
                route = {
                    'aid_center': aid_center,
                    'affected_areas': [],
                    'distance': 0,
                    'time': 0,
                    'path': [],
                    'geometry': None
                }
                
                while not self.routing.IsEnd(index):
                    node_index = self.manager.IndexToNode(index)
                    route['path'].append(node_index)
                    
                    # Add affected area if this is not an aid center
                    if node_index >= len(self.aid_centers if use_cached else aid_centers):
                        area_idx = node_index - len(self.aid_centers if use_cached else aid_centers)
                        route['affected_areas'].append(
                            self.affected_areas[area_idx] if use_cached else affected_areas[area_idx]
                        )
                    
                    previous_index = index
                    index = self.solution.Value(self.routing.NextVar(index))
                    
                    # Add distance and time for this leg
                    if not self.routing.IsEnd(index):
                        next_node_index = self.manager.IndexToNode(index)
                        route['distance'] += self.distance_matrix[node_index][next_node_index]
                        route['time'] += data['time_matrix'][node_index][next_node_index]
                
                # Add the last node (return to depot)
                node_index = self.manager.IndexToNode(index)
                route['path'].append(node_index)
                
                # Add geometric route if we have affected areas
                if route['affected_areas']:
                    # Create a complete geometric route from the aid center to all affected areas
                    coords = []
                    coords.append((aid_center['lon'], aid_center['lat']))
                    
                    for area in route['affected_areas']:
                        coords.append((area['lon'], area['lat']))
                    
                    # Return to aid center
                    coords.append((aid_center['lon'], aid_center['lat']))
                    
                    from shapely.geometry import LineString
                    route['geometry'] = LineString(coords)
                
                routes.append(route)
            
            # Calculate total distance across all routes
            total_distance = sum(route['distance'] for route in routes)
            
            return {
                'status': 'success',
                'routes': routes,
                'total_distance': total_distance,  # Just use sum of all route distances
                'affected_areas_served': sum(len(route['affected_areas']) for route in routes)
            }
        else:
            return {
                'status': 'failed',
                'message': 'No solution found'
            }
    
    def get_detailed_routes(self, flood_gdf=None):
        """
        Generate detailed routes with flood risk avoidance.
        
        Args:
            flood_gdf (GeoDataFrame, optional): GeoDataFrame with flood risk data
            
        Returns:
            list: Detailed routes with geometries
        """
        if self.solution is None:
            raise ValueError("No solution available. Call solve() first.")
        
        # Get solution routes
        solution = self.get_solution()
        
        if solution['status'] == 'failed':
            return []
        
        detailed_routes = []
        
        for route in solution['routes']:
            # Skip empty routes
            if not route['affected_areas']:
                continue
            
            aid_center = route['aid_center']
            affected_areas = route['affected_areas']
            
            # Create detailed route segments
            segments = []
            
            # From aid center to first affected area
            if affected_areas:
                start_area = affected_areas[0]
                route_geom = calculate_route(
                    aid_center['lat'], aid_center['lon'],
                    start_area['lat'], start_area['lon'],
                    flood_gdf
                )
                
                # Calculate route distance and risk level
                from utils.geo_utils import haversine
                distance = haversine(
                    aid_center['lon'], aid_center['lat'],
                    start_area['lon'], start_area['lat']
                )
                
                risk_level = self.get_route_risk_level(route_geom, flood_gdf)
                
                segments.append({
                    'geometry': route_geom,
                    'distance': distance,
                    'risk_level': risk_level,
                    'from': aid_center,
                    'to': start_area
                })
            
            # Between affected areas
            for i in range(len(affected_areas) - 1):
                from_area = affected_areas[i]
                to_area = affected_areas[i + 1]
                
                route_geom = calculate_route(
                    from_area['lat'], from_area['lon'],
                    to_area['lat'], to_area['lon'],
                    flood_gdf
                )
                
                distance = haversine(
                    from_area['lon'], from_area['lat'],
                    to_area['lon'], to_area['lat']
                )
                
                risk_level = self.get_route_risk_level(route_geom, flood_gdf)
                
                segments.append({
                    'geometry': route_geom,
                    'distance': distance,
                    'risk_level': risk_level,
                    'from': from_area,
                    'to': to_area
                })
            
            # From last affected area back to aid center
            if affected_areas:
                end_area = affected_areas[-1]
                route_geom = calculate_route(
                    end_area['lat'], end_area['lon'],
                    aid_center['lat'], aid_center['lon'],
                    flood_gdf
                )
                
                distance = haversine(
                    end_area['lon'], end_area['lat'],
                    aid_center['lon'], aid_center['lat']
                )
                
                risk_level = self.get_route_risk_level(route_geom, flood_gdf)
                
                segments.append({
                    'geometry': route_geom,
                    'distance': distance,
                    'risk_level': risk_level,
                    'from': end_area,
                    'to': aid_center
                })
            
            # Combine segments into a detailed route
            total_distance = sum(segment['distance'] for segment in segments)
            risk_levels = [segment['risk_level'] for segment in segments if segment['risk_level'] is not None]
            worst_risk = max(risk_levels) if risk_levels else 'low'
            
            detailed_route = {
                'aid_center': aid_center,
                'affected_areas': affected_areas,
                'segments': segments,
                'distance': total_distance,
                'risk_level': worst_risk,
                'time': total_distance / VEHICLE_SPEED * 60  # Convert to minutes
            }
            
            detailed_routes.append(detailed_route)
        
        return detailed_routes
    
    def get_route_risk_level(self, route_geom, flood_gdf):
        """
        Get the highest risk level along a route.
        
        Args:
            route_geom (LineString): Route geometry
            flood_gdf (GeoDataFrame): GeoDataFrame with flood risk data
            
        Returns:
            str: Risk level
        """
        if flood_gdf is None:
            return 'low'  # Default risk level
        
        # Create a buffer around the route
        route_buffer = route_geom.buffer(0.002)  # About 200m buffer
        
        # Create a GeoDataFrame for the buffer
        buffer_gdf = gpd.GeoDataFrame(geometry=[route_buffer], crs="EPSG:4326")
        
        # Spatial join to find risk points along the route
        risk_points = gpd.sjoin(flood_gdf, buffer_gdf)
        
        if len(risk_points) == 0:
            return 'low'  # No risk points found
        
        # Get the highest risk level
        risk_counts = risk_points['risk_level'].value_counts()
        
        # Convert to list of risk levels by frequency
        risk_levels = risk_counts.index.tolist()
        
        # Risk priority order
        priority = {'severe': 3, 'high': 2, 'medium': 1, 'low': 0}
        
        # Sort by priority
        risk_levels.sort(key=lambda x: priority.get(x, 0), reverse=True)
        
        return risk_levels[0] if risk_levels else 'low'
    
    def get_solution(self):
        """
        Get the current solution.
        
        Returns:
            dict: Solution with routes and metrics
        """
        if self.solution is None:
            return {'status': 'not_solved'}
        
        # Rebuild the solution using the cached data
        return self.solve(use_cached=True)

def optimize_routes(aid_centers=None, affected_areas=None, flood_gdf=None):
    """
    Optimize routes for aid delivery.
    
    Args:
        aid_centers (list, optional): List of aid centers
        affected_areas (list, optional): List of affected areas
        flood_gdf (GeoDataFrame, optional): GeoDataFrame with flood risk data
        
    Returns:
        dict: Solution with routes and metrics
    """
    optimizer = RouteOptimizer()
    solution = optimizer.solve(aid_centers, affected_areas, flood_gdf)
    
    if solution['status'] == 'success':
        detailed_routes = optimizer.get_detailed_routes(flood_gdf)
        solution['detailed_routes'] = detailed_routes
    
    return solution

if __name__ == "__main__":
    # Test route optimization
    solution = optimize_routes()
    print(f"Solution found: {solution['status']}")
    print(f"Total distance: {solution['total_distance']:.2f} km")
    print(f"Affected areas served: {solution['affected_areas_served']}")
    
    for i, route in enumerate(solution['routes']):
        print(f"\nRoute {i+1}:")
        print(f"  Aid center: {route['aid_center']['name']}")
        print(f"  Distance: {route['distance']:.2f} km")
        print(f"  Time: {route['time']:.0f} minutes")
        print(f"  Affected areas: {len(route['affected_areas'])}")
        
        for j, area in enumerate(route['affected_areas']):
            print(f"    Area {j+1}: Severity={area['severity']}, Population={area['population']}") 
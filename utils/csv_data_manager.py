import os
import pandas as pd
from datetime import datetime
import sys

# Add parent directory to path to import config
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.config import CSV_DATA_PATH

# Define CSV file paths
WEATHER_DATA_CSV = os.path.join(CSV_DATA_PATH, "weather_data.csv")
FLOOD_PREDICTIONS_CSV = os.path.join(CSV_DATA_PATH, "flood_predictions.csv")
AID_CENTERS_CSV = os.path.join(CSV_DATA_PATH, "aid_centers.csv")
ROUTES_CSV = os.path.join(CSV_DATA_PATH, "routes.csv")
HISTORICAL_DATA_CSV = os.path.join(CSV_DATA_PATH, "historical_data.csv")

def ensure_csv_files_exist():
    """Create the necessary CSV files if they don't exist."""
    # Ensure directory exists
    os.makedirs(CSV_DATA_PATH, exist_ok=True)
    
    # Create weather data CSV if it doesn't exist
    if not os.path.exists(WEATHER_DATA_CSV):
        df = pd.DataFrame(columns=[
            'id', 'date', 'latitude', 'longitude', 'rainfall', 
            'temperature', 'humidity', 'timestamp'
        ])
        df.to_csv(WEATHER_DATA_CSV, index=False)
    
    # Create flood predictions CSV if it doesn't exist
    if not os.path.exists(FLOOD_PREDICTIONS_CSV):
        df = pd.DataFrame(columns=[
            'id', 'latitude', 'longitude', 'risk_level', 
            'risk_score', 'prediction_date'
        ])
        df.to_csv(FLOOD_PREDICTIONS_CSV, index=False)
    
    # Create aid centers CSV if it doesn't exist
    if not os.path.exists(AID_CENTERS_CSV):
        df = pd.DataFrame(columns=[
            'id', 'name', 'latitude', 'longitude', 
            'capacity', 'current_stock'
        ])
        df.to_csv(AID_CENTERS_CSV, index=False)
    
    # Create routes CSV if it doesn't exist
    if not os.path.exists(ROUTES_CSV):
        df = pd.DataFrame(columns=[
            'id', 'aid_center_id', 'destination_lat', 'destination_lon',
            'distance', 'estimated_time', 'risk_level', 'created_date'
        ])
        df.to_csv(ROUTES_CSV, index=False)
    
    # Create historical data CSV if it doesn't exist
    if not os.path.exists(HISTORICAL_DATA_CSV):
        df = pd.DataFrame(columns=[
            'id', 'date', 'latitude', 'longitude', 'elevation', 
            'flow_accumulation', 'rainfall', 'flood_observed'
        ])
        df.to_csv(HISTORICAL_DATA_CSV, index=False)

def initialize_data_files():
    """Initialize all data files."""
    ensure_csv_files_exist()
    
    # Check if we have the uploaded dataset files
    uploaded_csv_dir = os.path.join(os.path.dirname(CSV_DATA_PATH), "csv")
    flood_zone_csv = os.path.join(uploaded_csv_dir, "Flood zone.csv")
    flood_history_csv = os.path.join(uploaded_csv_dir, "Flood history.csv")
    aid_data_csv = os.path.join(uploaded_csv_dir, "Aid data.csv")
    
    # Import uploaded datasets to create historical data
    if os.path.exists(flood_zone_csv) and os.path.exists(flood_history_csv):
        create_historical_data_from_uploads(flood_zone_csv, flood_history_csv)

def create_historical_data_from_uploads(flood_zone_csv, flood_history_csv):
    """Create historical data from uploaded datasets for model training."""
    try:
        # Read flood zone and history data
        zones_df = pd.read_csv(flood_zone_csv)
        history_df = pd.read_csv(flood_history_csv)
        
        # Create historical data
        historical_data = []
        
        # For each zone, create historical records
        for _, zone in zones_df.iterrows():
            # Find matching history record
            history = history_df[history_df['Region_Name'] == zone['Neighborhood']]
            
            if not history.empty:
                history = history.iloc[0]
                
                # Create a record for the zone
                record = {
                    'id': len(historical_data) + 1,
                    'date': datetime.now().strftime('%Y-%m-%d'),
                    'latitude': zone.get('Latitude', 0),  # Default to 0 if not present
                    'longitude': zone.get('Longitude', 0),  # Default to 0 if not present
                    'elevation': zone['Elevation_m'],
                    'flow_accumulation': 0,  # Not available in the datasets
                    'rainfall': history['Max_Flood_Depth_Historical_m'] * 100,  # Rough estimate
                    'flood_observed': 1 if zone['Flood_Risk_Level'] in ['High', 'Extreme'] else 0
                }
                
                historical_data.append(record)
        
        # Save to CSV
        if historical_data:
            historical_df = pd.DataFrame(historical_data)
            historical_df.to_csv(HISTORICAL_DATA_CSV, index=False)
            print(f"Created {len(historical_data)} historical data records")
            
    except Exception as e:
        print(f"Error creating historical data: {e}")

def get_next_id(csv_file):
    """Get the next available ID for a new entry."""
    try:
        df = pd.read_csv(csv_file)
        if df.empty:
            return 1
        return df['id'].max() + 1
    except Exception as e:
        print(f"Error getting next ID: {e}")
        return 1

def insert_weather_data(data):
    """Insert weather data into the CSV file."""
    try:
        df = pd.read_csv(WEATHER_DATA_CSV)
        new_row = {
            'id': get_next_id(WEATHER_DATA_CSV),
            'date': data['date'],
            'latitude': data['latitude'],
            'longitude': data['longitude'],
            'rainfall': data['rainfall'],
            'temperature': data['temperature'],
            'humidity': data['humidity'],
            'timestamp': datetime.now().isoformat()
        }
        df = df.append(new_row, ignore_index=True)
        df.to_csv(WEATHER_DATA_CSV, index=False)
    except Exception as e:
        print(f"Error inserting weather data: {e}")

def save_flood_prediction(data):
    """Save a flood prediction to the CSV file."""
    try:
        df = pd.read_csv(FLOOD_PREDICTIONS_CSV)
        new_row = {
            'id': get_next_id(FLOOD_PREDICTIONS_CSV),
            'latitude': data['latitude'],
            'longitude': data['longitude'],
            'risk_level': data['risk_level'],
            'risk_score': data['risk_score'],
            'prediction_date': datetime.now().isoformat()
        }
        df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
        df.to_csv(FLOOD_PREDICTIONS_CSV, index=False)
        return True
    except Exception as e:
        print(f"Error saving flood prediction: {e}")
        return False

def insert_flood_prediction(data):
    """Insert flood prediction into the CSV file."""
    return save_flood_prediction(data)

def insert_aid_center(data):
    """Insert an aid center into the CSV file."""
    try:
        df = pd.read_csv(AID_CENTERS_CSV)
        new_row = {
            'id': get_next_id(AID_CENTERS_CSV),
            'name': data['name'],
            'latitude': data['latitude'],
            'longitude': data['longitude'],
            'capacity': data['capacity'],
            'current_stock': data['current_stock']
        }
        df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
        df.to_csv(AID_CENTERS_CSV, index=False)
    except Exception as e:
        print(f"Error inserting aid center: {e}")

def insert_route(data):
    """Insert a route into the CSV file."""
    try:
        df = pd.read_csv(ROUTES_CSV)
        new_row = {
            'id': get_next_id(ROUTES_CSV),
            'aid_center_id': data['aid_center_id'],
            'destination_lat': data['destination_lat'],
            'destination_lon': data['destination_lon'],
            'distance': data['distance'],
            'estimated_time': data['estimated_time'],
            'risk_level': data['risk_level'],
            'created_date': datetime.now().isoformat()
        }
        df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
        df.to_csv(ROUTES_CSV, index=False)
    except Exception as e:
        print(f"Error inserting route: {e}")

def get_weather_data():
    """Get all weather data from the CSV file."""
    try:
        return pd.read_csv(WEATHER_DATA_CSV)
    except Exception as e:
        print(f"Error getting weather data: {e}")
        return pd.DataFrame()

def get_flood_predictions():
    """Get all flood predictions from the CSV file."""
    try:
        return pd.read_csv(FLOOD_PREDICTIONS_CSV)
    except Exception as e:
        print(f"Error getting flood predictions: {e}")
        return pd.DataFrame()

def get_aid_centers():
    """Get all aid centers from the CSV file."""
    try:
        return pd.read_csv(AID_CENTERS_CSV)
    except Exception as e:
        print(f"Error getting aid centers: {e}")
        return pd.DataFrame()

def get_routes():
    """Get all routes from the CSV file and join with aid center names."""
    try:
        routes_df = pd.read_csv(ROUTES_CSV)
        aid_centers_df = pd.read_csv(AID_CENTERS_CSV)
        
        # If either DataFrame is empty, return empty DataFrame or just routes
        if routes_df.empty:
            return pd.DataFrame()
        if aid_centers_df.empty:
            return routes_df
            
        # Join routes with aid centers to get the aid center name
        merged_df = pd.merge(
            routes_df, 
            aid_centers_df[['id', 'name']],
            left_on='aid_center_id', 
            right_on='id', 
            suffixes=('', '_aid_center')
        )
        merged_df.rename(columns={'name': 'aid_center_name'}, inplace=True)
        
        return merged_df
    except Exception as e:
        print(f"Error getting routes: {e}")
        return pd.DataFrame()

def get_historical_data():
    """Get historical data for model training."""
    try:
        # Check if historical data file exists
        if not os.path.exists(HISTORICAL_DATA_CSV):
            ensure_csv_files_exist()
            
        # Read the historical data
        historical_df = pd.read_csv(HISTORICAL_DATA_CSV)
        
        # If we have no historical data but have uploaded datasets, try to create historical data
        if historical_df.empty:
            initialize_data_files()
            if os.path.exists(HISTORICAL_DATA_CSV):
                historical_df = pd.read_csv(HISTORICAL_DATA_CSV)
            
        return historical_df
    except Exception as e:
        print(f"Error getting historical data: {e}")
        return pd.DataFrame()

# Initialize the CSV files
if __name__ == "__main__":
    ensure_csv_files_exist() 
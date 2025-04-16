import requests
import pandas as pd
from datetime import datetime, timedelta
import sys
import os

# Add parent directory to path to import config
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.config import OPENWEATHER_API_KEY

def get_current_weather(lat, lon):
    """Get current weather data for a specific location."""
    base_url = "https://api.openweathermap.org/data/2.5/weather"
    
    params = {
        "lat": lat,
        "lon": lon,
        "appid": OPENWEATHER_API_KEY,
        "units": "metric"  # Use metric units
    }
    
    try:
        response = requests.get(base_url, params=params)
        response.raise_for_status()  # Raise an exception for 4XX/5XX responses
        
        data = response.json()
        
        # Extract relevant information
        weather_data = {
            "date": datetime.now().strftime("%Y-%m-%d"),
            "latitude": lat,
            "longitude": lon,
            "rainfall": data.get("rain", {}).get("1h", 0) if "rain" in data else 0,  # mm of rain in the last hour
            "temperature": data["main"]["temp"],  # Celsius
            "humidity": data["main"]["humidity"],  # Percentage
        }
        
        return weather_data
    
    except requests.exceptions.RequestException as e:
        print(f"Error fetching weather data: {e}")
        # Return default data for demo purposes
        return {
            "date": datetime.now().strftime("%Y-%m-%d"),
            "latitude": lat,
            "longitude": lon,
            "rainfall": 0,
            "temperature": 25,
            "humidity": 80,
        }

def get_forecast(lat, lon, days=5):
    """Get weather forecast for a specific location."""
    base_url = "https://api.openweathermap.org/data/2.5/forecast"
    
    params = {
        "lat": lat,
        "lon": lon,
        "appid": OPENWEATHER_API_KEY,
        "units": "metric"  # Use metric units
    }
    
    try:
        response = requests.get(base_url, params=params)
        response.raise_for_status()
        
        data = response.json()
        
        # Extract forecast data
        forecast_data = []
        for item in data["list"]:
            forecast_time = datetime.fromtimestamp(item["dt"])
            
            # Only consider forecasts up to the specified number of days
            if forecast_time <= datetime.now() + timedelta(days=days):
                forecast_data.append({
                    "date": forecast_time.strftime("%Y-%m-%d"),
                    "time": forecast_time.strftime("%H:%M:%S"),
                    "latitude": lat,
                    "longitude": lon,
                    "rainfall": item.get("rain", {}).get("3h", 0) if "rain" in item else 0,  # mm of rain in 3h
                    "temperature": item["main"]["temp"],
                    "humidity": item["main"]["humidity"],
                })
        
        # Convert to DataFrame for easier analysis
        df = pd.DataFrame(forecast_data)
        return df
    
    except requests.exceptions.RequestException as e:
        print(f"Error fetching weather forecast: {e}")
        
        # Create dummy forecast data for demo purposes
        dummy_data = []
        for i in range(days * 8):  # 8 forecasts per day (every 3 hours)
            forecast_time = datetime.now() + timedelta(hours=i*3)
            dummy_data.append({
                "date": forecast_time.strftime("%Y-%m-%d"),
                "time": forecast_time.strftime("%H:%M:%S"),
                "latitude": lat,
                "longitude": lon,
                "rainfall": max(0, min(15, i % 24 / 2)),  # Simulated rainfall
                "temperature": 25 + 5 * (i % 24 / 12) * (1 if i % 2 == 0 else -1),  # Simulated temperature
                "humidity": 70 + 10 * (i % 24 / 12) * (1 if i % 2 == 0 else -1),  # Simulated humidity
            })
        
        return pd.DataFrame(dummy_data)

def get_historical_rainfall(lat, lon):
    """
    For the prototype, we'll create synthetic historical rainfall data 
    since historical data access may require premium API access.
    """
    # Create synthetic historical data for the last 30 days
    historical_data = []
    
    for i in range(30, 0, -1):
        date = datetime.now() - timedelta(days=i)
        
        # Synthetic rainfall pattern - higher in middle of the month
        day_of_month = date.day
        synthetic_rainfall = max(0, min(50, (15 - abs(day_of_month - 15)) * 1.2))
        
        historical_data.append({
            "date": date.strftime("%Y-%m-%d"),
            "latitude": lat,
            "longitude": lon,
            "rainfall": synthetic_rainfall,
            "temperature": 25 + 5 * (1 if day_of_month % 2 == 0 else -1),
            "humidity": 70 + 10 * (1 if day_of_month % 2 == 0 else -1),
        })
    
    return pd.DataFrame(historical_data)

def get_grid_weather_data(center_lat, center_lon, grid_size=5, spacing=0.02):
    """
    Generate weather data for a grid of points around a center location.
    
    Args:
        center_lat (float): Center latitude
        center_lon (float): Center longitude
        grid_size (int): Grid dimensions (grid_size x grid_size)
        spacing (float): Spacing between grid points in degrees
    
    Returns:
        DataFrame: Weather data for the grid points
    """
    grid_data = []
    
    # Calculate the starting point for the grid
    start_lat = center_lat - (grid_size // 2) * spacing
    start_lon = center_lon - (grid_size // 2) * spacing
    
    for i in range(grid_size):
        for j in range(grid_size):
            lat = start_lat + i * spacing
            lon = start_lon + j * spacing
            
            # For prototype, we'll generate synthetic variations of weather
            # based on distance from center
            dist_factor = ((lat - center_lat)**2 + (lon - center_lon)**2)**0.5 / spacing
            
            # Get actual weather for the center, then adjust for other points
            if i == grid_size // 2 and j == grid_size // 2:
                weather = get_current_weather(lat, lon)
            else:
                # For demo, simulate weather variations based on distance from center
                weather = {
                    "date": datetime.now().strftime("%Y-%m-%d"),
                    "latitude": lat,
                    "longitude": lon,
                    "rainfall": max(0, 5 + 2 * dist_factor + (i * j) % 5),  # Simulated rainfall pattern
                    "temperature": 25 - dist_factor * 0.5,  # Lower temperature away from center
                    "humidity": min(95, 70 + dist_factor * 2),  # Higher humidity away from center
                }
            
            grid_data.append(weather)
    
    return pd.DataFrame(grid_data)

if __name__ == "__main__":
    # Test the functions
    weather = get_current_weather(19.0760, 72.8777)
    print("Current Weather:", weather)
    
    forecast = get_forecast(19.0760, 72.8777, 2)
    print("\nForecast:")
    print(forecast.head()) 
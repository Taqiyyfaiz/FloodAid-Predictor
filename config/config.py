import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# API keys
OPENWEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY", "your_api_key_here")

# If DEMO_MODE is set in .env, use synthetic data
if OPENWEATHER_API_KEY == "DEMO_MODE":
    OPENWEATHER_API_KEY = "your_api_key_here"
    print("Using synthetic weather data (DEMO_MODE)")

# Database settings
DATABASE_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "floodaid.db")

# Model settings
MODEL_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models", "flood_prediction_model.pkl")
ROUTE_MODEL_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models", "route_optimization_model.pkl")

# Default region for demo (Mumbai, India)
DEFAULT_LATITUDE = 19.0760
DEFAULT_LONGITUDE = 72.8777
DEFAULT_ZOOM = 10

# Aid centers (sample locations)
AID_CENTERS = [
    {"name": "Mumbai Central", "lat": 18.9750, "lon": 72.8258, "capacity": 100},
    {"name": "Thane Center", "lat": 19.2183, "lon": 72.9781, "capacity": 75},
    {"name": "Navi Mumbai Hub", "lat": 19.0330, "lon": 73.0297, "capacity": 90}
]

# Visualization settings
FLOOD_COLORS = {
    "low": "#fef0d9",
    "medium": "#fdcc8a",
    "high": "#fc8d59",
    "severe": "#d7301f"
}

# Route optimization parameters
MAX_ROUTE_TIME = 180  # minutes
VEHICLE_SPEED = 30    # km/h 
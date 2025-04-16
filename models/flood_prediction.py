import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import sys
import os
import random

# Add parent directory to path to import utils
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.weather_api import get_grid_weather_data, get_historical_rainfall
from utils.geo_utils import create_grid, generate_elevation_data, calculate_flow_accumulation, calculate_flood_risk
from config.config import MODEL_PATH, DEFAULT_LATITUDE, DEFAULT_LONGITUDE

class FloodPredictionModel:
    """Flood prediction model using Random Forest regression."""
    
    def __init__(self):
        """Initialize the model."""
        self.model = None
        self.scaler = StandardScaler()
        self.feature_cols = ['elevation', 'flow_acc', 'nearest_rainfall']
    
    def generate_training_data(self, center_lat=DEFAULT_LATITUDE, center_lon=DEFAULT_LONGITUDE, 
                              grid_size=20, historical_days=30):
        """
        Generate synthetic training data for the model.
        
        Args:
            center_lat (float): Center latitude
            center_lon (float): Center longitude
            grid_size (int): Grid size for spatial discretization
            historical_days (int): Number of historical days to simulate
            
        Returns:
            tuple: X_train, X_test, y_train, y_test
        """
        # Create a spatial grid
        grid = create_grid(center_lat, center_lon, grid_size, spacing=0.01)
        
        # Generate elevation data
        grid = generate_elevation_data(grid)
        
        # Calculate flow accumulation
        grid = calculate_flow_accumulation(grid)
        
        # Get current weather data for the grid
        rainfall_data = get_grid_weather_data(center_lat, center_lon, grid_size=grid_size)
        
        # Calculate current flood risk (this will be our target)
        risk_data = calculate_flood_risk(grid, rainfall_data)
        
        # Create a list to store all training examples
        all_data = []
        
        # Use the current data
        for _, row in risk_data.iterrows():
            all_data.append({
                'elevation': row['elevation'],
                'norm_elev': row['norm_elev'],
                'flow_acc': row['flow_acc'],
                'nearest_rainfall': row['nearest_rainfall'],
                'flood_risk': row['flood_risk']
            })
        
        # Simulate historical scenarios with different rainfall patterns
        for day in range(1, historical_days + 1):
            # Create synthetic rainfall data for this historical day
            # More variation for better training
            synthetic_rainfall = rainfall_data.copy()
            
            # Vary the rainfall randomly
            rain_factor = random.uniform(0.2, 3.0)  # Random factor between 0.2 and 3.0
            synthetic_rainfall['rainfall'] = synthetic_rainfall['rainfall'] * rain_factor
            
            # Vary the temperature randomly (can affect flooding)
            temp_change = random.uniform(-5, 5)  # Random change between -5 and +5
            synthetic_rainfall['temperature'] = synthetic_rainfall['temperature'] + temp_change
            
            # Calculate flood risk for this historical scenario
            historical_risk = calculate_flood_risk(grid, synthetic_rainfall)
            
            # Add to training data
            for _, row in historical_risk.iterrows():
                all_data.append({
                    'elevation': row['elevation'],
                    'norm_elev': row['norm_elev'],
                    'flow_acc': row['flow_acc'],
                    'nearest_rainfall': row['nearest_rainfall'],
                    'flood_risk': row['flood_risk']
                })
        
        # Convert to DataFrame
        df = pd.DataFrame(all_data)
        
        # Split data into features and target
        X = df[self.feature_cols]
        y = df['flood_risk']
        
        # Split into training and test sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Normalize features
        X_train = self.scaler.fit_transform(X_train)
        X_test = self.scaler.transform(X_test)
        
        return X_train, X_test, y_train, y_test
    
    def train(self, X_train, y_train, n_estimators=100, max_depth=10):
        """
        Train the flood prediction model.
        
        Args:
            X_train (array): Training features
            y_train (array): Training targets
            n_estimators (int): Number of trees in the forest
            max_depth (int): Maximum depth of the trees
            
        Returns:
            self: Trained model instance
        """
        self.model = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=42
        )
        
        self.model.fit(X_train, y_train)
        return self
    
    def evaluate(self, X_test, y_test):
        """
        Evaluate the model on test data.
        
        Args:
            X_test (array): Test features
            y_test (array): Test targets
            
        Returns:
            dict: Dictionary with evaluation metrics
        """
        if self.model is None:
            raise ValueError("Model not trained yet.")
        
        y_pred = self.model.predict(X_test)
        
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, y_pred)
        
        return {
            'mse': mse,
            'rmse': rmse,
            'r2': r2
        }
    
    def predict(self, features):
        """
        Make flood risk predictions.
        
        Args:
            features (DataFrame): Features for prediction
            
        Returns:
            array: Predicted flood risk scores
        """
        if self.model is None:
            raise ValueError("Model not trained yet.")
        
        # Ensure the features have the correct columns
        features = features[self.feature_cols]
        
        # Normalize features
        features_scaled = self.scaler.transform(features)
        
        # Make predictions
        return self.model.predict(features_scaled)
    
    def predict_for_grid(self, grid, rainfall_data):
        """
        Make flood risk predictions for a grid.
        
        Args:
            grid (GeoDataFrame): Grid with elevation and flow accumulation data
            rainfall_data (DataFrame): Rainfall data for the grid
            
        Returns:
            GeoDataFrame: Grid with added flood risk predictions
        """
        if self.model is None:
            raise ValueError("Model not trained yet.")
        
        # Create a copy to avoid modifying the original
        result = grid.copy()
        
        # Find nearest rainfall values
        rainfall_gdf = pd.DataFrame(rainfall_data)
        rainfall_points = [Point(lon, lat) for lon, lat in zip(rainfall_data['longitude'], rainfall_data['latitude'])]
        rainfall_gdf['geometry'] = rainfall_points
        rainfall_gdf = gpd.GeoDataFrame(rainfall_gdf, crs="EPSG:4326")
        
        from utils.geo_utils import find_nearest_point
        result['nearest_rainfall'] = result.apply(
            lambda row: find_nearest_point(row.geometry, rainfall_gdf)['rainfall'],
            axis=1
        )
        
        # Extract features for prediction
        features = result[self.feature_cols]
        
        # Predict flood risk
        result['flood_risk'] = self.predict(features)
        
        # Classify risk levels
        result['risk_level'] = pd.cut(
            result['flood_risk'],
            bins=[0, 0.25, 0.5, 0.75, 1.0],
            labels=['low', 'medium', 'high', 'severe']
        )
        
        return result
    
    def save(self, path=MODEL_PATH):
        """
        Save the model to a file.
        
        Args:
            path (str): Path to save the model
        """
        if self.model is None:
            raise ValueError("No trained model to save.")
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        # Save model
        joblib.dump({
            'model': self.model,
            'scaler': self.scaler,
            'feature_cols': self.feature_cols
        }, path)
    
    def load(self, path=MODEL_PATH):
        """
        Load the model from a file.
        
        Args:
            path (str): Path to load the model from
            
        Returns:
            self: Model instance with loaded model
        """
        try:
            data = joblib.load(path)
            self.model = data['model']
            self.scaler = data['scaler']
            self.feature_cols = data['feature_cols']
            return self
        except (FileNotFoundError, KeyError) as e:
            print(f"Error loading model: {e}")
            return None

def train_model(save=True):
    """
    Train and save a new model.
    
    Args:
        save (bool): Whether to save the model
        
    Returns:
        FloodPredictionModel: Trained model instance
    """
    model = FloodPredictionModel()
    
    # Generate training data
    X_train, X_test, y_train, y_test = model.generate_training_data()
    
    # Train the model
    model.train(X_train, y_train)
    
    # Evaluate
    metrics = model.evaluate(X_test, y_test)
    print(f"Model evaluation: MSE={metrics['mse']:.4f}, RMSE={metrics['rmse']:.4f}, R²={metrics['r2']:.4f}")
    
    # Save if requested
    if save:
        model.save()
        print(f"Model saved to {MODEL_PATH}")
    
    return model

def load_or_train_model():
    """
    Load an existing model or train a new one if no saved model exists.
    
    Returns:
        FloodPredictionModel: Model instance
    """
    model = FloodPredictionModel()
    
    # Try to load the model
    loaded_model = model.load()
    
    # If loading failed, train a new model
    if loaded_model is None:
        print("No saved model found, training a new one...")
        return train_model(save=True)
    
    return loaded_model

if __name__ == "__main__":
    # Train and save a model
    train_model() 
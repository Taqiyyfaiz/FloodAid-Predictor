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
from utils.csv_data_manager import get_historical_data, save_flood_prediction, initialize_data_files
from config.config import MODEL_PATH, DEFAULT_LATITUDE, DEFAULT_LONGITUDE

class FloodPredictionModel:
    """Flood prediction model using Random Forest regression."""
    
    def __init__(self):
        """Initialize the model."""
        self.model = None
        self.scaler = StandardScaler()
        self.feature_cols = ['elevation', 'flow_accumulation', 'rainfall']
    
    def generate_training_data(self, center_lat=DEFAULT_LATITUDE, center_lon=DEFAULT_LONGITUDE, 
                              grid_size=20, use_historical=True):
        """
        Generate training data for the model, optionally using historical data from CSV.
        
        Args:
            center_lat (float): Center latitude
            center_lon (float): Center longitude
            grid_size (int): Grid size for spatial discretization
            use_historical (bool): Whether to use historical data from CSV
            
        Returns:
            tuple: X_train, X_test, y_train, y_test
        """
        if use_historical:
            # Try to load historical data from CSV
            try:
                print("Loading historical data from CSV...")
                df = get_historical_data()
                
                if len(df) < 100:  # If not enough data, generate synthetic data
                    print("Not enough historical data, generating synthetic data...")
                    return self._generate_synthetic_data(center_lat, center_lon, grid_size)
                
                # Use historical data
                print(f"Using {len(df)} records of historical data for training")
                X = df[self.feature_cols]
                y = df['flood_observed']
                
                # Split into training and test sets
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                
                # Normalize features
                X_train = self.scaler.fit_transform(X_train)
                X_test = self.scaler.transform(X_test)
                
                return X_train, X_test, y_train, y_test
                
            except Exception as e:
                print(f"Error loading historical data: {str(e)}")
                print("Falling back to synthetic data generation")
                return self._generate_synthetic_data(center_lat, center_lon, grid_size)
        else:
            # Generate synthetic data
            return self._generate_synthetic_data(center_lat, center_lon, grid_size)
    
    def _generate_synthetic_data(self, center_lat=DEFAULT_LATITUDE, center_lon=DEFAULT_LONGITUDE, 
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
        print("Generating synthetic training data...")
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
            # Convert risk level to binary target (1 for high/severe, 0 for low/medium)
            is_flood = 1 if row['risk_level'] in ['high', 'severe'] else 0
            
            all_data.append({
                'elevation': row['elevation'],
                'flow_accumulation': row['flow_acc'],
                'rainfall': row['nearest_rainfall'],
                'flood_observed': is_flood
            })
        
        # Simulate historical scenarios with different rainfall patterns
        for day in range(1, historical_days + 1):
            # Create synthetic rainfall data for this historical day
            # More variation for better training
            synthetic_rainfall = rainfall_data.copy()
            
            # Vary the rainfall randomly
            rain_factor = random.uniform(0.2, 3.0)  # Random factor between 0.2 and 3.0
            synthetic_rainfall['rainfall'] = synthetic_rainfall['rainfall'] * rain_factor
            
            # Calculate flood risk for this historical scenario
            historical_risk = calculate_flood_risk(grid, synthetic_rainfall)
            
            # Add to training data
            for _, row in historical_risk.iterrows():
                # Convert risk level to binary target
                is_flood = 1 if row['risk_level'] in ['high', 'severe'] else 0
                
                all_data.append({
                    'elevation': row['elevation'],
                    'flow_accumulation': row['flow_acc'],
                    'rainfall': row['nearest_rainfall'],
                    'flood_observed': is_flood
                })
        
        # Convert to DataFrame
        df = pd.DataFrame(all_data)
        
        # Split data into features and target
        X = df[self.feature_cols]
        y = df['flood_observed']
        
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
        
        # Prepare the features for prediction
        features = pd.DataFrame({
            'elevation': result['elevation'],
            'flow_accumulation': result['flow_acc'],
            'rainfall': result.apply(
                lambda row: find_nearest_rainfall(row, rainfall_data),
                axis=1
            )
        })
        
        # Make predictions
        result['predicted_risk'] = self.predict(features)
        
        # Convert to risk levels
        result['risk_level'] = pd.cut(
            result['predicted_risk'],
            bins=[0, 0.25, 0.5, 0.75, 1.0],
            labels=['low', 'medium', 'high', 'severe']
        )
        
        # Save predictions to CSV
        for _, row in result.iterrows():
            save_flood_prediction({
                'latitude': row['lat'],
                'longitude': row['lon'],
                'risk_level': row['risk_level'],
                'risk_score': row['predicted_risk']
            })
        
        return result
    
    def save(self, path=MODEL_PATH):
        """
        Save the model to a file.
        
        Args:
            path (str): Path to save the model
            
        Returns:
            bool: True if successful
        """
        if self.model is None:
            raise ValueError("Model not trained yet.")
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        # Save the model and scaler
        joblib.dump({
            'model': self.model,
            'scaler': self.scaler,
            'feature_cols': self.feature_cols
        }, path)
        
        return True
    
    def load(self, path=MODEL_PATH):
        """
        Load the model from a file.
        
        Args:
            path (str): Path to load the model from
            
        Returns:
            self: Loaded model instance
        """
        if not os.path.exists(path):
            raise ValueError(f"Model file not found: {path}")
        
        # Load the model and scaler
        data = joblib.load(path)
        self.model = data['model']
        self.scaler = data['scaler']
        self.feature_cols = data['feature_cols']
        
        return self

def find_nearest_rainfall(point, rainfall_data):
    """Find the nearest rainfall value for a point."""
    distances = []
    for _, row in rainfall_data.iterrows():
        # Calculate Euclidean distance
        dist = ((point['lat'] - row['latitude'])**2 + (point['lon'] - row['longitude'])**2)**0.5
        distances.append((dist, row['rainfall']))
    
    # Sort by distance
    distances.sort(key=lambda x: x[0])
    
    # Return the rainfall of the closest point
    return distances[0][1] if distances else 0

def train_model(save=True, use_historical=True):
    """
    Train the flood prediction model.
    
    Args:
        save (bool): Whether to save the model after training
        use_historical (bool): Whether to use historical data
        
    Returns:
        tuple: Model and evaluation metrics
    """
    # Ensure CSV data files are initialized
    initialize_data_files()
    
    # Create and train the model
    model = FloodPredictionModel()
    X_train, X_test, y_train, y_test = model.generate_training_data(use_historical=use_historical)
    model.train(X_train, y_train)
    
    # Evaluate the model
    metrics = model.evaluate(X_test, y_test)
    print(f"Model trained with metrics: {metrics}")
    
    # Save the model if requested
    if save:
        model.save()
        print(f"Model saved to: {MODEL_PATH}")
    
    return model, metrics

def load_or_train_model():
    """
    Load the model if it exists, otherwise train a new one.
    
    Returns:
        FloodPredictionModel: Loaded or trained model
    """
    model = FloodPredictionModel()
    
    try:
        # Try to load the model
        model.load()
        print("Model loaded successfully.")
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        print("Training a new model...")
        
        # Train a new model
        model, _ = train_model(save=True)
    
    return model

if __name__ == "__main__":
    model, metrics = train_model(save=True)
    print(f"Model trained with metrics:")
    print(f"MSE: {metrics['mse']:.4f}")
    print(f"RMSE: {metrics['rmse']:.4f}")
    print(f"R²: {metrics['r2']:.4f}") 
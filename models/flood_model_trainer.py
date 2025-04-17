import os
import sys
import numpy as np
import pandas as pd
import sqlite3
import joblib
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, mean_squared_error, r2_score

# Add parent directory to path to import config
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.config import DATABASE_PATH, MODEL_PATH

def get_column_mapping(df):
    """Get mapping between standardized column names and actual column names."""
    # Create a mapping between standardized column names and actual column names
    standard_to_actual = {}
    
    # Dictionary of standard column names to possible variations
    standard_columns = {
        'zone_id': ['zone_id', 'Zone_ID', 'zone_ID', 'ZONE_ID'],
        'neighborhood': ['neighborhood', 'Neighborhood', 'NEIGHBORHOOD'],
        'ward': ['ward', 'Ward', 'WARD'],
        'flood_risk_level': ['flood_risk_level', 'Flood_Risk_Level', 'FLOOD_RISK_LEVEL', 'risk_level'],
        'elevation_m': ['elevation_m', 'Elevation_m', 'ELEVATION_M', 'elevation'],
        'distance_coast_km': ['distance_coast_km', 'Distance_Coast_km', 'DISTANCE_COAST_KM'],
        'distance_river_km': ['distance_river_km', 'Distance_River_km', 'DISTANCE_RIVER_KM'],
        'avg_annual_flood_days': ['avg_annual_flood_days', 'Avg_Annual_Flood_Days', 'AVG_ANNUAL_FLOOD_DAYS', 'annual_flood_days'],
        'population_sqkm': ['population_sqkm', 'Population_SqKm', 'POPULATION_SQKM', 'population'],
        'drainage_quality_index': ['drainage_quality_index', 'Drainage_Quality_Index', 'DRAINAGE_QUALITY_INDEX'],
        'historical_aid_required_usd': ['historical_aid_required_usd', 'Historical_Aid_Required_USD', 'HISTORICAL_AID_REQUIRED_USD'],
        'flood_frequency_10yrs': ['flood_frequency_10yrs', 'Flood_Frequency_10yrs', 'FLOOD_FREQUENCY_10YRS'],
        'max_flood_depth_historical_m': ['max_flood_depth_historical_m', 'Max_Flood_Depth_Historical_m', 'MAX_FLOOD_DEPTH_HISTORICAL_M'],
        'avg_flood_duration_days': ['avg_flood_duration_days', 'Avg_Flood_Duration_Days', 'AVG_FLOOD_DURATION_DAYS']
    }
    
    # For each standard column, find a match in the DataFrame
    for standard, variations in standard_columns.items():
        for col in df.columns:
            if col in variations:
                standard_to_actual[standard] = col
                break
            elif col.lower() in [v.lower() for v in variations]:
                standard_to_actual[standard] = col
                break
    
    print("Column mapping:")
    for std, actual in standard_to_actual.items():
        print(f"  {std} -> {actual}")
    
    return standard_to_actual

def get_training_data():
    """Get training data from the database."""
    conn = sqlite3.connect(DATABASE_PATH)
    
    # Query to join flood zones and history data for training
    query = """
    SELECT 
        fz.*,
        fh.flood_frequency_10yrs, 
        fh.max_flood_depth_historical_m,
        fh.avg_flood_duration_days,
        ad.aid_delay_days,
        ad.infrastructure_damage_pct,
        ad.aid_satisfaction_1to10
    FROM 
        flood_zones fz
    LEFT JOIN 
        flood_history fh ON fz.neighborhood = fh.region_name
    LEFT JOIN 
        aid_data ad ON fz.ward = ad.ward
    """
    
    # Load data from database
    df = pd.read_sql_query(query, conn)
    conn.close()
    
    # Clean and prepare data
    df = df.dropna()  # Drop rows with missing values
    
    # Print column names for debugging
    print("Available columns in dataset:")
    print(df.columns.tolist())
    
    print(f"Loaded {len(df)} records for training")
    return df

def train_flood_risk_model(data):
    """Train a flood risk classification model."""
    # Get column mapping
    col_map = get_column_mapping(data)
    
    # Check if flood_risk_level column exists
    risk_column = col_map.get('flood_risk_level')
    
    if not risk_column:
        print(f"Error: Could not find flood risk level column.")
        return None
    
    print(f"Using risk column: {risk_column}")
    
    # Encoding flood risk levels
    risk_encoder = LabelEncoder()
    data['risk_encoded'] = risk_encoder.fit_transform(data[risk_column])
    
    # Define features and target using mapped column names
    feature_columns = ['elevation_m', 'distance_coast_km', 'distance_river_km',
                       'avg_annual_flood_days', 'drainage_quality_index',
                       'flood_frequency_10yrs', 'max_flood_depth_historical_m',
                       'avg_flood_duration_days']
    
    # Get actual column names
    features = [col_map.get(f) for f in feature_columns if col_map.get(f) is not None]
    
    # Ensure we have enough features
    if len(features) < 3:
        print("Error: Not enough valid features found for training.")
        print(f"Features found: {features}")
        return None
    
    print(f"Using features: {features}")
    
    X = data[features]
    y = data['risk_encoded']
    
    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Standardize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train RandomForest classifier
    print("Training flood risk classification model...")
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train_scaled, y_train)
    
    # Evaluate the model
    y_pred = model.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Model accuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # Save the model, scaler, and encoder
    model_data = {
        'model': model,
        'scaler': scaler,
        'risk_encoder': risk_encoder,
        'features': features,
        'risk_classes': risk_encoder.classes_,
        'risk_column': risk_column,
        'column_mapping': col_map
    }
    
    model_dir = os.path.dirname(MODEL_PATH)
    os.makedirs(model_dir, exist_ok=True)
    
    classification_model_path = os.path.join(model_dir, 'flood_risk_model.pkl')
    joblib.dump(model_data, classification_model_path)
    print(f"Classification model saved to: {classification_model_path}")
    
    return model_data

def train_aid_estimation_model(data):
    """Train a model to estimate required aid."""
    # Get column mapping
    col_map = get_column_mapping(data)
    
    # Determine which columns exist in the data
    aid_column = col_map.get('historical_aid_required_usd')
    
    if not aid_column:
        print(f"Error: Could not find historical aid column.")
        return None
    
    print(f"Using aid column: {aid_column}")
    
    # Define features for aid estimation using mapped column names
    feature_columns = ['elevation_m', 'distance_coast_km', 'distance_river_km',
                      'avg_annual_flood_days', 'population_sqkm', 'drainage_quality_index',
                      'flood_frequency_10yrs', 'max_flood_depth_historical_m', 'avg_flood_duration_days']
    
    # Get actual column names
    valid_features = [col_map.get(f) for f in feature_columns if col_map.get(f) is not None]
    
    # Ensure we have enough features
    if len(valid_features) < 3:
        print("Error: Not enough valid features found for aid estimation.")
        print(f"Features found: {valid_features}")
        return None
    
    print(f"Using features for aid estimation: {valid_features}")
    
    X = data[valid_features]
    y = data[aid_column]  # Target: historical aid required
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train RandomForest regressor
    print("\nTraining aid estimation model...")
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train_scaled, y_train)
    
    # Evaluate model
    y_pred = model.predict(X_test_scaled)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print(f"Mean Squared Error: {mse:.2f}")
    print(f"R² Score: {r2:.4f}")
    
    # Save the model
    model_data = {
        'model': model,
        'scaler': scaler,
        'features': valid_features,
        'aid_column': aid_column,
        'column_mapping': col_map
    }
    
    model_dir = os.path.dirname(MODEL_PATH)
    os.makedirs(model_dir, exist_ok=True)
    
    aid_model_path = os.path.join(model_dir, 'aid_estimation_model.pkl')
    joblib.dump(model_data, aid_model_path)
    print(f"Aid estimation model saved to: {aid_model_path}")
    
    return model_data

def train_models():
    """Train all models using the database data."""
    # Get data from the database
    data = get_training_data()
    
    if len(data) < 10:
        print("Not enough data available for training.")
        return None
    
    # Train flood risk classification model
    risk_model = train_flood_risk_model(data)
    
    # Train aid estimation model
    aid_model = train_aid_estimation_model(data)
    
    return {
        'risk_model': risk_model,
        'aid_model': aid_model
    }

if __name__ == "__main__":
    # Check if database exists
    if not os.path.exists(DATABASE_PATH):
        print(f"Database not found at {DATABASE_PATH}. Please import data first.")
        sys.exit(1)
    
    # Train models
    train_models() 
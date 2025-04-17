import os
import sys
import numpy as np
import pandas as pd
import joblib
import sqlite3

# Add parent directory to path to import config
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.config import MODEL_PATH, DATABASE_PATH

class FloodPredictor:
    """Class to make flood risk and aid requirement predictions."""
    
    def __init__(self):
        """Initialize the predictor."""
        self.risk_model = None
        self.aid_model = None
        self.column_mapping = {}
        self.load_models()
    
    def load_models(self):
        """Load trained models."""
        model_dir = os.path.dirname(MODEL_PATH)
        
        # Load risk classification model
        risk_model_path = os.path.join(model_dir, 'flood_risk_model.pkl')
        if os.path.exists(risk_model_path):
            self.risk_model = joblib.load(risk_model_path)
            print(f"Loaded flood risk model from: {risk_model_path}")
            
            # Get column mapping from risk model
            if 'column_mapping' in self.risk_model:
                self.column_mapping = self.risk_model['column_mapping']
                print(f"Loaded column mapping with {len(self.column_mapping)} entries")
        else:
            print(f"Flood risk model not found at: {risk_model_path}")
        
        # Load aid estimation model
        aid_model_path = os.path.join(model_dir, 'aid_estimation_model.pkl')
        if os.path.exists(aid_model_path):
            self.aid_model = joblib.load(aid_model_path)
            print(f"Loaded aid estimation model from: {aid_model_path}")
            
            # If we don't have a column mapping yet, get it from aid model
            if not self.column_mapping and 'column_mapping' in self.aid_model:
                self.column_mapping = self.aid_model['column_mapping']
                print(f"Loaded column mapping from aid model with {len(self.column_mapping)} entries")
        else:
            print(f"Aid estimation model not found at: {aid_model_path}")
    
    def map_feature_names(self, features_dict):
        """Map feature names to those expected by the model."""
        # Check if we have a column mapping
        if not self.column_mapping:
            return features_dict
            
        # Create a mapping from actual column names to standard names
        actual_to_standard = {v: k for k, v in self.column_mapping.items()}
        
        # Map the feature dict keys to standard names
        mapped_features = {}
        for key, value in features_dict.items():
            if key in actual_to_standard:
                # Use the standardized name
                mapped_features[actual_to_standard[key]] = value
            else:
                # Keep the original name
                mapped_features[key] = value
                
        return mapped_features
    
    def predict_flood_risk(self, features_dict):
        """
        Predict flood risk level based on input features.
        
        Args:
            features_dict (dict): Dictionary with feature values
            
        Returns:
            dict: Prediction results with risk level and confidence
        """
        if self.risk_model is None:
            return {"error": "Risk model not loaded"}
        
        # Map feature names
        mapped_features = self.map_feature_names(features_dict)
        
        # Extract model components
        model = self.risk_model['model']
        scaler = self.risk_model['scaler']
        risk_encoder = self.risk_model['risk_encoder']
        features = self.risk_model['features']
        risk_classes = self.risk_model['risk_classes']
        
        # Try to handle any column name mismatches
        input_features = []
        missing_features = []
        
        for feature in features:
            found = False
            
            # Check the exact name
            if feature in features_dict:
                input_features.append(features_dict[feature])
                found = True
            # Try lowercase version
            elif feature.lower() in [k.lower() for k in features_dict.keys()]:
                for k in features_dict.keys():
                    if k.lower() == feature.lower():
                        input_features.append(features_dict[k])
                        found = True
                        break
            
            if not found:
                missing_features.append(feature)
        
        if missing_features:
            return {"error": f"Missing required features: {', '.join(missing_features)}"}
        
        # Prepare input features
        X = np.array(input_features).reshape(1, -1)
        
        # Scale features
        X_scaled = scaler.transform(X)
        
        # Make prediction
        risk_class_idx = model.predict(X_scaled)[0]
        risk_level = risk_classes[risk_class_idx]
        
        # Get prediction probabilities
        proba = model.predict_proba(X_scaled)[0]
        confidence = proba[risk_class_idx] * 100
        
        return {
            "risk_level": risk_level,
            "confidence": confidence,
            "risk_probabilities": {risk_classes[i]: float(proba[i] * 100) for i in range(len(risk_classes))}
        }
    
    def predict_aid_required(self, features_dict):
        """
        Predict required aid amount based on input features.
        
        Args:
            features_dict (dict): Dictionary with feature values
            
        Returns:
            dict: Prediction results with estimated aid amount
        """
        if self.aid_model is None:
            return {"error": "Aid model not loaded"}
        
        # Map feature names
        mapped_features = self.map_feature_names(features_dict)
        
        # Extract model components
        model = self.aid_model['model']
        scaler = self.aid_model['scaler']
        features = self.aid_model['features']
        
        # Try to handle any column name mismatches
        input_features = []
        missing_features = []
        
        for feature in features:
            found = False
            
            # Check the exact name
            if feature in features_dict:
                input_features.append(features_dict[feature])
                found = True
            # Try lowercase version
            elif feature.lower() in [k.lower() for k in features_dict.keys()]:
                for k in features_dict.keys():
                    if k.lower() == feature.lower():
                        input_features.append(features_dict[k])
                        found = True
                        break
            
            if not found:
                missing_features.append(feature)
        
        if missing_features:
            return {"error": f"Missing required features for aid prediction: {', '.join(missing_features)}"}
        
        # Prepare input features
        X = np.array(input_features).reshape(1, -1)
        
        # Scale features
        X_scaled = scaler.transform(X)
        
        # Make prediction
        aid_amount = model.predict(X_scaled)[0]
        
        return {
            "estimated_aid_usd": aid_amount,
            "estimated_aid_formatted": f"${aid_amount:,.2f}",
            "feature_importance": {features[i]: float(model.feature_importances_[i]) for i in range(len(features))}
        }
    
    def predict_for_zone(self, zone_id):
        """
        Make predictions for a specific flood zone by ID.
        
        Args:
            zone_id (str): ID of the flood zone to predict for
            
        Returns:
            dict: Combined prediction results
        """
        # Connect to database
        conn = sqlite3.connect(DATABASE_PATH)
        
        # Get all column names in flood_zones
        cursor = conn.cursor()
        cursor.execute("PRAGMA table_info(flood_zones)")
        zone_columns = [row[1] for row in cursor.fetchall()]
        
        # Get all column names in flood_history
        cursor.execute("PRAGMA table_info(flood_history)")
        history_columns = [row[1] for row in cursor.fetchall()]
        
        # Build dynamic query based on available columns
        select_clauses = []
        
        # Add zone columns
        for col in zone_columns:
            select_clauses.append(f"fz.{col}")
        
        # Add history columns
        for col in history_columns:
            if col != 'region_id' and col != 'region_name' and col != 'country':
                select_clauses.append(f"fh.{col}")
        
        # Join all select clauses
        select_sql = ", ".join(select_clauses)
        
        # Determine zone_id column name
        zone_id_col = self.column_mapping.get('zone_id', 'zone_id')
        
        # Find the actual column name in the database
        for col in zone_columns:
            if col.lower() == zone_id_col.lower():
                zone_id_col = col
                break
        
        # Build query
        query = f"""
        SELECT 
            {select_sql}
        FROM 
            flood_zones fz
        LEFT JOIN 
            flood_history fh ON fz.neighborhood = fh.region_name
        WHERE 
            fz.{zone_id_col} = ?
        """
        
        df = pd.read_sql_query(query, conn, params=(zone_id,))
        conn.close()
        
        if len(df) == 0:
            return {"error": f"Zone ID not found: {zone_id}"}
        
        # Convert row to dictionary
        features_dict = df.iloc[0].to_dict()
        
        # Make predictions
        risk_prediction = self.predict_flood_risk(features_dict)
        aid_prediction = self.predict_aid_required(features_dict)
        
        # Get the actual column names
        neighborhood_col = self.column_mapping.get('neighborhood', 'neighborhood')
        ward_col = self.column_mapping.get('ward', 'ward')
        risk_col = self.column_mapping.get('flood_risk_level', 'flood_risk_level')
        
        # Find the actual column names in the features
        for col in features_dict.keys():
            if col.lower() == neighborhood_col.lower():
                neighborhood_col = col
            elif col.lower() == ward_col.lower():
                ward_col = col
            elif col.lower() == risk_col.lower():
                risk_col = col
        
        # Return combined results
        return {
            "zone_id": zone_id,
            "neighborhood": features_dict.get(neighborhood_col),
            "ward": features_dict.get(ward_col),
            "actual_risk_level": features_dict.get(risk_col),
            "predicted_risk": risk_prediction,
            "aid_prediction": aid_prediction
        }
    
    def predict_all_zones(self):
        """
        Make predictions for all flood zones in the database.
        
        Returns:
            list: Prediction results for all zones
        """
        # Connect to database
        conn = sqlite3.connect(DATABASE_PATH)
        
        # Determine zone_id column name
        zone_id_col = self.column_mapping.get('zone_id', 'zone_id')
        
        # Get all zone IDs
        cursor = conn.cursor()
        cursor.execute("PRAGMA table_info(flood_zones)")
        zone_columns = [row[1] for row in cursor.fetchall()]
        
        # Find the actual column name in the database
        for col in zone_columns:
            if col.lower() == zone_id_col.lower():
                zone_id_col = col
                break
        
        # Get all zone IDs
        query = f"SELECT {zone_id_col} FROM flood_zones"
        zones_df = pd.read_sql_query(query, conn)
        conn.close()
        
        # Make predictions for each zone
        results = []
        for zone_id in zones_df[zone_id_col]:
            prediction = self.predict_for_zone(zone_id)
            results.append(prediction)
        
        return results

def main():
    """Main function to demonstrate prediction."""
    predictor = FloodPredictor()
    
    # Check if zone ID is provided as argument
    if len(sys.argv) > 1:
        zone_id = sys.argv[1]
        print(f"Making prediction for zone: {zone_id}")
        result = predictor.predict_for_zone(zone_id)
        
        if "error" in result:
            print(f"Error: {result['error']}")
            return
            
        print("\nPrediction Result:")
        print(f"Zone ID: {result['zone_id']}")
        print(f"Neighborhood: {result['neighborhood']}")
        print(f"Ward: {result['ward']}")
        print(f"Actual Risk Level: {result['actual_risk_level']}")
        print(f"Predicted Risk Level: {result['predicted_risk'].get('risk_level', 'N/A')} (Confidence: {result['predicted_risk'].get('confidence', 0):.2f}%)")
        print(f"Estimated Aid Required: {result['aid_prediction'].get('estimated_aid_formatted', 'N/A')}")
    else:
        # Determine required features from model
        required_features = {}
        
        if predictor.risk_model and 'features' in predictor.risk_model:
            risk_features = predictor.risk_model['features']
            print(f"Flood risk model requires {len(risk_features)} features: {risk_features}")
        
        if predictor.aid_model and 'features' in predictor.aid_model:
            aid_features = predictor.aid_model['features']
            print(f"Aid estimation model requires {len(aid_features)} features: {aid_features}")
        
        print("\nNo zone ID provided. Using sample features for prediction.")
        
        # Create sample features based on the model
        sample_features = {}
        if predictor.risk_model and 'features' in predictor.risk_model:
            for i, feature in enumerate(predictor.risk_model['features']):
                # Generate sample values appropriate for each feature
                if 'elevation' in feature.lower():
                    sample_features[feature] = 3.0  # elevation in meters
                elif 'distance' in feature.lower() and 'coast' in feature.lower():
                    sample_features[feature] = 2.5  # distance from coast in km
                elif 'distance' in feature.lower() and 'river' in feature.lower():
                    sample_features[feature] = 0.8  # distance from river in km
                elif 'flood' in feature.lower() and 'days' in feature.lower():
                    sample_features[feature] = 15  # annual flood days
                elif 'drainage' in feature.lower():
                    sample_features[feature] = 4  # drainage quality index
                elif 'frequency' in feature.lower():
                    sample_features[feature] = 3  # flood frequency over 10 years
                elif 'depth' in feature.lower():
                    sample_features[feature] = 1.5  # max flood depth in meters
                elif 'duration' in feature.lower():
                    sample_features[feature] = 5  # flood duration in days
                else:
                    # Default numeric value
                    sample_features[feature] = i + 1
        
        print("Sample features for prediction:")
        for feature, value in sample_features.items():
            print(f"  {feature}: {value}")
        
        risk_result = predictor.predict_flood_risk(sample_features)
        
        if "error" in risk_result:
            print(f"Risk Prediction Error: {risk_result['error']}")
        else:
            print("\nRisk Prediction:")
            print(f"Risk Level: {risk_result['risk_level']}")
            print(f"Confidence: {risk_result['confidence']:.2f}%")
        
        aid_result = predictor.predict_aid_required(sample_features)
        
        if "error" in aid_result:
            print(f"Aid Prediction Error: {aid_result['error']}")
        else:
            print("\nAid Prediction:")
            print(f"Estimated Aid Required: {aid_result['estimated_aid_formatted']}")

if __name__ == "__main__":
    main() 
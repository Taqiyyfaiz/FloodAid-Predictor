import os
import sys
import argparse

# Add the current directory to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import our modules
from utils.db_data_importer import import_all_csv_data
from models.flood_model_trainer import train_models
from models.flood_predictor import FloodPredictor
from config.config import DATABASE_PATH

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Run the flood prediction pipeline.')
    parser.add_argument('--import-data', action='store_true', help='Import CSV data into database')
    parser.add_argument('--train-models', action='store_true', help='Train prediction models')
    parser.add_argument('--predict', action='store_true', help='Make predictions')
    parser.add_argument('--zone-id', type=str, help='Zone ID to predict for (optional)')
    parser.add_argument('--all', action='store_true', help='Run the complete pipeline')
    
    return parser.parse_args()

def run_import_data():
    """Run the data import process."""
    print("\n=====================")
    print("IMPORTING CSV DATA")
    print("=====================\n")
    
    # CSV file paths
    csv_dir = os.path.join(os.path.dirname(__file__), "data", "csv")
    flood_zones_csv = os.path.join(csv_dir, "Flood zone.csv")
    flood_history_csv = os.path.join(csv_dir, "Flood history.csv")
    aid_data_csv = os.path.join(csv_dir, "Aid data.csv")
    
    # Check if files exist
    missing_files = []
    for file_path in [flood_zones_csv, flood_history_csv, aid_data_csv]:
        if not os.path.exists(file_path):
            missing_files.append(file_path)
    
    if missing_files:
        print("Error: The following files are missing:")
        for file_path in missing_files:
            print(f"- {file_path}")
        return False
    
    # Import data
    import_all_csv_data(flood_zones_csv, flood_history_csv, aid_data_csv)
    
    return os.path.exists(DATABASE_PATH)

def run_train_models():
    """Run the model training process."""
    print("\n=====================")
    print("TRAINING MODELS")
    print("=====================\n")
    
    # Check if database exists
    if not os.path.exists(DATABASE_PATH):
        print(f"Error: Database not found at {DATABASE_PATH}")
        print("Please run the data import step first.")
        return False
    
    # Train models
    models = train_models()
    
    return models is not None

def run_predictions(zone_id=None):
    """Run the prediction process."""
    print("\n=====================")
    print("MAKING PREDICTIONS")
    print("=====================\n")
    
    # Initialize predictor
    predictor = FloodPredictor()
    
    if zone_id:
        # Predict for specific zone
        print(f"Making prediction for zone: {zone_id}")
        result = predictor.predict_for_zone(zone_id)
        
        if "error" in result:
            print(f"Error: {result['error']}")
            return False
        
        print("\nPrediction Result:")
        print(f"Zone ID: {result['zone_id']}")
        print(f"Neighborhood: {result['neighborhood']}")
        print(f"Ward: {result['ward']}")
        print(f"Actual Risk Level: {result['actual_risk_level']}")
        print(f"Predicted Risk Level: {result['predicted_risk']['risk_level']} (Confidence: {result['predicted_risk']['confidence']:.2f}%)")
        print(f"Estimated Aid Required: {result['aid_prediction']['estimated_aid_formatted']}")
    else:
        # Predict for all zones
        print("Making predictions for all zones...")
        results = predictor.predict_all_zones()
        
        # Print summary
        risk_levels = {}
        total_aid = 0
        
        for result in results:
            if "error" not in result:
                risk_level = result['predicted_risk'].get('risk_level')
                if risk_level in risk_levels:
                    risk_levels[risk_level] += 1
                else:
                    risk_levels[risk_level] = 1
                
                aid_amount = result['aid_prediction'].get('estimated_aid_usd', 0)
                total_aid += aid_amount
        
        print("\nPrediction Summary:")
        print(f"Total zones predicted: {len(results)}")
        print("\nRisk Level Distribution:")
        for level, count in risk_levels.items():
            print(f"- {level}: {count} zones ({count/len(results)*100:.1f}%)")
        
        print(f"\nTotal Estimated Aid Required: ${total_aid:,.2f}")
    
    return True

def main():
    """Main function to run the pipeline."""
    args = parse_arguments()
    
    # If no arguments provided or --all flag is set, run the complete pipeline
    if len(sys.argv) == 1 or args.all:
        print("Running the complete pipeline...\n")
        
        # Import data
        if not run_import_data():
            print("Error importing data. Pipeline stopped.")
            return
        
        # Train models
        if not run_train_models():
            print("Error training models. Pipeline stopped.")
            return
        
        # Make predictions
        run_predictions()
    else:
        # Run only the specified steps
        if args.import_data:
            run_import_data()
        
        if args.train_models:
            run_train_models()
        
        if args.predict:
            run_predictions(args.zone_id)
    
    print("\nPipeline execution completed.")

if __name__ == "__main__":
    main() 
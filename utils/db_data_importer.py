import os
import pandas as pd
import sqlite3
import sys

# Add parent directory to path to import config
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.config import DATABASE_PATH

def create_database_tables():
    """Create database tables if they don't exist."""
    # Ensure database directory exists
    os.makedirs(os.path.dirname(DATABASE_PATH), exist_ok=True)
    
    # Connect to the database
    conn = sqlite3.connect(DATABASE_PATH)
    cursor = conn.cursor()
    
    # Create flood_zones table
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS flood_zones (
        zone_id TEXT PRIMARY KEY,
        neighborhood TEXT,
        ward TEXT,
        flood_risk_level TEXT,
        elevation_m REAL,
        distance_coast_km REAL,
        distance_river_km REAL,
        avg_annual_flood_days INTEGER,
        population_sqkm INTEGER,
        drainage_quality_index INTEGER,
        historical_aid_required_usd INTEGER
    )
    ''')
    
    # Create flood_history table
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS flood_history (
        region_id TEXT PRIMARY KEY,
        region_name TEXT,
        country TEXT,
        flood_last_5_years TEXT,
        flood_frequency_10yrs INTEGER,
        worst_flood_year INTEGER,
        max_flood_depth_historical_m REAL,
        avg_flood_duration_days REAL,
        people_affected_avg INTEGER,
        historical_aid_avg_usd INTEGER
    )
    ''')
    
    # Create aid_data table
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS aid_data (
        id INTEGER PRIMARY KEY,
        ward TEXT,
        population_affected INTEGER,
        economic_loss_inr INTEGER,
        aid_requested_inr INTEGER,
        aid_received_inr INTEGER,
        aid_delay_days INTEGER,
        infrastructure_damage_pct INTEGER,
        relief_camps INTEGER,
        aid_satisfaction_1to10 INTEGER
    )
    ''')
    
    # Commit changes
    conn.commit()
    conn.close()
    
    print("Database tables created successfully")

def import_flood_zones(csv_path):
    """Import flood zones data from CSV to SQLite database."""
    try:
        # Read CSV file
        df = pd.read_csv(csv_path)
        
        # Connect to database
        conn = sqlite3.connect(DATABASE_PATH)
        
        # Insert data into flood_zones table
        df.to_sql('flood_zones', conn, if_exists='replace', index=False)
        
        conn.commit()
        conn.close()
        
        print(f"Imported {len(df)} flood zone records")
        return True
    except Exception as e:
        print(f"Error importing flood zones: {str(e)}")
        return False

def import_flood_history(csv_path):
    """Import flood history data from CSV to SQLite database."""
    try:
        # Read CSV file
        df = pd.read_csv(csv_path)
        
        # Connect to database
        conn = sqlite3.connect(DATABASE_PATH)
        
        # Insert data into flood_history table
        df.to_sql('flood_history', conn, if_exists='replace', index=False)
        
        conn.commit()
        conn.close()
        
        print(f"Imported {len(df)} flood history records")
        return True
    except Exception as e:
        print(f"Error importing flood history: {str(e)}")
        return False

def import_aid_data(csv_path):
    """Import aid data from CSV to SQLite database."""
    try:
        # Read CSV file
        df = pd.read_csv(csv_path)
        
        # Connect to database
        conn = sqlite3.connect(DATABASE_PATH)
        
        # Insert data into aid_data table
        df.to_sql('aid_data', conn, if_exists='replace', index=False)
        
        conn.commit()
        conn.close()
        
        print(f"Imported {len(df)} aid data records")
        return True
    except Exception as e:
        print(f"Error importing aid data: {str(e)}")
        return False

def import_all_csv_data(flood_zones_csv, flood_history_csv, aid_data_csv):
    """Import all CSV data into the SQLite database."""
    # Create tables
    create_database_tables()
    
    # Import data
    import_flood_zones(flood_zones_csv)
    import_flood_history(flood_history_csv)
    import_aid_data(aid_data_csv)
    
    print("All data imported successfully")

if __name__ == "__main__":
    # CSV file paths
    FLOOD_ZONES_CSV = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "csv", "Flood zone.csv")
    FLOOD_HISTORY_CSV = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "csv", "Flood history.csv")
    AID_DATA_CSV = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "csv", "Aid data.csv")
    
    # Import all data
    import_all_csv_data(FLOOD_ZONES_CSV, FLOOD_HISTORY_CSV, AID_DATA_CSV) 
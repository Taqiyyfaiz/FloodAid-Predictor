import sqlite3
import os
import pandas as pd
from datetime import datetime
import sys
import os

# Add parent directory to path to import config
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.config import DATABASE_PATH

def create_connection():
    """Create a database connection to the SQLite database."""
    conn = None
    try:
        # Ensure directory exists
        os.makedirs(os.path.dirname(DATABASE_PATH), exist_ok=True)
        conn = sqlite3.connect(DATABASE_PATH)
        return conn
    except sqlite3.Error as e:
        print(f"Error connecting to database: {e}")
    return conn

def create_tables():
    """Create the necessary tables if they don't exist."""
    conn = create_connection()
    if conn is not None:
        cursor = conn.cursor()
        
        # Weather data table
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS weather_data (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            date TEXT,
            latitude REAL,
            longitude REAL,
            rainfall REAL,
            temperature REAL,
            humidity REAL,
            timestamp TEXT
        )
        ''')
        
        # Flood predictions table
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS flood_predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            latitude REAL,
            longitude REAL,
            risk_level TEXT,
            risk_score REAL,
            prediction_date TEXT
        )
        ''')
        
        # Aid centers table
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS aid_centers (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT,
            latitude REAL,
            longitude REAL,
            capacity INTEGER,
            current_stock INTEGER
        )
        ''')
        
        # Routes table
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS routes (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            aid_center_id INTEGER,
            destination_lat REAL,
            destination_lon REAL,
            distance REAL,
            estimated_time REAL,
            risk_level TEXT,
            created_date TEXT,
            FOREIGN KEY (aid_center_id) REFERENCES aid_centers (id)
        )
        ''')
        
        conn.commit()
        conn.close()
    else:
        print("Error! Cannot create the database connection.")

def insert_weather_data(data):
    """Insert weather data into the database."""
    conn = create_connection()
    if conn is not None:
        cursor = conn.cursor()
        cursor.execute('''
        INSERT INTO weather_data (date, latitude, longitude, rainfall, temperature, humidity, timestamp)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (data['date'], data['latitude'], data['longitude'], data['rainfall'], 
              data['temperature'], data['humidity'], datetime.now().isoformat()))
        conn.commit()
        conn.close()
    else:
        print("Error! Cannot create the database connection.")

def insert_flood_prediction(data):
    """Insert flood prediction into the database."""
    conn = create_connection()
    if conn is not None:
        cursor = conn.cursor()
        cursor.execute('''
        INSERT INTO flood_predictions (latitude, longitude, risk_level, risk_score, prediction_date)
        VALUES (?, ?, ?, ?, ?)
        ''', (data['latitude'], data['longitude'], data['risk_level'], 
              data['risk_score'], datetime.now().isoformat()))
        conn.commit()
        conn.close()
    else:
        print("Error! Cannot create the database connection.")

def insert_aid_center(data):
    """Insert an aid center into the database."""
    conn = create_connection()
    if conn is not None:
        cursor = conn.cursor()
        cursor.execute('''
        INSERT INTO aid_centers (name, latitude, longitude, capacity, current_stock)
        VALUES (?, ?, ?, ?, ?)
        ''', (data['name'], data['latitude'], data['longitude'], 
              data['capacity'], data['current_stock']))
        conn.commit()
        conn.close()
    else:
        print("Error! Cannot create the database connection.")

def insert_route(data):
    """Insert a route into the database."""
    conn = create_connection()
    if conn is not None:
        cursor = conn.cursor()
        cursor.execute('''
        INSERT INTO routes (aid_center_id, destination_lat, destination_lon, distance, 
                          estimated_time, risk_level, created_date)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (data['aid_center_id'], data['destination_lat'], data['destination_lon'],
              data['distance'], data['estimated_time'], data['risk_level'], 
              datetime.now().isoformat()))
        conn.commit()
        conn.close()
    else:
        print("Error! Cannot create the database connection.")

def get_weather_data():
    """Get all weather data from the database."""
    conn = create_connection()
    if conn is not None:
        df = pd.read_sql_query("SELECT * FROM weather_data", conn)
        conn.close()
        return df
    else:
        print("Error! Cannot create the database connection.")
        return pd.DataFrame()

def get_flood_predictions():
    """Get all flood predictions from the database."""
    conn = create_connection()
    if conn is not None:
        df = pd.read_sql_query("SELECT * FROM flood_predictions", conn)
        conn.close()
        return df
    else:
        print("Error! Cannot create the database connection.")
        return pd.DataFrame()

def get_aid_centers():
    """Get all aid centers from the database."""
    conn = create_connection()
    if conn is not None:
        df = pd.read_sql_query("SELECT * FROM aid_centers", conn)
        conn.close()
        return df
    else:
        print("Error! Cannot create the database connection.")
        return pd.DataFrame()

def get_routes():
    """Get all routes from the database."""
    conn = create_connection()
    if conn is not None:
        query = """
        SELECT r.*, a.name as aid_center_name 
        FROM routes r
        JOIN aid_centers a ON r.aid_center_id = a.id
        """
        df = pd.read_sql_query(query, conn)
        conn.close()
        return df
    else:
        print("Error! Cannot create the database connection.")
        return pd.DataFrame()

# Initialize the database
if __name__ == "__main__":
    create_tables() 
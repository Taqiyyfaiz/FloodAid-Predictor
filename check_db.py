import sqlite3

# Connect to the database
conn = sqlite3.connect('floodaid.db')
cursor = conn.cursor()

# Get all tables
cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
tables = cursor.fetchall()

print("Tables in the database:")
for table in tables:
    print(f"- {table[0]}")
    # Get schema for each table
    cursor.execute(f"PRAGMA table_info({table[0]});")
    columns = cursor.fetchall()
    print("  Columns:")
    for col in columns:
        print(f"    - {col[1]} ({col[2]})")
    print()

# Close connection
cursor.close()
conn.close() 
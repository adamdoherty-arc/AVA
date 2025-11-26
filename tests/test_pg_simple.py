"""Simple PostgreSQL connection test"""

import psycopg2
from dotenv import load_dotenv
import os

load_dotenv()

try:
    conn = psycopg2.connect(
        host=os.getenv('DB_HOST', 'localhost'),
        port=os.getenv('DB_PORT', '5432'),
        user=os.getenv('DB_USER', 'postgres'),
        password=os.getenv('DB_PASSWORD', 'postgres123!'),
        database='postgres'
    )

    print("SUCCESS! Connected to PostgreSQL")

    cursor = conn.cursor()

    # List databases
    cursor.execute("SELECT datname FROM pg_database WHERE datistemplate = false;")
    databases = cursor.fetchall()

    print("\nExisting databases:")
    for db in databases:
        print(f"  - {db[0]}")

    # Check if wheel_strategy exists
    db_names = [db[0] for db in databases]

    if 'wheel_strategy' not in db_names:
        print("\nCreating 'wheel_strategy' database...")
        conn.set_isolation_level(psycopg2.extensions.ISOLATION_LEVEL_AUTOCOMMIT)
        cursor.execute("CREATE DATABASE wheel_strategy")
        print("Database 'wheel_strategy' created!")
    else:
        print("\nDatabase 'wheel_strategy' already exists!")

    if 'magnus' in db_names:
        print("Found your 'magnus' database - you can use it if you prefer")

    cursor.close()
    conn.close()

    print("\nPostgreSQL is ready to use!")
    print("You can now use:")
    print("  - Database Scan feature")
    print("  - TradingView Watchlists (will store in DB)")

except Exception as e:
    print(f"Connection failed: {e}")
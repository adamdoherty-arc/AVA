"""Test magnus database connection"""

import psycopg2
from dotenv import load_dotenv
import os

load_dotenv()

try:
    # Connect to magnus database
    conn = psycopg2.connect(
        host=os.getenv('DB_HOST', 'localhost'),
        port=os.getenv('DB_PORT', '5432'),
        user=os.getenv('DB_USER', 'postgres'),
        password=os.getenv('DB_PASSWORD', 'postgres123!'),
        database='magnus'
    )

    print("SUCCESS! Connected to magnus database")

    cursor = conn.cursor()

    # List tables in magnus
    cursor.execute("""
        SELECT table_name
        FROM information_schema.tables
        WHERE table_schema = 'public'
        ORDER BY table_name;
    """)

    tables = cursor.fetchall()

    if tables:
        print("\nExisting tables in magnus database:")
        for table in tables:
            print(f"  - {table[0]}")

        # Get row count for each table
        print("\nTable row counts:")
        for table in tables:
            cursor.execute(f"SELECT COUNT(*) FROM {table[0]}")
            count = cursor.fetchone()[0]
            print(f"  - {table[0]}: {count} rows")
    else:
        print("\nNo tables found in magnus database")
        print("Tables will be created when you use the app features")

    cursor.close()
    conn.close()

    print("\n" + "="*50)
    print("Magnus database is ready to use!")
    print("The app will now use your existing magnus database")
    print("="*50)

except Exception as e:
    print(f"Connection failed: {e}")
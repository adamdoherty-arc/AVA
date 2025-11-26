"""Test PostgreSQL connection and setup"""

import psycopg2
from psycopg2 import sql
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

print("PostgreSQL Connection Test")
print("=" * 50)

# Connection parameters from your magnus database
connections_to_try = [
    {
        "name": "Current .env settings",
        "host": os.getenv('DB_HOST', 'localhost'),
        "port": os.getenv('DB_PORT', '5432'),
        "user": os.getenv('DB_USER', 'postgres'),
        "password": os.getenv('DB_PASSWORD', 'postgres123!'),
        "database": "postgres"  # Connect to default DB first
    },
    {
        "name": "Your magnus database settings",
        "host": "localhost",
        "port": "5432",
        "user": "postgres",
        "password": "postgres123!",
        "database": "postgres"
    }
]

for conn_params in connections_to_try:
    print(f"\nTrying: {conn_params['name']}")
    print(f"  Host: {conn_params['host']}")
    print(f"  Port: {conn_params['port']}")
    print(f"  User: {conn_params['user']}")
    print(f"  Password: {'*' * len(conn_params['password'])}")

    try:
        # Try to connect
        conn = psycopg2.connect(
            host=conn_params['host'],
            port=conn_params['port'],
            user=conn_params['user'],
            password=conn_params['password'],
            database=conn_params['database']
        )

        print("  ✓ Connection successful!")

        # Get cursor
        cursor = conn.cursor()

        # List all databases
        cursor.execute("SELECT datname FROM pg_database WHERE datistemplate = false;")
        databases = cursor.fetchall()

        print(f"  Available databases:")
        for db in databases:
            print(f"    - {db[0]}")

        # Check if wheel_strategy database exists
        db_names = [db[0] for db in databases]

        if 'wheel_strategy' not in db_names:
            print("\n  Creating 'wheel_strategy' database...")
            conn.set_isolation_level(psycopg2.extensions.ISOLATION_LEVEL_AUTOCOMMIT)
            cursor.execute(sql.SQL("CREATE DATABASE {}").format(
                sql.Identifier('wheel_strategy')
            ))
            print("  ✓ Database 'wheel_strategy' created!")
        else:
            print("  ✓ Database 'wheel_strategy' already exists")

        # Check if magnus database exists (for reuse)
        if 'magnus' in db_names:
            print("  ✓ Found your 'magnus' database - you can reuse its data")

        cursor.close()
        conn.close()

        print("\n  SUCCESS! PostgreSQL is properly configured.")
        print("\n  Next steps:")
        print("  1. The password 'postgres123!' is correct")
        print("  2. Database 'wheel_strategy' is ready")
        print("  3. You can now use Database Scan in the app")

        break

    except psycopg2.OperationalError as e:
        print(f"  ✗ Connection failed: {e}")

        # Provide specific guidance
        if "password authentication failed" in str(e):
            print("\n  The password might be different. Try these steps:")
            print("  1. Open pgAdmin or psql")
            print("  2. Reset the postgres password:")
            print("     ALTER USER postgres PASSWORD 'postgres123!';")
        elif "could not connect to server" in str(e):
            print("\n  PostgreSQL server might not be running. Try:")
            print("  1. Start PostgreSQL service:")
            print("     - Windows: Check Services app for 'postgresql'")
            print("     - Or run: net start postgresql-x64-14 (or your version)")
        continue

    except Exception as e:
        print(f"  ✗ Unexpected error: {e}")
        continue

print("\n" + "=" * 50)
"""
Test direct connection with hardcoded password
"""
import psycopg2

# Test with hardcoded working password
try:
    print("Testing connection with 'postgres123'...")
    conn = psycopg2.connect(
        host='localhost',
        port='5432',
        database='postgres',
        user='postgres',
        password='postgres123'  # Hardcoded working password
    )
    print("SUCCESS! Connected to PostgreSQL")

    # Get version
    cursor = conn.cursor()
    cursor.execute("SELECT version();")
    version = cursor.fetchone()[0]
    print(f"PostgreSQL version: {version.split(',')[0]}")

    # Check for magnus database
    cursor.execute("SELECT 1 FROM pg_database WHERE datname='magnus';")
    magnus_exists = cursor.fetchone() is not None
    print(f"Magnus database exists: {magnus_exists}")

    cursor.close()
    conn.close()

    if not magnus_exists:
        print("Creating magnus database...")
        # Reconnect with autocommit to create database
        conn = psycopg2.connect(
            host='localhost',
            port='5432',
            database='postgres',
            user='postgres',
            password='postgres123'
        )
        conn.set_isolation_level(0)  # AUTOCOMMIT
        cursor = conn.cursor()
        cursor.execute("CREATE DATABASE magnus;")
        cursor.close()
        conn.close()
        print("Magnus database created!")

except Exception as e:
    print(f"FAILED: {e}")

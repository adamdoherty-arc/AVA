"""
Apply database schemas without TimescaleDB dependency
"""
import psycopg2
from dotenv import load_dotenv
import os
import re

load_dotenv(override=True)

# Connect to magnus database
conn = psycopg2.connect(
    host=os.getenv('DB_HOST', 'localhost'),
    port=os.getenv('DB_PORT', '5432'),
    database='magnus',
    user=os.getenv('DB_USER', 'postgres'),
    password=os.getenv('DB_PASSWORD')
)

cursor = conn.cursor()

print("Applying database schemas to 'magnus' database...")
print("=" * 70)

# First, enable uuid extension (needed for UUIDs)
print("\n[PROCESSING] Enabling required extensions...")
try:
    cursor.execute('CREATE EXTENSION IF NOT EXISTS "uuid-ossp";')
    conn.commit()
    print("[OK] uuid-ossp extension enabled")
except Exception as e:
    print(f"[WARNING] Could not enable uuid-ossp: {e}")

# Apply main schema without TimescaleDB
print("\n[PROCESSING] database_schema.sql (modified to skip TimescaleDB)...")
try:
    with open('database_schema.sql', 'r') as f:
        schema_sql = f.read()

    # Remove TimescaleDB extension
    schema_sql = re.sub(r"CREATE EXTENSION IF NOT EXISTS timescaledb;", "-- TimescaleDB skipped", schema_sql)

    # Remove any hypertable conversions (TimescaleDB specific)
    schema_sql = re.sub(r"SELECT create_hypertable\([^)]+\);", "-- Hypertable creation skipped", schema_sql)

    # Execute the modified schema
    cursor.execute(schema_sql)
    conn.commit()
    print("[OK] database_schema.sql applied successfully (without TimescaleDB)")
except Exception as e:
    print(f"[ERROR] database_schema.sql failed: {e}")
    conn.rollback()

# Try earnings schema
print("\n[PROCESSING] database_earnings_schema.sql...")
try:
    with open('database_earnings_schema.sql', 'r') as f:
        earnings_sql = f.read()

    # Split into individual statements and execute one by one
    statements = [s.strip() for s in earnings_sql.split(';') if s.strip()]

    for i, statement in enumerate(statements):
        try:
            cursor.execute(statement)
            conn.commit()
        except Exception as e:
            # Skip failing statements (like problematic indexes)
            print(f"  [SKIP] Statement {i+1}: {str(e)[:80]}")
            conn.rollback()
            continue

    print("[OK] database_earnings_schema.sql applied (with some skips)")
except Exception as e:
    print(f"[ERROR] database_earnings_schema.sql failed: {e}")
    conn.rollback()

# Check what tables were created
print("\n" + "=" * 70)
print("Checking tables in magnus database...")
cursor.execute("""
    SELECT table_name
    FROM information_schema.tables
    WHERE table_schema = 'public'
    ORDER BY table_name;
""")
tables = cursor.fetchall()

if tables:
    print(f"\n[OK] Successfully created {len(tables)} tables:")
    for table in tables:
        cursor.execute(f"SELECT COUNT(*) FROM {table[0]};")
        count = cursor.fetchone()[0]
        print(f"  - {table[0]}: {count} rows")
else:
    print("\n[WARNING] No tables found in magnus database")

cursor.close()
conn.close()

print("\n" + "=" * 70)
print("Schema application complete!")
print("=" * 70)
print("\nNext steps:")
print("1. Run 'python check_postgres.py' to verify database")
print("2. Dashboard should now be able to connect to database")

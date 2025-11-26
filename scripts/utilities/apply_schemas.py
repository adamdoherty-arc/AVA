"""
Apply database schemas to magnus database
"""
import psycopg2
from dotenv import load_dotenv
import os

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

# Schema files to apply in order
schema_files = [
    'database_schema.sql',
    'database_earnings_schema.sql',
    'database_schema_prediction_markets.sql',
]

print("Applying database schemas to 'magnus' database...")
print("=" * 70)

for schema_file in schema_files:
    if not os.path.exists(schema_file):
        print(f"[SKIP] {schema_file} - file not found")
        continue

    print(f"\n[PROCESSING] {schema_file}...")
    try:
        with open(schema_file, 'r') as f:
            schema_sql = f.read()

        # Execute the schema
        cursor.execute(schema_sql)
        conn.commit()
        print(f"[OK] {schema_file} applied successfully")
    except Exception as e:
        print(f"[ERROR] {schema_file} failed: {e}")
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

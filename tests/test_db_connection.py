"""Test database connection and check stock count"""
import psycopg2
from psycopg2.extras import RealDictCursor
from dotenv import load_dotenv
import os

load_dotenv()

# Get database config
db_config = {
    'host': os.getenv('DB_HOST', 'localhost'),
    'port': os.getenv('DB_PORT', '5432'),
    'database': os.getenv('DB_NAME', 'magnus'),
    'user': os.getenv('DB_USER', 'postgres'),
    'password': os.getenv('DB_PASSWORD', 'postgres123!')
}

print(f"Connecting to database: {db_config['database']}")
print(f"Host: {db_config['host']}")
print(f"User: {db_config['user']}")

try:
    conn = psycopg2.connect(**db_config)
    cur = conn.cursor(cursor_factory=RealDictCursor)

    # Check stock count
    cur.execute("SELECT COUNT(*) as count FROM stocks")
    count = cur.fetchone()['count']
    print(f"\nOK Stock count: {count}")

    # Check column names first
    cur.execute("SELECT column_name FROM information_schema.columns WHERE table_name = 'stocks' ORDER BY ordinal_position")
    columns = cur.fetchall()
    print("\nOK Column names in 'stocks' table:")
    for col in columns:
        print(f"  - {col['column_name']}")

    # Get sample data
    cur.execute("SELECT * FROM stocks LIMIT 5")
    samples = cur.fetchall()
    print("\nOK Sample stocks:")
    for row in samples:
        print(f"  - {row}")

    # Check table name - maybe it's 'stock' not 'stocks'?
    cur.execute("SELECT table_name FROM information_schema.tables WHERE table_schema = 'public' AND table_type = 'BASE TABLE'")
    tables = cur.fetchall()
    print("\nOK Available tables:")
    for table in tables:
        print(f"  - {table['table_name']}")

    cur.close()
    conn.close()

except Exception as e:
    print(f"\nERROR: {e}")

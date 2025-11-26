"""
Clear system environment variables and test with override
"""
import os
from dotenv import load_dotenv

# First check what's set
print("Before clearing:")
print(f"  DB_PASSWORD = {os.environ.get('DB_PASSWORD', 'NOT SET')}")
print(f"  PGPASSWORD = {os.environ.get('PGPASSWORD', 'NOT SET')}")

# Clear from current process
if 'DB_PASSWORD' in os.environ:
    del os.environ['DB_PASSWORD']
    print("\nCleared DB_PASSWORD from current process")

if 'PGPASSWORD' in os.environ:
    del os.environ['PGPASSWORD']
    print("Cleared PGPASSWORD from current process")

# Now load .env with override
print("\nLoading .env with override=True...")
load_dotenv(override=True)

print(f"  DB_PASSWORD = {os.getenv('DB_PASSWORD')}")
print(f"  PGPASSWORD = {os.getenv('PGPASSWORD')}")

# Test connection
print("\nTesting PostgreSQL connection...")
import psycopg2

try:
    conn = psycopg2.connect(
        host=os.getenv('DB_HOST', 'localhost'),
        port=os.getenv('DB_PORT', '5432'),
        database='postgres',
        user=os.getenv('DB_USER', 'postgres'),
        password=os.getenv('DB_PASSWORD')
    )
    print("SUCCESS! Connected to PostgreSQL")
    conn.close()
except Exception as e:
    print(f"FAILED: {e}")

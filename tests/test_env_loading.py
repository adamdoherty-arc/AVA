"""
Test .env file loading
"""
from dotenv import load_dotenv
import os

print("Loading .env file...")
load_dotenv()

db_password = os.getenv('DB_PASSWORD')
print(f"DB_PASSWORD from .env: '{db_password}'")
print(f"DB_HOST from .env: '{os.getenv('DB_HOST')}'")
print(f"DB_PORT from .env: '{os.getenv('DB_PORT')}'")
print(f"DB_USER from .env: '{os.getenv('DB_USER')}'")

# Now test connection with this password
import psycopg2

try:
    print(f"\nTrying to connect with password from .env...")
    conn = psycopg2.connect(
        host=os.getenv('DB_HOST', 'localhost'),
        port=os.getenv('DB_PORT', '5432'),
        database='postgres',
        user=os.getenv('DB_USER', 'postgres'),
        password=db_password
    )
    print("SUCCESS! Connected to PostgreSQL")
    conn.close()
except Exception as e:
    print(f"FAILED: {e}")

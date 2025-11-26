"""
Debug .env file reading
"""
import os

# Read .env file manually
print("Reading .env file manually...")
env_vars = {}
with open('.env', 'r') as f:
    for line in f:
        line = line.strip()
        if line and not line.startswith('#') and '=' in line:
            key, value = line.split('=', 1)
            env_vars[key] = value
            if 'PASSWORD' in key or 'DB_' in key:
                print(f"  {key} = {value}")

print("\nUsing python-dotenv:")
from dotenv import load_dotenv
load_dotenv()

print(f"  DB_PASSWORD = {os.getenv('DB_PASSWORD')}")
print(f"  PGPASSWORD = {os.getenv('PGPASSWORD')}")

print("\nChecking for environment variables already set:")
print(f"  DB_PASSWORD env var = {os.environ.get('DB_PASSWORD', 'NOT SET')}")

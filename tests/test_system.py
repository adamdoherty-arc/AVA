#!/usr/bin/env python
"""Test script to verify system components"""

import sys
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

print("="*60)
print("WHEEL STRATEGY SYSTEM - COMPONENT TEST")
print("="*60)

# Test 1: Python version
print("\n1. Python Version:")
print(f"   {sys.version}")

# Test 2: Check key packages
print("\n2. Package Installation Status:")
packages = [
    'fastapi', 'uvicorn', 'pydantic', 'pandas', 'numpy',
    'yfinance', 'requests', 'redis', 'psycopg2', 'streamlit',
    'plotly', 'loguru', 'aiohttp', 'python-dotenv'
]

installed = []
missing = []

for package in packages:
    try:
        __import__(package.replace('-', '_'))
        installed.append(package)
    except ImportError:
        missing.append(package)

print(f"   Installed ({len(installed)}): {', '.join(installed[:5])}...")
if missing:
    print(f"   Missing ({len(missing)}): {', '.join(missing)}")

# Test 3: Redis connection
print("\n3. Redis Connection:")
try:
    import redis
    r = redis.Redis(host='localhost', port=6379, db=0)
    r.ping()
    print("   STATUS: Connected")
    r.set('test_key', 'test_value')
    value = r.get('test_key')
    print(f"   Test write/read: Success")
except Exception as e:
    print(f"   STATUS: Not available - {str(e)[:50]}")

# Test 4: PostgreSQL connection
print("\n4. PostgreSQL Connection:")
try:
    import psycopg2
    conn = psycopg2.connect(
        host=os.getenv('PGHOST', 'localhost'),
        port=os.getenv('PGPORT', '5432'),
        database='wheel_strategy',
        user=os.getenv('PGUSER', 'postgres'),
        password=os.getenv('PGPASSWORD', 'postgres123!')
    )
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM information_schema.tables WHERE table_schema = 'public';")
    table_count = cursor.fetchone()[0]
    print(f"   STATUS: Connected")
    print(f"   Tables found: {table_count}")
    conn.close()
except Exception as e:
    print(f"   STATUS: Not available - {str(e)[:50]}")

# Test 5: Market data
print("\n5. Market Data Test (yfinance):")
try:
    import yfinance as yf
    ticker = yf.Ticker("AAPL")
    info = ticker.info
    price = info.get('currentPrice', info.get('regularMarketPrice', 'N/A'))
    print(f"   AAPL Current Price: ${price}")
    print("   STATUS: Working")
except Exception as e:
    print(f"   STATUS: Error - {str(e)[:50]}")

# Test 6: File structure
print("\n6. Project Structure:")
dirs = ['src', 'src/agents', 'logs', 'docs']
for dir in dirs:
    exists = os.path.exists(dir)
    print(f"   {dir}: {'EXISTS' if exists else 'MISSING'}")

# Test 7: Configuration
print("\n7. Configuration:")
if os.path.exists('config.json'):
    import json
    with open('config.json', 'r') as f:
        config = json.load(f)
    print(f"   config.json: Found")
    print(f"   Starting capital: ${config.get('starting_capital', 0):,}")
    print(f"   Max stock price: ${config.get('max_stock_price', 0)}")
else:
    print("   config.json: Not found")

if os.path.exists('.env'):
    print("   .env file: Found")
else:
    print("   .env file: Not found")

print("\n" + "="*60)
print("TEST COMPLETE")
print("="*60)

# Summary
print("\nSUMMARY:")
if not missing and (len(installed) >= 10):
    print("  [READY] All core packages installed")
else:
    print(f"  [WARNING] {len(missing)} packages missing")

print("\nTo start the system:")
print("  1. Ensure Redis is running (or install if needed)")
print("  2. Run: python start_clean.py")
print("  3. Open browser to: http://localhost:8501")
print("\nOptional: Setup PostgreSQL for full features:")
print("  Run: python setup_database.py")
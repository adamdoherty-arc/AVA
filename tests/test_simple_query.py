"""Test if raw_data->> is the issue"""
import psycopg2.extras
from datetime import datetime
from src.kalshi_db_manager import KalshiDBManager

db = KalshiDBManager()
conn = db.get_connection()
cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)

# Test 1: Simple query without ->>
print("Test 1: Simple query without ->>")
query1 = """
SELECT ticker, title
FROM kalshi_markets
WHERE title ILIKE %s AND title ILIKE %s
LIMIT 1
"""
try:
    cur.execute(query1, ('%Buffalo%', '%Houston%'))
    result = cur.fetchone()
    print(f"SUCCESS: {result['ticker'] if result else 'No result'}")
except Exception as e:
    print(f"FAILED: {e}")

# Test 2: Query with ->> but no %s after it
print("\nTest 2: Query with ->> but no %s after it")
query2 = """
SELECT ticker, title, raw_data->>'expected_expiration_time' as game_time
FROM kalshi_markets
WHERE title ILIKE %s AND title ILIKE %s
LIMIT 1
"""
try:
    cur.execute(query2, ('%Buffalo%', '%Houston%'))
    result = cur.fetchone()
    print(f"SUCCESS: {result['ticker'] if result else 'No result'}")
except Exception as e:
    print(f"FAILED: {e}")

# Test 3: Query with ->> and %s after it (the problematic one)
print("\nTest 3: Query with ->> and %s in WHERE clause")
query3 = """
SELECT ticker, title, raw_data->>'expected_expiration_time' as game_time
FROM kalshi_markets
WHERE title ILIKE %s
  AND raw_data->>'expected_expiration_time' IS NOT NULL
  AND (raw_data->>'expected_expiration_time')::timestamp >= %s
LIMIT 1
"""
try:
    params = ('%Buffalo%', datetime(2025, 11, 18))
    print(f"Params: {params}")
    cur.execute(query3, params)
    result = cur.fetchone()
    print(f"SUCCESS: {result['ticker'] if result else 'No result'}")
except Exception as e:
    print(f"FAILED: {e}")
    import traceback
    traceback.print_exc()

cur.close()
db.release_connection(conn)

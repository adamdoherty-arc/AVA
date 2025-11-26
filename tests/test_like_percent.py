"""Test if LIKE '...%' causes the issue"""
import psycopg2.extras
from datetime import datetime
from src.kalshi_db_manager import KalshiDBManager

db = KalshiDBManager()
conn = db.get_connection()
cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)

# Test with LIKE '%'  in the string
print("Test 1: Query with LIKE 'pattern%' and %s parameters")
query = """
SELECT ticker
FROM kalshi_markets
WHERE ticker LIKE 'KXNFLGAME%'
  AND title ILIKE %s
LIMIT 1
"""
try:
    cur.execute(query, ('%Buffalo%',))
    result = cur.fetchone()
    print(f"SUCCESS: {result}")
except Exception as e:
    print(f"FAILED: {e}")

# Test with multiple LIKE '%' and multiple %s
print("\nTest 2: Query with multiple LIKE 'pattern%' and multiple %s")
query2 = """
SELECT ticker
FROM kalshi_markets
WHERE (ticker LIKE 'KXNFLGAME%' OR ticker LIKE 'KXNCAAFGAME%')
  AND title ILIKE %s AND title ILIKE %s
LIMIT 1
"""
try:
    cur.execute(query2, ('%Buffalo%', '%Houston%'))
    result = cur.fetchone()
    print(f"SUCCESS: {result}")
except Exception as e:
    print(f"FAILED: {e}")

# Test with the full problematic section
print("\nTest 3: Full problematic WHERE clause")
query3 = """
SELECT ticker, title
FROM kalshi_markets
WHERE
    (title ILIKE %s AND title ILIKE %s)
    AND (
        market_type IN ('nfl', 'cfb', 'winner', 'all')
        OR raw_data->>'market_type' IN ('nfl', 'cfb', 'winner')
        OR ticker LIKE 'KXNFLGAME%'
        OR ticker LIKE 'KXNCAAFGAME%'
    )
    AND close_time >= %s
LIMIT 1
"""
try:
    cur.execute(query3, ('%Buffalo%', '%Houston%', datetime(2025, 11, 18)))
    result = cur.fetchone()
    print(f"SUCCESS: {result}")
except Exception as e:
    print(f"FAILED: {e}")
    import traceback
    traceback.print_exc()

cur.close()
db.release_connection(conn)

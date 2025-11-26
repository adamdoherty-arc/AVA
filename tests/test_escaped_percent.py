"""Test escaping % in LIKE clauses"""
import psycopg2.extras
from src.kalshi_db_manager import KalshiDBManager

db = KalshiDBManager()
conn = db.get_connection()
cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)

# Test with %% instead of %
print("Test: Query with LIKE 'pattern%%' (escaped) and %s parameters")
query = """
SELECT ticker
FROM kalshi_markets
WHERE ticker LIKE 'KXNFLGAME%%'
  AND title ILIKE %s
LIMIT 1
"""
try:
    cur.execute(query, ('%Buffalo%',))
    result = cur.fetchone()
    print(f"SUCCESS: {result['ticker'] if result else 'No result'}")
except Exception as e:
    print(f"FAILED: {e}")

cur.close()
db.release_connection(conn)

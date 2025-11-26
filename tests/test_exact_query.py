"""Test the EXACT query from the matcher"""
import psycopg2.extras
from datetime import datetime, timedelta
from src.kalshi_db_manager import KalshiDBManager

db = KalshiDBManager()
conn = db.get_connection()
cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)

# This is the EXACT query from espn_kalshi_matcher.py lines 158-194
query = """
SELECT
    ticker,
    title,
    yes_price,
    no_price,
    volume,
    close_time,
    market_type,
    raw_data->>'expected_expiration_time' as game_time_str
FROM kalshi_markets
WHERE
    (
        title ILIKE %s AND title ILIKE %s
    )
    AND (
        market_type IN ('nfl', 'cfb', 'winner', 'all')
        OR raw_data->>'market_type' IN ('nfl', 'cfb', 'winner')
        OR ticker LIKE 'KXNFLGAME%'
        OR ticker LIKE 'KXNCAAFGAME%'
    )
    AND (
        -- Match by expected_expiration_time (actual game time) if available
        (raw_data->>'expected_expiration_time' IS NOT NULL
         AND (raw_data->>'expected_expiration_time')::timestamp >= %s
         AND (raw_data->>'expected_expiration_time')::timestamp <= %s)
        OR
        -- Fallback to close_time for older data
        (raw_data->>'expected_expiration_time' IS NULL
         AND close_time >= %s
         AND close_time <= %s)
    )
    AND status != 'closed'
    AND yes_price IS NOT NULL
ORDER BY volume DESC, close_time ASC
LIMIT 1
"""

# Count placeholders manually
import re
placeholders = re.findall(r'%s', query)
print(f"Placeholders found: {len(placeholders)}")
print(f"Placeholders: {placeholders}")

params = (
    '%Buffalo%',
    '%Houston%',
    datetime(2025, 11, 18).date(),
    datetime(2025, 11, 24).date(),
    datetime(2025, 11, 18).date(),
    datetime(2025, 11, 24).date()
)

print(f"\nParameters: {len(params)}")
for i, p in enumerate(params, 1):
    print(f"  {i}. {p} ({type(p).__name__})")

try:
    cur.execute(query, params)
    result = cur.fetchone()
    if result:
        print(f"\nSUCCESS: Found {result['ticker']}")
    else:
        print("\nNo results found")
except Exception as e:
    print(f"\nFAILED: {e}")
    import traceback
    traceback.print_exc()

cur.close()
db.release_connection(conn)

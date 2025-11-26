"""Use mogrify to see the actual SQL"""
import psycopg2.extras
from datetime import datetime, timedelta
from src.kalshi_db_manager import KalshiDBManager

db = KalshiDBManager()
conn = db.get_connection()
cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)

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

params = (
    '%Buffalo%',
    '%Houston%',
    datetime(2025, 11, 18).date(),
    datetime(2025, 11, 24).date(),
    datetime(2025, 11, 18).date(),
    datetime(2025, 11, 24).date()
)

print(f"Parameters: {params}")
print(f"Parameter count: {len(params)}")

# Count %s in query
placeholder_count = query.count('%s')
print(f"Placeholder count: {placeholder_count}")

try:
    # Use mogrify to see what the final query looks like
    final_query = cur.mogrify(query, params)
    print("\nMogrified query (first 500 chars):")
    print(final_query[:500])
except Exception as e:
    print(f"\nMogrify error: {e}")
    import traceback
    traceback.print_exc()

cur.close()
db.release_connection(conn)

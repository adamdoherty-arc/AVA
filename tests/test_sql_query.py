"""Test the SQL query directly"""
import psycopg2.extras
from datetime import datetime, timedelta
from src.kalshi_db_manager import KalshiDBManager

db = KalshiDBManager()
conn = db.get_connection()
cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)

# Test with Buffalo Bills vs Houston Texans
away_var = 'Buffalo'
home_var = 'Houston'
game_time = datetime(2025, 11, 21, 1, 15)
game_date = game_time.date()
date_start = game_date - timedelta(days=3)
date_end = game_date + timedelta(days=3)

print(f"Searching for: {away_var} @ {home_var}")
print(f"Date range: {date_start} to {date_end}")
print()

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
         AND (raw_data->>'expected_expiration_time')::timestamp >= %s::timestamp
         AND (raw_data->>'expected_expiration_time')::timestamp <= %s::timestamp)
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

try:
    params = (
        f'%{away_var}%',
        f'%{home_var}%',
        date_start,
        date_end,
        date_start,
        date_end
    )
    print(f"Query params: {params}")
    print()

    cur.execute(query, params)
    result = cur.fetchone()

    if result:
        print("FOUND MATCH:")
        print(f"Ticker: {result['ticker']}")
        print(f"Title: {result['title']}")
        print(f"Prices: {result['yes_price']} / {result['no_price']}")
    else:
        print("NO MATCH FOUND")

except Exception as e:
    print(f"ERROR: {e}")
    import traceback
    traceback.print_exc()

cur.close()
db.release_connection(conn)

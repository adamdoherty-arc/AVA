"""Test Kalshi markets availability"""
from src.kalshi_db_manager import KalshiDBManager

db = KalshiDBManager()
conn = db.get_connection()
cur = conn.cursor()

# Count total markets
cur.execute("SELECT COUNT(*) FROM kalshi_markets")
total = cur.fetchone()[0]
print(f"Total Kalshi markets in database: {total}")

# Count active markets
cur.execute("SELECT COUNT(*) FROM kalshi_markets WHERE status = 'active'")
active = cur.fetchone()[0]
print(f"Active Kalshi markets: {active}")

# Show recent markets
print("\nRecent Kalshi markets:")
cur.execute("""
    SELECT ticker, title, status, market_type, yes_price, no_price
    FROM kalshi_markets
    ORDER BY synced_at DESC
    LIMIT 10
""")
for row in cur.fetchall():
    ticker, title, status, market_type, yes_price, no_price = row
    print(f"{ticker} - {title[:60]} - {status} - {market_type}")
    print(f"  Prices: Yes={yes_price}, No={no_price}")

# Check for NFL markets specifically
print("\nNFL markets:")
cur.execute("""
    SELECT ticker, title, status, yes_price, no_price
    FROM kalshi_markets
    WHERE market_type = 'nfl' OR ticker LIKE 'KXNFLGAME%'
    ORDER BY synced_at DESC
    LIMIT 5
""")
nfl_markets = cur.fetchall()
if nfl_markets:
    for row in nfl_markets:
        ticker, title, status, yes_price, no_price = row
        print(f"{ticker} - {title[:60]} - {status}")
        print(f"  Prices: Yes={yes_price}, No={no_price}")
else:
    print("No NFL markets found!")

cur.close()
db.release_connection(conn)

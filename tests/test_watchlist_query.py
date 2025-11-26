"""Test the watchlist query"""
import psycopg2
import os
from dotenv import load_dotenv

load_dotenv()

conn = psycopg2.connect(
    host=os.getenv('DB_HOST', 'localhost'),
    port=os.getenv('DB_PORT', 5432),
    database=os.getenv('DB_NAME', 'magnus'),
    user=os.getenv('DB_USER', 'postgres'),
    password=os.getenv('DB_PASSWORD', 'postgres123!')
)

cur = conn.cursor()

# Check watchlists
print("=== Checking watchlists ===")
cur.execute("SELECT name, symbol_count FROM tv_watchlists_api ORDER BY name")
watchlists = cur.fetchall()
print(f"Found {len(watchlists)} watchlists:")
for name, count in watchlists:
    print(f"  - {name}: {count} symbols")

# Get symbols for NVDA
print("\n=== Getting NVDA symbols ===")
cur.execute("""
    SELECT symbol FROM tv_symbols_api s
    JOIN tv_watchlists_api w ON s.watchlist_id = w.watchlist_id
    WHERE w.name = 'NVDA'
    LIMIT 10
""")
symbols = [row[0] for row in cur.fetchall()]
print(f"First 10 NVDA symbols: {symbols}")

# Filter to stocks only
stock_symbols = [s for s in symbols if not any(x in s for x in ['USDT', 'USD', 'BTC', '.D', 'WETH'])]
print(f"Stock symbols only: {stock_symbols}")

# Test the problematic query
print("\n=== Testing watchlist query ===")
print(f"Passing {len(stock_symbols)} stock symbols")

try:
    cur.execute("""
        WITH watchlist_symbols AS (
            SELECT unnest(%s::text[]) as symbol
        )
        SELECT
            ws.symbol,
            sd.company_name,
            sd.current_price,
            sd.price_change,
            sd.price_change_pct,
            sd.volume,
            sd.last_updated,
            (SELECT json_agg(json_build_object(
                'strike_price', strike_price,
                'premium', premium,
                'monthly_return', monthly_return,
                'dte', dte
            ))
            FROM stock_premiums
            WHERE symbol = ws.symbol AND strike_type = '5%_OTM'
            ) as premiums
        FROM watchlist_symbols ws
        LEFT JOIN stock_data sd ON ws.symbol = sd.symbol
        ORDER BY sd.current_price DESC NULLS LAST, ws.symbol
    """, (stock_symbols,))

    stocks = cur.fetchall()
    print(f"✓ Query successful! Got {len(stocks)} results")

    # Show first result
    if stocks:
        print(f"First result: {stocks[0][0]} - Price: {stocks[0][2]}")

except Exception as e:
    print(f"✗ Query failed: {e}")
    import traceback
    traceback.print_exc()

cur.close()
conn.close()

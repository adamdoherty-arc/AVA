"""Clean up fake/test watchlists from database"""

import psycopg2
import os
from dotenv import load_dotenv

load_dotenv()

# Connect to database
conn = psycopg2.connect(
    host=os.getenv('DB_HOST', 'localhost'),
    port=os.getenv('DB_PORT', 5432),
    database=os.getenv('DB_NAME', 'magnus'),
    user=os.getenv('DB_USER', 'postgres'),
    password=os.getenv('DB_PASSWORD', 'postgres123!')
)

cur = conn.cursor()

print("="*60)
print("Current Watchlists in Database")
print("="*60)

# Check tv_watchlists_api (TradingView API synced)
print("\n[TV API Watchlists]")
cur.execute("SELECT name, symbol_count FROM tv_watchlists_api ORDER BY name;")
api_lists = cur.fetchall()
for name, count in api_lists:
    print(f"  {name}: {count} symbols")

# Check tv_watchlists (old table)
print("\n[TV Manual Watchlists]")
cur.execute("SELECT DISTINCT name FROM tv_watchlists ORDER BY name;")
manual_lists = cur.fetchall()
for (name,) in manual_lists:
    cur.execute("SELECT COUNT(*) FROM tv_watchlist_symbols WHERE watchlist_id = (SELECT id FROM tv_watchlists WHERE name = %s LIMIT 1);", (name,))
    count = cur.fetchone()[0]
    print(f"  {name}: {count} symbols")

print("\n" + "="*60)
print("Cleaning Up Fake Watchlists")
print("="*60)

# Delete "Tech Giants" from both tables
fake_lists = ['Tech Giants']

for fake_name in fake_lists:
    # Delete from tv_watchlists (manual imports)
    cur.execute("DELETE FROM tv_watchlist_symbols WHERE watchlist_id IN (SELECT id FROM tv_watchlists WHERE name = %s);", (fake_name,))
    deleted_symbols = cur.rowcount
    cur.execute("DELETE FROM tv_watchlists WHERE name = %s;", (fake_name,))
    deleted_watchlists = cur.rowcount

    if deleted_watchlists > 0:
        print(f"[OK] Deleted '{fake_name}' from tv_watchlists ({deleted_symbols} symbols)")

    # Delete from tv_watchlists_api (TradingView synced) - just in case
    cur.execute("DELETE FROM tv_symbols_api WHERE watchlist_id IN (SELECT watchlist_id FROM tv_watchlists_api WHERE name = %s);", (fake_name,))
    deleted_api_symbols = cur.rowcount
    cur.execute("DELETE FROM tv_watchlists_api WHERE name = %s;", (fake_name,))
    deleted_api_watchlists = cur.rowcount

    if deleted_api_watchlists > 0:
        print(f"[OK] Deleted '{fake_name}' from tv_watchlists_api ({deleted_api_symbols} symbols)")

conn.commit()

print("\n" + "="*60)
print("Remaining Real Watchlists")
print("="*60)

# Show remaining watchlists
print("\n[Your TradingView Watchlists]")
cur.execute("SELECT name, symbol_count FROM tv_watchlists_api ORDER BY symbol_count DESC;")
real_lists = cur.fetchall()
if real_lists:
    for name, count in real_lists:
        print(f"  [OK] {name}: {count} symbols")
else:
    print("  (No watchlists synced yet - run: python src/tradingview_api_sync.py)")

cur.close()
conn.close()

print("\n" + "="*60)
print("Cleanup Complete!")
print("="*60)
print("\nRefresh your dashboard to see the changes.")

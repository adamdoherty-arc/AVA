"""
Script to sync options for all stocks in the database
"""
from src.tradingview_db_manager import TradingViewDBManager

# Get all stock symbols from database
tv_manager = TradingViewDBManager()
conn = tv_manager.get_connection()
cur = conn.cursor()

print("Fetching all stock symbols from database...")
cur.execute("SELECT ticker FROM stocks WHERE asset_type = 'STOCK' ORDER BY ticker")
all_symbols = [row[0] for row in cur.fetchall()]

print(f"Found {len(all_symbols)} stocks in database")
print(f"Sample symbols: {', '.join(all_symbols[:10])}...")

# Create AllStocks watchlist - check if it exists first
print("\nCreating 'AllStocks' watchlist in database...")
cur.execute("SELECT COUNT(*) FROM tv_watchlists_api WHERE name = 'AllStocks'")
exists = cur.fetchone()[0]

if exists:
    print("AllStocks watchlist already exists, updating...")
    cur.execute("UPDATE tv_watchlists_api SET symbols = %s WHERE name = 'AllStocks'", (all_symbols,))
else:
    print("Creating new AllStocks watchlist...")
    cur.execute("INSERT INTO tv_watchlists_api (name, symbols) VALUES ('AllStocks', %s)", (all_symbols,))

conn.commit()

print(f"✅ Created/Updated 'AllStocks' watchlist with {len(all_symbols)} symbols")
print("\nStarting sync...")

import subprocess
import os
subprocess.Popen([
    'python', 'src/watchlist_sync_service.py', 'AllStocks'
], creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0)
print("✅ Sync started in background!")

cur.close()
conn.close()

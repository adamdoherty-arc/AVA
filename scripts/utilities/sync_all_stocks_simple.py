"""
Sync options for all stocks in the database in batches
"""
from src.tradingview_db_manager import TradingViewDBManager
from src.watchlist_sync_service import WatchlistSyncService
import time

# Get all stock symbols
tv_manager = TradingViewDBManager()
conn = tv_manager.get_connection()
cur = conn.cursor()

print("Fetching all stock symbols from database...")
cur.execute("SELECT ticker FROM stocks WHERE asset_type = 'STOCK' ORDER BY ticker")
all_symbols = [row[0] for row in cur.fetchall()]
cur.close()
conn.close()

print(f"\n{'='*80}")
print(f"SYNCING OPTIONS FOR {len(all_symbols)} STOCKS")
print(f"{'='*80}\n")
print(f"This will take approximately {len(all_symbols) * 3 / 60:.0f} minutes")
print(f"Sample symbols: {', '.join(all_symbols[:10])}...\n")

# Initialize sync service
svc = WatchlistSyncService()
svc.login_robinhood()

# Sync in batches of 50
batch_size = 50
total_synced = 0
total_with_options = 0

for i in range(0, len(all_symbols), batch_size):
    batch = all_symbols[i:i+batch_size]
    batch_num = (i // batch_size) + 1
    total_batches = (len(all_symbols) + batch_size - 1) // batch_size

    print(f"\n{'='*80}")
    print(f"BATCH {batch_num}/{total_batches}: Syncing {len(batch)} stocks")
    print(f"{'='*80}")

    for symbol in batch:
        try:
            # Get stock price
            price_data = svc.polygon_client.get_stock_price(symbol)
            if price_data:
                svc.upsert_stock_data(symbol, price_data)
                print(f"  ✓ {symbol}: ${price_data['current_price']:.2f}")

                # Get options
                all_options = svc.enhanced_fetcher.get_all_expirations_data(symbol)
                if all_options:
                    for opt_data in all_options:
                        svc.upsert_options_data(opt_data)
                    print(f"    → {len(all_options)} expirations synced")
                    total_with_options += 1
                else:
                    print(f"    → No options available")

                total_synced += 1
            else:
                print(f"  ✗ {symbol}: Price data not available")

        except Exception as e:
            print(f"  ✗ {symbol}: Error - {str(e)[:50]}")
            continue

    print(f"\nProgress: {total_synced}/{len(all_symbols)} stocks synced, {total_with_options} with options")

    # Brief pause between batches
    if i + batch_size < len(all_symbols):
        time.sleep(2)

print(f"\n{'='*80}")
print(f"SYNC COMPLETE!")
print(f"{'='*80}")
print(f"Total stocks synced: {total_synced}/{len(all_symbols)}")
print(f"Stocks with options: {total_with_options}")
print(f"Success rate: {total_synced/len(all_symbols)*100:.1f}%")

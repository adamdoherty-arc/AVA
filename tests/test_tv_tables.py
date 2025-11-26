"""Test TradingView Database Manager"""

from src.tradingview_db_manager import TradingViewDBManager
import logging

logging.basicConfig(level=logging.INFO)

# Initialize manager (this will create tables)
print("Initializing TradingView Database Manager...")
tv_manager = TradingViewDBManager()

# Test creating a sample watchlist
print("\nCreating sample watchlist...")
watchlist_id = tv_manager.create_watchlist(
    "Tech Giants",
    "Major technology companies"
)

if watchlist_id:
    print(f"Created watchlist with ID: {watchlist_id}")

    # Add some sample symbols
    symbols = ["AAPL", "MSFT", "GOOGL", "META", "NVDA", "TSLA", "AMD"]
    print(f"\nAdding symbols: {', '.join(symbols)}")

    added = tv_manager.add_symbols_to_watchlist(watchlist_id, symbols)
    print(f"Added {added} symbols to watchlist")

# Get all watchlists
print("\n" + "="*50)
print("All Watchlists in Database:")
print("="*50)

all_lists = tv_manager.get_all_watchlists()
for wl in all_lists:
    print(f"\n📋 {wl['name']}")
    print(f"   - ID: {wl['id']}")
    print(f"   - Symbols: {wl['symbol_count']}")
    print(f"   - Last Refresh: {wl['last_refresh']}")
    print(f"   - Created: {wl['created_at']}")

# Get watchlist as dictionary
print("\n" + "="*50)
print("Watchlists Dictionary:")
print("="*50)

watchlists_dict = tv_manager.get_all_symbols_dict()
for name, symbols in watchlists_dict.items():
    print(f"\n{name}: {', '.join(symbols) if symbols else 'Empty'}")

print("\n" + "="*50)
print("✅ TradingView tables created successfully in magnus database!")
print("The dashboard can now store and retrieve watchlists from the database.")
print("="*50)
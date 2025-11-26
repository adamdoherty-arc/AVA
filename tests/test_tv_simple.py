"""Test TradingView fetch without database"""

from src.tradingview_db_sync import TradingViewDBSync
import os
from dotenv import load_dotenv

load_dotenv()

print("Testing TradingView Fetch (no database)")
print("=" * 50)

# Test sync
sync = TradingViewDBSync()

print("\nFetching TradingView watchlists...")
watchlists = sync.fetch_tradingview_watchlists()

if watchlists:
    print(f"Found {len(watchlists)} watchlists:")
    for name, symbols in watchlists.items():
        print(f"\n{name}: {len(symbols)} stocks")
        print(f"Symbols: {', '.join(symbols[:10])}")
else:
    print("No watchlists found")

print("\nTest complete!")
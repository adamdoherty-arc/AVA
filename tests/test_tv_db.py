"""Test TradingView Database Sync"""

from src.tradingview_db_sync import TradingViewDBSync
import os
from dotenv import load_dotenv

load_dotenv()

print("Testing TradingView Database Sync")
print("=" * 50)

# Check credentials
username = os.getenv('TRADINGVIEW_USERNAME')
password = os.getenv('TRADINGVIEW_PASSWORD')

print(f"Username: {username}")
print(f"Password: {'*' * len(password) if password else 'None'}")

# Test sync
sync = TradingViewDBSync()

print("\n1. Testing database connection...")
if sync.connect_db():
    print("✅ Connected to database")

    print("\n2. Creating tables...")
    if sync.create_tables():
        print("✅ Tables created")

    print("\n3. Fetching TradingView watchlists...")
    watchlists = sync.fetch_tradingview_watchlists()

    if watchlists:
        print(f"✅ Found {len(watchlists)} watchlists:")
        for name, symbols in watchlists.items():
            print(f"   - {name}: {len(symbols)} stocks")
            print(f"     Symbols: {', '.join(symbols[:5])}...")

        print("\n4. Syncing to database...")
        if sync.sync_to_database(watchlists):
            print("✅ Synced to database")

            print("\n5. Reading back from database...")
            db_watchlists = sync.get_watchlists_from_db()
            print(f"✅ Retrieved {len(db_watchlists)} watchlists from database")

    else:
        print("❌ No watchlists found")

    sync.disconnect_db()
else:
    print("❌ Failed to connect to database")

print("\nTest complete!")
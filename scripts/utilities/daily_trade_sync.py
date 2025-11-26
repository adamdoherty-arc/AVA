"""
Daily Trade Sync Script
Runs automatically once per day to sync trades from Robinhood to database
Schedule with Windows Task Scheduler or cron
"""

import os
import sys
from datetime import datetime
from dotenv import load_dotenv
import robin_stocks.robinhood as rh

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.trade_history_sync import TradeHistorySyncService

load_dotenv()

def main():
    """Main sync function"""
    print(f"\n{'='*60}")
    print(f"Magnus Trade Sync - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*60}\n")

    try:
        # Login to Robinhood
        print("📡 Connecting to Robinhood...")
        username = os.getenv('ROBINHOOD_USERNAME', 'brulecapital@gmail.com')
        password = os.getenv('ROBINHOOD_PASSWORD', 'FortKnox')

        rh.login(username=username, password=password)
        print("✅ Connected to Robinhood\n")

        # Initialize sync service
        print("🔄 Syncing trades from Robinhood to database...")
        sync_service = TradeHistorySyncService()

        # Sync trades
        count = sync_service.sync_trades_from_robinhood(rh)
        print(f"✅ Successfully synced {count} new trade(s)\n")

        # Get last sync time
        last_sync = sync_service.get_last_sync_time()
        if last_sync:
            print(f"📅 Last sync: {last_sync.strftime('%Y-%m-%d %I:%M %p')}")

        print(f"\n{'='*60}")
        print("✅ Daily sync completed successfully!")
        print(f"{'='*60}\n")

        return 0

    except Exception as e:
        print(f"\n{'='*60}")
        print(f"❌ ERROR: Sync failed!")
        print(f"{'='*60}")
        print(f"Error details: {e}\n")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)

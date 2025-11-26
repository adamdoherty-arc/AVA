"""Test TradingView Auto-Sync"""

from src.tradingview_scraper import TradingViewScraper
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

print("Testing TradingView Auto-Sync")
print("=" * 50)

# Check credentials
username = os.getenv('TRADINGVIEW_USERNAME')
password = os.getenv('TRADINGVIEW_PASSWORD')

if not username or not password:
    print("Error: Missing TradingView credentials in .env file")
    print("Please ensure TRADINGVIEW_USERNAME and TRADINGVIEW_PASSWORD are set")
    exit(1)

print(f"Username: {username}")
print(f"Password: {'*' * len(password)}")

# Test scraper
scraper = TradingViewScraper()
print("\nAttempting to fetch watchlists...")

watchlists = scraper.run()

if watchlists:
    print(f"\nSuccessfully fetched {len(watchlists)} watchlists:")
    for name, symbols in watchlists.items():
        print(f"  - {name}: {len(symbols)} stocks")
        if symbols:
            print(f"    Sample: {', '.join(symbols[:5])}")
else:
    print("\nFailed to fetch watchlists. Using default watchlists.")

print("\nTest complete!")
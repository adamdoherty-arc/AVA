"""Test real TradingView login"""

from src.tradingview_scraper import TradingViewScraper
import os
from dotenv import load_dotenv

load_dotenv()

print("Testing TradingView Real Login")
print("=" * 50)

username = os.getenv('TRADINGVIEW_USERNAME')
password = os.getenv('TRADINGVIEW_PASSWORD')

print(f"Username: {username}")
print(f"Password: {'*' * len(password) if password else 'None'}")

scraper = TradingViewScraper()

# Try to setup driver and login
print("\nSetting up Chrome driver...")
try:
    scraper.setup_driver()
    print("Driver setup successful")

    print("\nAttempting login...")
    if scraper.login():
        print("Login successful!")

        # Try to get watchlists
        watchlists = scraper.get_watchlists()
        print(f"\nFound {len(watchlists)} watchlists:")
        for name, symbols in watchlists.items():
            print(f"  - {name}: {symbols[:5] if symbols else 'empty'}")
    else:
        print("Login failed - checking fallback")

except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()

finally:
    if scraper.driver:
        scraper.driver.quit()
#!/usr/bin/env python
"""Test simple Robinhood integration with session caching"""

from src.robinhood_simple import *
from dotenv import load_dotenv
import os
from pathlib import Path

# Load environment variables
load_dotenv()

print("="*60)
print("TESTING SIMPLE ROBINHOOD WITH SESSION CACHING")
print("="*60)

# Check if cached session exists
pickle_path = Path.home() / '.robinhood_token.pickle'
if pickle_path.exists():
    print("\nCached session found at:", pickle_path)
    print("This should connect WITHOUT requiring MFA")
else:
    print("\nNo cached session found")
    print("First login will require MFA")

print("\nAttempting login...")
print("-"*40)

# Login (will use cached session if available)
if login_robinhood():
    print("\nSUCCESS: Logged in!")

    # Get account summary
    account = get_account_summary()
    if account:
        print(f"\nAccount Summary:")
        print(f"  Buying Power: ${account.get('buying_power', 0):,.2f}")
        print(f"  Portfolio Value: ${account.get('portfolio_value', 0):,.2f}")
        print(f"  Cash: ${account.get('cash', 0):,.2f}")

    # Get positions
    stocks = get_positions()
    options = get_options()

    print(f"\nPositions:")
    print(f"  Stocks: {len(stocks)}")
    print(f"  Options: {len(options)}")

    # Identify wheel positions
    wheel = identify_wheel_positions(stocks, options)
    print(f"\nWheel Strategy Positions: {len(wheel)}")

    for pos in wheel[:5]:  # Show first 5
        print(f"  {pos['strategy']}: {pos['symbol']}")

    # Logout
    import robin_stocks.robinhood as rh
    rh.authentication.logout()
    print("\nLogged out.")

else:
    print("\nFAILED: Could not login")
    print("\nCheck your .env file has:")
    print("  ROBINHOOD_USERNAME=your_username")
    print("  ROBINHOOD_PASSWORD=your_password")

print("\n" + "="*60)
print("TEST COMPLETE")
print("="*60)
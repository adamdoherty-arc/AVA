#!/usr/bin/env python
"""Test the fixed Robinhood integration"""

from src.robinhood_fixed import *
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

print("TESTING FIXED ROBINHOOD INTEGRATION")
print("="*60)

if login_robinhood():
    print("\n1. Account Summary:")
    account = get_account_summary()
    print(f"   Buying Power: ${account.get('buying_power', 0):,.2f}")
    print(f"   Cash: ${account.get('cash', 0):,.2f}")

    print("\n2. Stock Positions:")
    stocks = get_positions()
    print(f"   Found {len(stocks)} stocks")

    print("\n3. Option Positions:")
    options = get_options()
    print(f"   Found {len(options)} options")
    for opt in options:
        print(f"   - {opt['symbol']} {opt['option_type']} {opt['side']}")

    print("\n4. Wheel Positions:")
    wheel = identify_wheel_positions(stocks, options)
    print(f"   Found {len(wheel)} wheel positions")
    for pos in wheel:
        print(f"   - {pos['strategy']}: {pos['symbol']}")

    # Logout
    import robin_stocks.robinhood as rh
    rh.authentication.logout()
    print("\nDone!")
else:
    print("Login failed")
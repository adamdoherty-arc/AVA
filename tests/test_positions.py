#!/usr/bin/env python
"""Test Robinhood positions retrieval"""

from src.robinhood_rate_limited import *
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

print("="*60)
print("TESTING ROBINHOOD POSITIONS")
print("="*60)

print("\n1. Attempting login...")
if login_robinhood():
    print("[OK] Login successful!")

    print("\n2. Getting account summary...")
    account = get_account_summary()
    if account:
        print(f"   Buying Power: ${account.get('buying_power', 0):,.2f}")
        print(f"   Portfolio Value: ${account.get('portfolio_value', 0):,.2f}")
    else:
        print("   [X] No account data returned")

    print("\n3. Getting stock positions...")
    stocks = get_positions()
    if stocks:
        print(f"   Found {len(stocks)} stock positions:")
        for stock in stocks[:5]:  # Show first 5
            print(f"   - {stock['symbol']}: {stock['quantity']} shares @ ${stock.get('avg_cost', 0):.2f}")
    else:
        print("   [X] No stock positions found (empty list or error)")

    print("\n4. Getting option positions...")
    options = get_options()
    if options:
        print(f"   Found {len(options)} option positions:")
        for opt in options[:5]:  # Show first 5
            print(f"   - {opt['symbol']} {opt['option_type']} {opt['strike']} exp {opt['expiration']}")
    else:
        print("   [X] No option positions found (empty list or error)")

    print("\n5. Identifying wheel positions...")
    wheel = identify_wheel_positions(stocks, options)
    if wheel:
        print(f"   Found {len(wheel)} wheel positions:")
        for pos in wheel[:5]:
            print(f"   - {pos['strategy']}: {pos['symbol']}")
    else:
        print("   [X] No wheel positions identified")

    # Logout
    import robin_stocks.robinhood as rh
    rh.authentication.logout()
    print("\n[OK] Logged out")
else:
    print("[X] Login failed")

print("\n" + "="*60)
print("If positions aren't showing:")
print("1. Check if you have any open positions in Robinhood")
print("2. Try the Robinhood app to verify positions exist")
print("3. Check for any API errors above")
print("="*60)
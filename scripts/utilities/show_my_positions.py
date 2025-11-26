#!/usr/bin/env python
"""Show your actual Robinhood positions in a simple format"""

from src.robinhood_rate_limited import *
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

print("="*60)
print("YOUR ROBINHOOD POSITIONS")
print("="*60)

# Login
print("\nLogging in...")
if login_robinhood():
    print("[OK] Logged in successfully!")

    # Get account
    print("\n--- ACCOUNT SUMMARY ---")
    account = get_account_summary()
    print(f"Buying Power: ${account.get('buying_power', 0):,.2f}")
    print(f"Portfolio Value: ${account.get('portfolio_value', 0):,.2f}")
    print(f"Cash: ${account.get('cash', 0):,.2f}")

    # Get stocks
    print("\n--- STOCK POSITIONS ---")
    stocks = get_positions()
    if stocks:
        for stock in stocks:
            print(f"{stock['symbol']}: {stock['quantity']} shares @ ${stock.get('avg_cost', 0):.2f}")
    else:
        print("No stock positions")

    # Get options
    print("\n--- OPTION POSITIONS ---")
    options = get_options()
    if options:
        for opt in options:
            side = "SOLD" if opt.get('side') == 'short' else "BOUGHT"
            print(f"{opt['symbol']} {opt.get('option_type', '?').upper()} ${opt.get('strike', 0)} exp {opt.get('expiration', '?')} - {side}")
            print(f"  Premium: ${opt.get('avg_price', 0) * 100:.2f} per contract")
    else:
        print("No option positions")

    # Identify wheel positions
    print("\n--- WHEEL STRATEGY POSITIONS ---")
    wheel = identify_wheel_positions(stocks, options)
    if wheel:
        for pos in wheel:
            if pos['strategy'] == 'CSP':
                print(f"Cash-Secured Put: {pos['symbol']} ${pos.get('strike', 0)} exp {pos.get('expiration', '?')}")
            elif pos['strategy'] == 'CC':
                print(f"Covered Call: {pos['symbol']} ${pos.get('strike', 0)} exp {pos.get('expiration', '?')}")
            elif pos['strategy'] == 'Potential CC':
                print(f"Stock for Covered Calls: {pos['symbol']} - {pos.get('contracts_available', 0)} contracts available")
    else:
        print("No wheel positions identified")

    # Logout
    import robin_stocks.robinhood as rh
    rh.authentication.logout()
    print("\n[OK] Done")

else:
    print("[ERROR] Could not log in")
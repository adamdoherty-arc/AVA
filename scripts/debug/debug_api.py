#!/usr/bin/env python
"""Debug Robinhood API responses"""

import robin_stocks.robinhood as rh
import os
from dotenv import load_dotenv
import json

# Load environment variables
load_dotenv()

print("DEBUGGING ROBINHOOD API")
print("="*60)

username = os.getenv('ROBINHOOD_USERNAME')
password = os.getenv('ROBINHOOD_PASSWORD')

# Login
login = rh.authentication.login(
    username=username,
    password=password,
    store_session=True
)

if login:
    print("Login successful!")

    # Test stock positions
    print("\n1. Testing stock positions API:")
    try:
        stocks = rh.account.get_open_stock_positions()
        print(f"   Type: {type(stocks)}")
        if stocks:
            print(f"   Count: {len(stocks)}")
            if len(stocks) > 0:
                print(f"   First item type: {type(stocks[0])}")
                print(f"   First item keys: {stocks[0].keys() if isinstance(stocks[0], dict) else 'Not a dict'}")
        else:
            print("   Empty or None")
    except Exception as e:
        print(f"   Error: {e}")

    # Test option positions
    print("\n2. Testing option positions API:")
    try:
        options = rh.options.get_open_option_positions()
        print(f"   Type: {type(options)}")
        if options:
            print(f"   Count: {len(options)}")
            if len(options) > 0:
                print(f"   First item type: {type(options[0])}")
                if isinstance(options[0], dict):
                    print(f"   First item keys: {list(options[0].keys())[:5]}...")
                else:
                    print(f"   First item: {options[0]}")
        else:
            print("   Empty or None")
    except Exception as e:
        print(f"   Error: {e}")

    # Test all positions
    print("\n3. Testing all positions:")
    try:
        all_pos = rh.account.get_all_positions()
        print(f"   Type: {type(all_pos)}")
        if all_pos:
            print(f"   Count: {len(all_pos)}")
    except Exception as e:
        print(f"   Error: {e}")

    # Logout
    rh.authentication.logout()
    print("\nLogged out")
else:
    print("Login failed")
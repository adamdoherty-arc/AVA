#!/usr/bin/env python
"""Debug option details to fix type detection"""

import robin_stocks.robinhood as rh
import os
from dotenv import load_dotenv

load_dotenv()

username = os.getenv('ROBINHOOD_USERNAME')
password = os.getenv('ROBINHOOD_PASSWORD')

login = rh.authentication.login(username=username, password=password, store_session=True)

if login:
    print("Debugging Option Details")
    print("="*60)

    options = rh.options.get_open_option_positions()

    # Just check first option in detail
    if options and len(options) > 0:
        opt = options[0]
        print("First option raw data:")
        print(f"  chain_symbol: {opt.get('chain_symbol')}")
        print(f"  type: {opt.get('type')}")
        print(f"  option_id: {opt.get('option_id')}")

        # Get the option instrument details
        if opt.get('option'):
            option_url = opt.get('option')
            print(f"\nFetching option details from URL: {option_url}")
            try:
                option_details = rh.helper.request_get(option_url)
                print(f"  Option type from URL: {option_details.get('type')}")
                print(f"  Strike: ${option_details.get('strike_price')}")
                print(f"  Expiration: {option_details.get('expiration_date')}")
            except Exception as e:
                print(f"  Error: {e}")

        # Also try the option_id method
        if opt.get('option_id'):
            print(f"\nUsing option_id: {opt['option_id']}")
            try:
                market_data = rh.options.get_option_market_data_by_id(opt['option_id'])
                print(f"  Market data keys: {list(market_data.keys())[:10]}")

                # Try to get the instrument
                instrument = rh.options.get_option_instrument_by_id(opt['option_id'])
                print(f"  Instrument type: {instrument.get('type')}")
            except Exception as e:
                print(f"  Error: {e}")

    rh.authentication.logout()
else:
    print("Login failed")
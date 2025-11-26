#!/usr/bin/env python
"""Show actual option positions"""

import robin_stocks.robinhood as rh
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

username = os.getenv('ROBINHOOD_USERNAME')
password = os.getenv('ROBINHOOD_PASSWORD')

# Login
login = rh.authentication.login(username=username, password=password, store_session=True)

if login:
    print("YOUR OPTION POSITIONS:")
    print("="*60)

    options = rh.options.get_open_option_positions()

    for i, opt in enumerate(options, 1):
        if float(opt.get('quantity', 0)) != 0:  # Only show non-zero positions
            print(f"\nOption #{i}:")
            print(f"  Symbol: {opt.get('chain_symbol', 'N/A')}")
            print(f"  Type: {opt.get('type', 'N/A')}")
            print(f"  Quantity: {opt.get('quantity', 'N/A')}")
            print(f"  Average Price: ${float(opt.get('average_price', 0)) / 100:.2f}")
            print(f"  Option ID: {opt.get('option_id', 'N/A')}")

            # Get more details if we have option_id
            if opt.get('option_id'):
                try:
                    details = rh.options.get_option_market_data_by_id(opt['option_id'])
                    if details:
                        print(f"  Strike: ${float(details.get('strike_price', 0)):.2f}")
                        print(f"  Expiration: {details.get('expiration_date', 'N/A')}")
                        print(f"  Current Price: ${float(details.get('mark_price', 0)):.2f}")
                except:
                    pass

    # Check for any stocks that might be from assigned options
    print("\n" + "="*60)
    print("CHECKING FOR STOCKS (might be from assigned options):")
    print("="*60)

    all_positions = rh.account.get_all_positions()
    stock_count = 0

    for pos in all_positions:
        if float(pos.get('quantity', 0)) > 0:
            stock_count += 1
            # Get symbol
            try:
                instrument = rh.stocks.get_instrument_by_url(pos['instrument'])
                symbol = instrument.get('symbol', 'Unknown')
                print(f"\n{symbol}:")
                print(f"  Shares: {pos['quantity']}")
                print(f"  Avg Cost: ${pos.get('average_buy_price', 'N/A')}")
            except:
                pass

    if stock_count == 0:
        print("\nNo stock positions found")

    rh.authentication.logout()

else:
    print("Login failed")
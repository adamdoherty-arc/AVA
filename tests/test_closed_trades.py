"""
Test script to check Robinhood closed option orders
Debug why trade history isn't showing
"""

import robin_stocks.robinhood as rh
import json
from datetime import datetime

print("=" * 60)
print("ROBINHOOD CLOSED TRADES TEST")
print("=" * 60)

# Login
print("\n1. Logging into Robinhood...")
try:
    rh.login(username='brulecapital@gmail.com', password='FortKnox')
    print("[OK] Login successful")
except Exception as e:
    print(f"[ERROR] Login failed: {e}")
    exit(1)

# Get all option orders
print("\n2. Fetching all option orders...")
try:
    all_orders = rh.get_all_option_orders()
    print(f"[OK] Found {len(all_orders)} total option orders")

    # Analyze order states
    states = {}
    for order in all_orders:
        state = order.get('state', 'unknown')
        states[state] = states.get(state, 0) + 1

    print(f"\nOrder states breakdown:")
    for state, count in states.items():
        print(f"  {state}: {count}")

except Exception as e:
    print(f"[ERROR] Failed to get orders: {e}")
    exit(1)

# Filter for filled closing orders
print("\n3. Analyzing filled orders...")
filled_orders = [o for o in all_orders if o.get('state') == 'filled']
print(f"Found {len(filled_orders)} filled orders")

if filled_orders:
    print("\n4. Checking first 5 filled orders for closing trades...")

    closing_trades = []
    for idx, order in enumerate(filled_orders[:10]):  # Check first 10
        print(f"\n--- Order {idx+1} ---")
        print(f"  State: {order.get('state')}")
        print(f"  Direction: {order.get('direction')}")  # 'debit' or 'credit'
        print(f"  Opening/Closing: {order.get('opening_strategy')}, {order.get('closing_strategy')}")
        print(f"  Created: {order.get('created_at')}")
        print(f"  Updated: {order.get('updated_at')}")

        legs = order.get('legs', [])
        if legs:
            leg = legs[0]
            print(f"  Leg side: {leg.get('side')}")  # 'buy' or 'sell'
            print(f"  Position effect: {leg.get('position_effect')}")  # 'open' or 'close'

            # Try to get option details
            opt_url = leg.get('option')
            if opt_url:
                try:
                    opt_data = rh.get_option_instrument_data(opt_url)
                    if opt_data:
                        symbol = opt_data.get('chain_symbol', 'Unknown')
                        strike = opt_data.get('strike_price')
                        exp_date = opt_data.get('expiration_date')
                        opt_type = opt_data.get('type')
                        print(f"  Option: {symbol} ${strike} {opt_type} exp {exp_date}")

                        # Is this a closing trade?
                        if leg.get('position_effect') == 'close':
                            closing_trades.append({
                                'symbol': symbol,
                                'strike': strike,
                                'exp_date': exp_date,
                                'opt_type': opt_type,
                                'side': leg.get('side'),
                                'quantity': order.get('quantity'),
                                'price': order.get('average_price'),
                                'close_date': order.get('updated_at')
                            })
                            print(f"  [***] THIS IS A CLOSING TRADE!")
                except Exception as e:
                    print(f"  [WARNING] Could not fetch option data: {e}")

    print(f"\n5. SUMMARY:")
    print(f"Total filled orders: {len(filled_orders)}")
    print(f"Closing trades found: {len(closing_trades)}")

    if closing_trades:
        print(f"\nClosed Trades:")
        for trade in closing_trades:
            print(f"  - {trade['symbol']} ${trade['strike']} {trade['opt_type']} ({trade['close_date'][:10]})")
    else:
        print("\n[WARNING] No closing trades found in first 10 orders!")
        print("This could mean:")
        print("  1. All your closed trades were opened/closed more than 10 orders ago")
        print("  2. The position_effect field might not be 'close'")
        print("  3. Robinhood uses different fields to identify closing trades")

else:
    print("[WARNING] No filled orders found at all!")

print("\n" + "=" * 60)
print("TEST COMPLETE")
print("=" * 60)

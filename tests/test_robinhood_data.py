"""
Test script to verify Robinhood data retrieval
Checks:
- Account balance
- Buying power
- Open option positions (especially DKNG)
- Premium calculations
"""

import robin_stocks.robinhood as rh

print("=" * 60)
print("ROBINHOOD DATA TEST")
print("=" * 60)

# Login
print("\n1. Logging into Robinhood...")
try:
    login = rh.login(username='brulecapital@gmail.com', password='FortKnox')
    print("[OK] Login successful")
except Exception as e:
    print(f"[ERROR] Login failed: {e}")
    exit(1)

# Get account data
print("\n2. Fetching account data...")
try:
    account_data = rh.load_account_profile()
    portfolio_data = rh.build_holdings()

    # Get portfolio value
    portfolio_value = rh.profiles.load_portfolio_profile()

    print(f"Account Data Keys: {list(account_data.keys())}")
    print(f"Portfolio Value Keys: {list(portfolio_value.keys()) if portfolio_value else 'None'}")

    if portfolio_value:
        equity = float(portfolio_value.get('equity', 0))
        equity_previous = float(portfolio_value.get('equity_previous_close', 0))
        print(f"\n📊 Portfolio Value: ${equity:,.2f}")
        print(f"📊 Previous Close: ${equity_previous:,.2f}")
        print(f"📊 Change: ${equity - equity_previous:,.2f}")

    # Buying power
    buying_power_data = rh.load_account_profile()
    if 'buying_power' in buying_power_data:
        buying_power = float(buying_power_data.get('buying_power', 0))
        print(f"💵 Buying Power: ${buying_power:,.2f}")

except Exception as e:
    print(f"[ERROR] Error fetching account: {e}")

# Get option positions
print("\n3. Fetching option positions...")
try:
    positions = rh.get_open_option_positions()

    if not positions:
        print("[WARNING] No open option positions found")
    else:
        print(f"[OK] Found {len(positions)} open option positions\n")

        for idx, pos in enumerate(positions, 1):
            print(f"\n--- Position {idx} ---")

            # Get option details
            opt_id = pos.get('option_id')
            if not opt_id:
                print("  [WARNING] No option_id")
                continue

            opt_data = rh.get_option_instrument_data_by_id(opt_id)
            symbol = opt_data.get('chain_symbol', 'Unknown')
            strike = float(opt_data.get('strike_price', 0))
            exp_date = opt_data.get('expiration_date', 'Unknown')
            opt_type = opt_data.get('type', 'unknown')

            # Position details
            position_type = pos.get('type', 'unknown')
            quantity = float(pos.get('quantity', 0))
            avg_price = abs(float(pos.get('average_price', 0)))

            print(f"  Symbol: {symbol}")
            print(f"  Type: {opt_type.upper()} - {position_type.upper()}")
            print(f"  Strike: ${strike:.2f}")
            print(f"  Expiration: {exp_date}")
            print(f"  Contracts: {int(quantity)}")
            print(f"  Avg Price (raw): {avg_price}")

            # Calculate premium
            # Robinhood's average_price is per contract (not per share)
            total_premium = avg_price * quantity
            print(f"  Total Premium: ${total_premium:.2f}")

            # Get current market price
            market_data = rh.get_option_market_data_by_id(opt_id)
            if market_data and len(market_data) > 0:
                # adjusted_mark_price is per share, need to multiply by 100
                mark_price = float(market_data[0].get('adjusted_mark_price', 0))
                current_price_per_contract = mark_price * 100
                current_value = current_price_per_contract * quantity

                print(f"  Mark Price (per share): ${mark_price:.2f}")
                print(f"  Current Price (per contract): ${current_price_per_contract:.2f}")
                print(f"  Current Value: ${current_value:.2f}")

                # Calculate P/L
                if position_type == 'short':
                    pl = total_premium - current_value
                else:
                    pl = current_value - total_premium

                print(f"  P/L: ${pl:.2f} ({(pl/total_premium*100):.1f}%)")

            # Identify DKNG specifically
            if symbol == 'DKNG':
                print("  [***] FOUND DKNG POSITION!")

except Exception as e:
    print(f"[ERROR] Error fetching positions: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 60)
print("TEST COMPLETE")
print("=" * 60)

"""Test Robinhood Options API"""
import os
from dotenv import load_dotenv
import robin_stocks.robinhood as rh

load_dotenv()

print("="*60)
print("Testing Robinhood Options API")
print("="*60)

# Login
print("\n[1] Logging into Robinhood...")
username = os.getenv('ROBINHOOD_USERNAME')
password = os.getenv('ROBINHOOD_PASSWORD')

try:
    login = rh.authentication.login(
        username=username,
        password=password,
        expiresIn=86400,
        store_session=True
    )
    print("[OK] Logged in successfully")
except Exception as e:
    print(f"[FAIL] Login failed: {e}")
    print("\nRobinhood requires 2FA. You'll need to complete it manually.")
    exit(1)

# Test getting options for AAPL
print("\n[2] Getting options chain for AAPL...")
symbol = "AAPL"

try:
    # Get chains
    chains = rh.options.get_chains(symbol)
    if chains:
        print(f"[OK] Got chains data")
        # chains is a dict with expiration_dates
        exp_dates = chains.get('expiration_dates', [])
        print(f"[OK] Available expirations: {len(exp_dates)}")
        print(f"     First 5: {exp_dates[:5]}")
    else:
        print("[FAIL] No chains found")
        exit(1)

    # Get current price
    quote = rh.stocks.get_stock_quote_by_symbol(symbol)
    current_price = float(quote.get('last_trade_price', 0))
    print(f"[OK] Current price: ${current_price:.2f}")

    # Get put options for nearest expiration
    nearest_exp = exp_dates[10] if len(exp_dates) > 10 else exp_dates[0]
    print(f"\n[3] Getting puts for expiration: {nearest_exp}...")

    puts = rh.options.find_options_by_expiration(
        symbol,
        nearest_exp,
        optionType='put'
    )

    if puts:
        print(f"[OK] Found {len(puts)} put options")

        # Find 5% OTM
        target_strike = current_price * 0.95
        closest_put = min(puts, key=lambda x: abs(float(x.get('strike_price', 0)) - target_strike))

        strike_price = float(closest_put.get('strike_price', 0))
        print(f"\n[4] Closest to 5% OTM (${target_strike:.2f}): ${strike_price:.2f}")

        # Get market data for this option
        option_id = closest_put['id']
        market_data = rh.options.get_option_market_data_by_id(option_id)

        if market_data:
            data = market_data[0]
            bid = float(data.get('bid_price', 0))
            ask = float(data.get('ask_price', 0))
            mid = (bid + ask) / 2
            premium = mid * 100
            volume = int(data.get('volume', 0))
            oi = int(data.get('open_interest', 0))
            iv = float(data.get('implied_volatility', 0))

            print(f"     Bid: ${bid:.2f}")
            print(f"     Ask: ${ask:.2f}")
            print(f"     Mid: ${mid:.2f}")
            print(f"     Premium: ${premium:.2f}")
            print(f"     Volume: {volume}")
            print(f"     Open Interest: {oi}")
            print(f"     IV: {iv*100:.1f}%")

            # Calculate returns
            capital = strike_price * 100
            premium_pct = (premium / capital * 100) if capital > 0 else 0

            from datetime import datetime
            exp_date = datetime.strptime(nearest_exp, '%Y-%m-%d')
            dte = (exp_date - datetime.now()).days
            monthly_return = (premium_pct / dte * 30) if dte > 0 else 0
            annual_return = (premium_pct / dte * 365) if dte > 0 else 0

            print(f"\n[5] Returns Calculation:")
            print(f"     DTE: {dte} days")
            print(f"     Premium %: {premium_pct:.2f}%")
            print(f"     Monthly Return: {monthly_return:.2f}%")
            print(f"     Annual Return: {annual_return:.2f}%")

            print(f"\n[SUCCESS] Robinhood options API works!")

        else:
            print("[FAIL] No market data found")

    else:
        print("[FAIL] No puts found")

except Exception as e:
    print(f"[FAIL] Error: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "="*60)

"""Diagnostic script to identify positions page issues"""
import os
from dotenv import load_dotenv
import robin_stocks.robinhood as rh

load_dotenv()

# Test 1: Check Robinhood credentials
print("=== TEST 1: Robinhood Credentials ===")
username = os.getenv('ROBINHOOD_USERNAME')
password = os.getenv('ROBINHOOD_PASSWORD')
print(f"Username configured: {bool(username)}")
print(f"Password configured: {bool(password)}")

if not username or not password:
    print("ERROR: Robinhood credentials not configured")
    exit(1)

# Test 2: Try to login
print("\n=== TEST 2: Robinhood Login ===")
try:
    rh.logout()
except:
    pass

try:
    login_result = rh.login(username=username, password=password, store_session=True)
    print("Login successful!")

    # Verify session
    profile = rh.profiles.load_account_profile()
    if profile:
        print(f"Account verified: {profile.get('account_number', 'Unknown')}")
    else:
        print("ERROR: Login succeeded but could not load profile")
except Exception as e:
    print(f"ERROR: Login failed: {e}")
    exit(1)

# Test 3: Get stock positions
print("\n=== TEST 3: Stock Positions ===")
try:
    stock_positions = rh.get_open_stock_positions()
    print(f"API returned {len(stock_positions) if stock_positions else 0} stock positions")

    # Filter for quantity > 0
    active_stocks = [s for s in stock_positions if float(s.get('quantity', 0)) > 0]
    print(f"Active stock positions (quantity > 0): {len(active_stocks)}")

    if active_stocks:
        print("\nStock positions found:")
        for pos in active_stocks:
            quantity = float(pos.get('quantity', 0))
            instrument_url = pos.get('instrument')

            if instrument_url:
                instrument_data = rh.get_instrument_by_url(instrument_url)
                symbol = instrument_data.get('symbol', 'Unknown')
                print(f"  - {symbol}: {quantity} shares")
    else:
        print("No active stock positions found")

except Exception as e:
    print(f"ERROR getting stock positions: {e}")
    import traceback
    traceback.print_exc()

# Test 4: Get option positions
print("\n=== TEST 4: Option Positions ===")
try:
    option_positions = rh.get_open_option_positions()
    print(f"API returned {len(option_positions) if option_positions else 0} option positions")

    if option_positions:
        print("\nOption positions found:")
        for pos in option_positions[:5]:  # Show first 5
            opt_id = pos.get('option_id')
            if opt_id:
                opt_data = rh.get_option_instrument_data_by_id(opt_id)
                symbol = opt_data.get('chain_symbol', 'Unknown')
                strike = opt_data.get('strike_price', 0)
                exp_date = opt_data.get('expiration_date', 'Unknown')
                opt_type = opt_data.get('type', 'unknown')
                quantity = pos.get('quantity', 0)
                pos_type = pos.get('type', 'unknown')

                print(f"  - {symbol} ${strike} {opt_type.upper()} exp {exp_date} ({pos_type}, qty: {quantity})")
    else:
        print("No option positions found")

except Exception as e:
    print(f"ERROR getting option positions: {e}")
    import traceback
    traceback.print_exc()

# Test 5: Portfolio summary
print("\n=== TEST 5: Portfolio Summary ===")
try:
    portfolio = rh.profiles.load_portfolio_profile()
    if portfolio:
        equity = float(portfolio.get('equity', 0))
        print(f"Total equity: ${equity:,.2f}")
    else:
        print("ERROR: Could not load portfolio profile")
except Exception as e:
    print(f"ERROR: {e}")

print("\n=== DIAGNOSTIC COMPLETE ===")

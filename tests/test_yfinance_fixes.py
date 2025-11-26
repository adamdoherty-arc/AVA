"""
Test script for yfinance error handling fixes

Tests safe wrapper functions with delisted and valid symbols.
"""

from src.yfinance_utils import (
    safe_get_current_price,
    safe_get_info,
    safe_get_history,
    safe_get_options_expirations,
    is_symbol_delisted,
    get_delisted_symbols,
    add_delisted_symbol
)

print("=" * 70)
print("YFinance Error Handling Test Suite")
print("=" * 70)

# Test 1: Delisted symbols (should not show ERROR logs)
print("\n[TEST 1] Testing delisted symbols (should show INFO/WARNING, not ERROR):")
print("-" * 70)
delisted_symbols = ['BMNR', 'PLUG', 'BBAI']

for symbol in delisted_symbols:
    print(f"\nTesting {symbol}:")

    # Check if marked as delisted
    is_delisted = is_symbol_delisted(symbol)
    print(f"  Is delisted: {is_delisted}")

    # Try to get price (should handle gracefully)
    price = safe_get_current_price(symbol, suppress_warnings=False)
    print(f"  Current price: ${price if price else 'N/A (delisted)'}")

    # Try to get info
    info = safe_get_info(symbol, suppress_warnings=True)
    print(f"  Info available: {'Yes' if info else 'No (expected for delisted)'}")


# Test 2: Valid symbols (should work normally)
print("\n\n[TEST 2] Testing valid symbols (should work normally):")
print("-" * 70)
valid_symbols = ['AAPL', 'MSFT', 'GOOGL']

for symbol in valid_symbols:
    print(f"\nTesting {symbol}:")

    price = safe_get_current_price(symbol)
    print(f"  Current price: ${price:.2f}" if price else "  ERROR: Could not get price")

    info = safe_get_info(symbol, suppress_warnings=True)
    if info:
        print(f"  Company: {info.get('longName', 'N/A')}")
        print(f"  Market Cap: ${info.get('marketCap', 0):,}")

    hist = safe_get_history(symbol, period='5d', suppress_warnings=True)
    if hist is not None and not hist.empty:
        print(f"  5-day data points: {len(hist)}")


# Test 3: Options data for valid symbol
print("\n\n[TEST 3] Testing options data (valid symbol):")
print("-" * 70)
symbol = 'AAPL'
print(f"Testing options for {symbol}:")

expirations = safe_get_options_expirations(symbol, suppress_warnings=True)
if expirations:
    print(f"  Available expirations: {len(expirations)}")
    print(f"  First 3 dates: {expirations[:3]}")
else:
    print(f"  No options available (unexpected for {symbol})")


# Test 4: Check delisted symbols list
print("\n\n[TEST 4] Checking known delisted symbols:")
print("-" * 70)
delisted = get_delisted_symbols()
print(f"Known delisted symbols: {delisted}")
print(f"Total count: {len(delisted)}")


# Test 5: Suppress warnings mode
print("\n\n[TEST 5] Testing suppress_warnings mode:")
print("-" * 70)
print("Getting prices for delisted symbols with warnings suppressed...")
print("(Should not see any WARNING logs, only INFO)")

for symbol in ['BMNR', 'PLUG']:
    price = safe_get_current_price(symbol, suppress_warnings=True)
    print(f"  {symbol}: {'${:.2f}'.format(price) if price else 'N/A'}")


# Test 6: Error handling with retry
print("\n\n[TEST 6] Testing retry logic with fallback periods:")
print("-" * 70)
symbol = 'AAPL'
print(f"Getting history for {symbol} with retry periods...")

hist = safe_get_history(symbol, period='1d', retry_periods=['5d', '1mo'], suppress_warnings=False)
if hist is not None and not hist.empty:
    print(f"  Success: Retrieved {len(hist)} data points")
    print(f"  Latest close: ${hist['Close'].iloc[-1]:.2f}")
else:
    print(f"  FAILED: Could not retrieve data")


# Summary
print("\n\n" + "=" * 70)
print("TEST SUMMARY")
print("=" * 70)
print("\nExpected Results:")
print("  1. No ERROR logs for delisted symbols (BMNR, PLUG, BBAI)")
print("  2. Valid symbols return proper data (AAPL, MSFT, GOOGL)")
print("  3. Options data available for valid symbols")
print("  4. Delisted symbols list populated")
print("  5. Suppress mode works (no WARNING logs in test 5)")
print("  6. Retry logic works for valid symbols")
print("\nIf you see:")
print("  - INFO logs for delisted symbols: EXPECTED and GOOD")
print("  - WARNING logs for delisted symbols: EXPECTED and GOOD")
print("  - ERROR logs for delisted symbols: UNEXPECTED - NEEDS FIX")
print("  - Valid symbols work normally: GOOD")
print("\n" + "=" * 70)

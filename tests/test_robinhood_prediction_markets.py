"""
Test script to check if Robinhood API provides prediction markets data
"""

import robin_stocks.robinhood as rh
import inspect

print("=" * 60)
print("ROBINHOOD PREDICTION MARKETS API TEST")
print("=" * 60)

# Login
print("\n1. Logging into Robinhood...")
try:
    rh.login(username='brulecapital@gmail.com', password='FortKnox')
    print("[OK] Login successful")
except Exception as e:
    print(f"[ERROR] Login failed: {e}")
    exit(1)

# Check available API methods
print("\n2. Searching for prediction/event/market related API methods...")

# Get all functions in robin_stocks.robinhood module
all_methods = []
for module_name in dir(rh):
    module = getattr(rh, module_name)
    if hasattr(module, '__name__') and not module_name.startswith('_'):
        for func_name in dir(module):
            if not func_name.startswith('_') and callable(getattr(module, func_name)):
                all_methods.append(f"{module_name}.{func_name}")

# Search for keywords
keywords = ['prediction', 'event', 'market', 'contract', 'forecast', 'bet']
related_methods = []

for method in all_methods:
    method_lower = method.lower()
    for keyword in keywords:
        if keyword in method_lower:
            related_methods.append(method)
            break

if related_methods:
    print(f"\nFound {len(related_methods)} potentially related methods:")
    for method in related_methods:
        print(f"  - {method}")
else:
    print("\n[INFO] No obvious prediction market methods found")

# Check all available methods
print("\n3. All available robin_stocks methods:")
print(f"Total methods: {len(all_methods)}")

# Look for any market-related endpoints
print("\n4. Checking for 'markets' related methods:")
market_methods = [m for m in all_methods if 'market' in m.lower()]
for method in market_methods[:20]:  # Show first 20
    print(f"  - {method}")

# Try to call markets endpoint directly
print("\n5. Attempting to access markets data...")
try:
    # Try various potential endpoints
    endpoints_to_try = [
        ('rh.markets.get_markets()', lambda: rh.markets.get_markets()),
        ('rh.markets.get_market_hours()', lambda: rh.markets.get_market_hours('XNAS', None)),
    ]

    for name, func in endpoints_to_try:
        try:
            print(f"\nTrying: {name}")
            result = func()
            if result:
                print(f"  [OK] Got data: {type(result)}")
                if isinstance(result, list) and len(result) > 0:
                    print(f"  Sample keys: {list(result[0].keys())[:5] if isinstance(result[0], dict) else 'N/A'}")
        except Exception as e:
            print(f"  [SKIP] {str(e)[:100]}")

except Exception as e:
    print(f"[ERROR] {e}")

# Check if there's a specific predictions or events module
print("\n6. Checking for prediction markets module...")
try:
    # Try to import or access prediction markets
    if hasattr(rh, 'predictions'):
        print("[FOUND] rh.predictions module exists!")
    elif hasattr(rh, 'events'):
        print("[FOUND] rh.events module exists!")
    elif hasattr(rh, 'contracts'):
        print("[FOUND] rh.contracts module exists!")
    else:
        print("[INFO] No dedicated prediction markets module found in robin_stocks")

except Exception as e:
    print(f"[ERROR] {e}")

print("\n7. Conclusion:")
print("Robinhood offers prediction markets through their app, but the")
print("robin_stocks Python library may not have these endpoints exposed.")
print("\nThe robin_stocks library primarily covers:")
print("  - Stocks trading")
print("  - Options trading")
print("  - Crypto trading")
print("  - Account management")
print("\nPrediction markets may require:")
print("  - Direct API calls to undocumented endpoints")
print("  - Using browser automation")
print("  - Waiting for robin_stocks library update")

print("\n" + "=" * 60)
print("TEST COMPLETE")
print("=" * 60)

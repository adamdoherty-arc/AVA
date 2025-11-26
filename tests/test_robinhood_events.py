"""
Test script to explore Robinhood's get_events() API
Check what event data is available
"""

import robin_stocks.robinhood as rh
import json
from datetime import datetime

print("=" * 60)
print("ROBINHOOD GET_EVENTS() TEST")
print("=" * 60)

# Login
print("\n1. Logging into Robinhood...")
try:
    rh.login(username='brulecapital@gmail.com', password='FortKnox')
    print("[OK] Login successful")
except Exception as e:
    print(f"[ERROR] Login failed: {e}")
    exit(1)

# Test get_events from different modules
print("\n2. Testing get_events() from various modules...\n")

# Try stocks.get_events()
print("--- stocks.get_events(symbol='AAPL') ---")
try:
    events = rh.stocks.get_events('AAPL')
    if events:
        print(f"[OK] Found {len(events)} events for AAPL")
        if len(events) > 0:
            print(f"\nFirst AAPL event:")
            print(json.dumps(events[0], indent=2, default=str))
    else:
        print("[INFO] No events for AAPL")
except Exception as e:
    print(f"[ERROR] {e}")

# Check get_earnings (related to events)
print("\n\n--- markets.get_earnings(symbol='AAPL') ---")
try:
    earnings = rh.markets.get_earnings('AAPL')
    if earnings:
        print(f"[OK] Found earnings data")
        print(json.dumps(earnings, indent=2, default=str)[:500])
    else:
        print("[INFO] No earnings data")
except Exception as e:
    print(f"[ERROR] {e}")

# Test with your portfolio symbols
print("\n\n3. Testing events for your portfolio symbols...")
portfolio_symbols = ['DKNG', 'UPST', 'CIFR', 'HIMS', 'PLUG']

for symbol in portfolio_symbols[:2]:
    print(f"\n--- Events for {symbol} ---")
    try:
        events = rh.stocks.get_events(symbol)
        if events and len(events) > 0:
            print(f"  Found {len(events)} events")
            print(json.dumps(events[0], indent=2, default=str)[:300])
        else:
            print(f"  No events found")
    except Exception as e:
        print(f"  [ERROR] {e}")

print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)
print("get_events() returns corporate events like:")
print("  - Earnings announcements")
print("  - Dividend dates") 
print("  - Stock splits")
print("  - Company announcements")
print("\nThis is NOT prediction markets (those aren't in the API)")
print("=" * 60)

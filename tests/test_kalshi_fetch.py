"""
Test Kalshi market fetching and filtering
"""
from src.kalshi_client import KalshiClient

print("=" * 80)
print("TESTING KALSHI MARKET FETCH")
print("=" * 80)

# Initialize client
client = KalshiClient()

# Try to login (but don't exit if it fails - public API will be used)
print("\n1. Attempting Kalshi login...")
if client.login():
    print("   [OK] Login successful - using authenticated API")
else:
    print("   [INFO] Login failed - will use public API")

# Fetch all markets (will automatically fall back to public API if needed)
print("\n2. Fetching all markets...")
all_markets = client.get_all_markets(status='open', limit=100)
print(f"   Total markets fetched: {len(all_markets)}")

if all_markets:
    print(f"\n3. Sample markets (first 5):")
    for i, market in enumerate(all_markets[:5]):
        print(f"\n   Market {i+1}:")
        print(f"   - Ticker: {market.get('ticker')}")
        print(f"   - Title: {market.get('title')}")
        print(f"   - Category: {market.get('category')}")
        print(f"   - Status: {market.get('status')}")

# Test filtering
print("\n4. Testing non-sports filter...")
markets_by_sector = client.get_non_sports_markets()

print("\n5. Markets by sector:")
total_non_sports = 0
for sector, markets in markets_by_sector.items():
    count = len(markets)
    total_non_sports += count
    if count > 0:
        print(f"   {sector}: {count} markets")
        # Show one example
        if markets:
            example = markets[0]
            print(f"      Example: {example.get('title')[:60]}...")

print(f"\n6. Summary:")
print(f"   Total markets from API: {len(all_markets)}")
print(f"   Non-sports markets: {total_non_sports}")
print(f"   Sports markets filtered out: {len(all_markets) - total_non_sports}")

print("\n" + "=" * 80)

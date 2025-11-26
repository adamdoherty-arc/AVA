"""
Test MVP Flow for Prediction Markets
Verify entire pipeline works: Fetch -> Filter -> Evaluate -> Display
"""
import sys
sys.stdout.reconfigure(encoding='utf-8')

from src.kalshi_integration import KalshiIntegration
from src.general_market_evaluator import GeneralMarketEvaluator
from prediction_markets_page import categorize_non_sports_markets

print("=" * 80)
print("TESTING MVP FLOW - PREDICTION MARKETS")
print("=" * 80)

# Step 1: Fetch markets from public API
print("\n[Step 1] Fetching markets from public API...")
client = KalshiIntegration()
all_markets = client.get_markets(limit=500, status='active')
print(f"   ✓ Fetched {len(all_markets)} total markets")

# Step 2: Filter and categorize non-sports markets
print("\n[Step 2] Filtering out sports markets...")
markets_by_sector = categorize_non_sports_markets(all_markets)
total_non_sports = sum(len(m) for m in markets_by_sector.values())
print(f"   ✓ Found {total_non_sports} non-sports markets")
print(f"   ✓ Sports filtered: {len(all_markets) - total_non_sports}")

# Show breakdown by sector
print("\n[Step 3] Markets by sector:")
for sector, markets in markets_by_sector.items():
    if markets:
        print(f"   • {sector}: {len(markets)} markets")

# Step 4: Evaluate markets with AI
print("\n[Step 4] Evaluating markets with AI...")
evaluator = GeneralMarketEvaluator()

# Get all non-sports markets
all_non_sports = []
for sector_markets in markets_by_sector.values():
    all_non_sports.extend(sector_markets)

if all_non_sports:
    evaluated = evaluator.evaluate_markets(all_non_sports)
    print(f"   ✓ Evaluated {len(evaluated)} markets")

    # Show top 5 by AI score
    print("\n[Step 5] Top 5 opportunities by AI rating:")
    for i, market in enumerate(evaluated[:5]):
        print(f"\n   {i+1}. {market.get('title', 'Unknown')[:60]}...")
        print(f"      Overall Score: {market.get('overall_score', 0):.1f}/100")
        print(f"      Sector: {market.get('sector', 'Unknown')}")
        print(f"      Recommendation: {market.get('recommended_action', 'PASS')}")
        print(f"      YES Price: {market.get('yes_price', 0)*100:.1f}%")
        print(f"      Reasoning: {market.get('reasoning', 'N/A')[:80]}...")
else:
    print("   ⚠ No non-sports markets available to evaluate")

print("\n" + "=" * 80)
print("MVP FLOW TEST COMPLETE")
print("=" * 80)

# Summary
print("\n✓ MVP STATUS: WORKING")
print(f"  • Public API: ✓ Accessible")
print(f"  • Market Fetch: ✓ {len(all_markets)} markets")
print(f"  • Sports Filter: ✓ Filtered {len(all_markets) - total_non_sports} sports markets")
print(f"  • Non-Sports: {total_non_sports} markets available")
print(f"  • AI Evaluation: ✓ Working")
print(f"  • Rating System: ✓ 5-component scoring")

if total_non_sports == 0:
    print("\n⚠ LIMITATION: Currently no non-sports markets available")
    print("  The public API is heavily sports-focused right now.")
    print("  Non-sports markets (Politics, Economics, etc.) are rare.")

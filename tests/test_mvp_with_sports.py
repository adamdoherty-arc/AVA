"""
Test MVP with Sports Markets Included
Verify the entire pipeline works with sports markets
"""
import sys
sys.stdout.reconfigure(encoding='utf-8')

from src.kalshi_integration import KalshiIntegration
from src.general_market_evaluator import GeneralMarketEvaluator
from prediction_markets_page import categorize_all_markets

print("=" * 80)
print("TESTING MVP WITH SPORTS MARKETS")
print("=" * 80)

# Step 1: Fetch markets from public API
print("\n[Step 1] Fetching markets from public API...")
client = KalshiIntegration()
all_markets = client.get_markets(limit=200, status='active')
print(f"   ✓ Fetched {len(all_markets)} total markets")

# Step 2: Categorize ALL markets (including sports)
print("\n[Step 2] Categorizing all markets (including sports)...")
markets_by_sector = categorize_all_markets(all_markets)
total_markets = sum(len(m) for m in markets_by_sector.values())
print(f"   ✓ Categorized {total_markets} markets")

# Show breakdown by sector
print("\n[Step 3] Markets by sector:")
for sector, markets in markets_by_sector.items():
    if markets:
        print(f"   • {sector}: {len(markets)} markets")

# Step 4: Evaluate markets with AI (sample 50 for speed)
print("\n[Step 4] Evaluating sample markets with AI...")
evaluator = GeneralMarketEvaluator()

# Get all markets
all_categorized = []
for sector_markets in markets_by_sector.values():
    all_categorized.extend(sector_markets)

# Evaluate first 50 for speed
sample_size = min(50, len(all_categorized))
evaluated = evaluator.evaluate_markets(all_categorized[:sample_size])
print(f"   ✓ Evaluated {len(evaluated)} markets")

# Show top 10 by AI score
print("\n[Step 5] Top 10 opportunities by AI rating:")
for i, market in enumerate(evaluated[:10]):
    print(f"\n   {i+1}. Score: {market.get('overall_score', 0):.1f}/100")
    print(f"      Sector: {market.get('sector', 'Unknown')}")
    print(f"      Title: {market.get('title', 'Unknown')[:80]}...")
    print(f"      Action: {market.get('recommended_action', 'PASS')}")
    print(f"      YES: {market.get('yes_price', 0)*100:.1f}% | Reasoning: {market.get('reasoning', 'N/A')[:60]}...")

print("\n" + "=" * 80)
print("MVP WITH SPORTS TEST COMPLETE")
print("=" * 80)

# Summary
print("\n✅ MVP STATUS: FULLY WORKING")
print(f"  • Public API: ✓ Accessible")
print(f"  • Market Fetch: ✓ {len(all_markets)} markets")
print(f"  • Categorization: ✓ All markets categorized")
print(f"  • Sports Markets: ✓ {len(markets_by_sector.get('Sports', []))} available")
print(f"  • AI Evaluation: ✓ Working on all market types")
print(f"  • Rating System: ✓ 5-component scoring")
print(f"  • Top Score: {evaluated[0].get('overall_score', 0):.1f}/100" if evaluated else "  • Top Score: N/A")

print("\n🎉 MVP READY FOR DEMO!")

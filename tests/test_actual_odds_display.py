"""
Test what odds are actually being displayed
"""
import sys
sys.path.insert(0, 'c:\\code\\Magnus')

from src.espn_ncaa_live_data import get_espn_ncaa_client
from src.espn_kalshi_matcher_optimized import enrich_games_with_kalshi_odds_optimized

print("=" * 100)
print("TESTING ACTUAL ODDS DISPLAY")
print("=" * 100)

# Fetch ESPN NCAA games
print("\n1. Fetching ESPN NCAA games...")
espn = get_espn_ncaa_client()
games = espn.get_scoreboard(group='80')  # FBS
print(f"[OK] Fetched {len(games)} games")

# Enrich with Kalshi odds
print("\n2. Enriching with Kalshi odds...")
enriched = enrich_games_with_kalshi_odds_optimized(games[:10], sport='ncaaf')
print(f"[OK] Enriched {len(enriched)} games")

# Display what would show in UI
print("\n3. What UI would display:")
print("-" * 100)

matched_count = 0
for i, game in enumerate(enriched, 1):
    away = game.get('away_team', 'Unknown')
    home = game.get('home_team', 'Unknown')
    kalshi_odds = game.get('kalshi_odds')

    print(f"\n{i}. {away} @ {home}")

    if kalshi_odds:
        matched_count += 1

        # This is what the UI does (lines 1155-1156 in game_cards_visual_page.py)
        away_odds = float(kalshi_odds.get('away_win_price', 0)) * 100
        home_odds = float(kalshi_odds.get('home_win_price', 0)) * 100

        print(f"   Kalshi Odds:")
        print(f"     {away}: {away_odds:.1f}%")
        print(f"     {home}: {home_odds:.1f}%")
        print(f"   Raw values:")
        print(f"     away_win_price: {kalshi_odds.get('away_win_price')}")
        print(f"     home_win_price: {kalshi_odds.get('home_win_price')}")
        print(f"   Ticker: {kalshi_odds.get('ticker')}")
    else:
        print(f"   No Kalshi odds")

print(f"\n\nMatched {matched_count}/{len(enriched)} games with Kalshi odds")

# Check for the "54%" issue
print("\n4. Checking for 54% issue...")
print("-" * 100)

fifty_four_count = 0
for game in enriched:
    kalshi_odds = game.get('kalshi_odds')
    if kalshi_odds:
        away_odds = float(kalshi_odds.get('away_win_price', 0)) * 100
        home_odds = float(kalshi_odds.get('home_win_price', 0)) * 100

        if abs(away_odds - 54) < 1 or abs(home_odds - 54) < 1:
            fifty_four_count += 1
            print(f"   Found 54%: {game.get('away_team')} @ {game.get('home_team')}")
            print(f"     Away: {away_odds:.1f}% | Home: {home_odds:.1f}%")

if fifty_four_count > 0:
    print(f"\n[!] Found {fifty_four_count} games with ~54% odds")
else:
    print(f"\n[OK] No games showing 54% odds in test")

print("\n" + "=" * 100)

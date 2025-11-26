"""Test how games get kalshi odds"""
from src.espn_kalshi_matcher import ESPNKalshiMatcher
from src.espn_live_data import get_espn_client

espn = get_espn_client()
games = espn.get_scoreboard()
matcher = ESPNKalshiMatcher()

print(f"Total NFL games: {len(games)}\n")

if games:
    print(f"First game BEFORE enrichment:")
    print(f"  {games[0].get('away_team')} @ {games[0].get('home_team')}")
    print(f"  kalshi_odds: {games[0].get('kalshi_odds')}")

    enriched_games = matcher.enrich_espn_games_with_kalshi(games)

    if enriched_games:
        game = enriched_games[0]
        print(f"\nAfter enrichment:")
        print(f"  {game.get('away_team')} @ {game.get('home_team')}")
        print(f"  kalshi_odds: {game.get('kalshi_odds')}")

        if game.get('kalshi_odds'):
            odds = game['kalshi_odds']
            print(f"\n  Odds details:")
            print(f"    away_win_price: {odds.get('away_win_price')}")
            print(f"    home_win_price: {odds.get('home_win_price')}")

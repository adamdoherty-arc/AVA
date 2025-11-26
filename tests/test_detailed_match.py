"""Detailed test to find the exact error"""
import traceback
from src.espn_live_data import get_espn_client
from src.espn_kalshi_matcher import ESPNKalshiMatcher

# Get one game
espn_client = get_espn_client()
games = espn_client.get_scoreboard()

if games:
    game = games[0]  # First game
    print(f"Testing match for: {game['away_team']} @ {game['home_team']}")
    print(f"Game time: {game.get('game_time')}")
    print(f"Game time type: {type(game.get('game_time'))}")
    print()

    # Try to match
    matcher = ESPNKalshiMatcher()
    try:
        result = matcher.match_game_to_kalshi(game)
        if result:
            print(f"SUCCESS: Matched to {result['ticker']}")
        else:
            print("No match found")
    except Exception as e:
        print(f"ERROR: {e}")
        print("\nFull traceback:")
        traceback.print_exc()

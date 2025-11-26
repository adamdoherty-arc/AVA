"""
Test subscription functionality
"""
import os
import logging
from dotenv import load_dotenv
from src.game_watchlist_manager import GameWatchlistManager
from src.kalshi_db_manager import KalshiDBManager
from src.espn_live_data import get_espn_client

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

load_dotenv()

# Initialize managers
db = KalshiDBManager()
watchlist_manager = GameWatchlistManager(db)

# Get test user ID
test_user_id = os.getenv('TELEGRAM_AUTHORIZED_USERS', 'test_user').split(',')[0]
print(f"Test User ID: {test_user_id}\n")

# Fetch real games from ESPN
print("Fetching games from ESPN...")
espn = get_espn_client()
games = espn.get_scoreboard(week=12)
print(f"Fetched {len(games)} games\n")

# Find Miami and Virginia Tech games
miami_games = [g for g in games if 'Miami' in g.get('home_team', '') or 'Miami' in g.get('away_team', '')]
vt_games = [g for g in games if 'Virginia' in g.get('home_team', '') or 'Virginia' in g.get('away_team', '')]

print("="*60)
print("MIAMI GAMES:")
print("="*60)
for game in miami_games:
    print(f"Game ID: {game.get('game_id')}")
    print(f"  {game.get('away_team')} @ {game.get('home_team')}")
    print(f"  Status: {game.get('status')}")
    print()

print("="*60)
print("VIRGINIA TECH GAMES:")
print("="*60)
for game in vt_games:
    print(f"Game ID: {game.get('game_id')}")
    print(f"  {game.get('away_team')} @ {game.get('home_team')}")
    print(f"  Status: {game.get('status')}")
    print()

# Test subscription
if miami_games:
    print("="*60)
    print("TESTING MIAMI SUBSCRIPTION")
    print("="*60)
    test_game = miami_games[0]
    print(f"Subscribing to: {test_game.get('away_team')} @ {test_game.get('home_team')}")
    print(f"Game ID: {test_game.get('game_id')}")

    # Add sport field (ESPN doesn't include it)
    test_game['sport'] = 'NFL'

    success = watchlist_manager.add_game_to_watchlist(test_user_id, test_game, selected_team=None)

    if success:
        print("\n✅ Subscription SUCCESSFUL")
        print("   - Check Telegram for notification")
        print("   - Verify in database")
    else:
        print("\n❌ Subscription FAILED")
        print("   - Check logs above for error")

# Verify subscriptions
print("\n" + "="*60)
print("VERIFYING SUBSCRIPTIONS IN DATABASE")
print("="*60)
watchlist = watchlist_manager.get_user_watchlist(test_user_id)
print(f"Total subscriptions: {len(watchlist)}\n")
for sub in watchlist:
    print(f"Game: {sub.get('away_team')} @ {sub.get('home_team')}")
    print(f"  Sport: {sub.get('sport')}")
    print(f"  Game ID: {sub.get('game_id')}")
    print()

"""
COMPREHENSIVE SUBSCRIPTION TEST & SETUP
Tests subscription system and subscribes to Miami and VT games
"""
import os
import logging
from dotenv import load_dotenv
from src.game_watchlist_manager import GameWatchlistManager
from src.kalshi_db_manager import KalshiDBManager
from src.espn_ncaa_live_data import get_espn_ncaa_client
from src.espn_live_data import get_espn_client

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

load_dotenv()

print("="*70)
print("COMPREHENSIVE SUBSCRIPTION TEST")
print("="*70)

# Initialize
db = KalshiDBManager()
watchlist_manager = GameWatchlistManager(db)
user_id = os.getenv('TELEGRAM_AUTHORIZED_USERS', '').split(',')[0]

print(f"\nUser ID: {user_id}")

# Step 1: Find Miami @ VT game (NCAA)
print("\n" + "="*70)
print("STEP 1: Finding Miami @ Virginia Tech (NCAA)")
print("="*70)

espn_ncaa = get_espn_ncaa_client()
ncaa_games = espn_ncaa.get_scoreboard(week=13)
print(f"Found {len(ncaa_games)} NCAA games")

miami_vt_game = None
for game in ncaa_games:
    if ('Miami' in game.get('away_team', '') and 'Virginia Tech' in game.get('home_team', '')) or \
       ('Miami' in game.get('home_team', '') and 'Virginia Tech' in game.get('away_team', '')):
        miami_vt_game = game
        print(f"\nFOUND GAME:")
        print(f"  {game.get('away_team')} @ {game.get('home_team')}")
        print(f"  Status: {game.get('status')}")
        print(f"  Game ID: {game.get('game_id')}")
        print(f"  Score: {game.get('away_score')}-{game.get('home_score')}")
        break

if not miami_vt_game:
    print("\nWARNING: Miami @ VT game not found in Week 13")
    print("Checking other weeks...")
    for week in [12, 14]:
        ncaa_games = espn_ncaa.get_scoreboard(week=week)
        for game in ncaa_games:
            if 'Miami' in str(game.get('away_team', '')) or 'Miami' in str(game.get('home_team', '')):
                if 'Virginia Tech' in str(game.get('away_team', '')) or 'Virginia Tech' in str(game.get('home_team', '')):
                    miami_vt_game = game
                    print(f"\nFOUND IN WEEK {week}:")
                    print(f"  {game.get('away_team')} @ {game.get('home_team')}")
                    break

# Step 2: Find Miami NFL game
print("\n" + "="*70)
print("STEP 2: Finding Miami Dolphins (NFL)")
print("="*70)

espn_nfl = get_espn_client()
nfl_games = espn_nfl.get_scoreboard(week=13)
print(f"Found {len(nfl_games)} NFL games")

miami_nfl_game = None
for game in nfl_games:
    if 'Miami' in game.get('away_team', '') or 'Miami' in game.get('home_team', ''):
        miami_nfl_game = game
        print(f"\nFOUND GAME:")
        print(f"  {game.get('away_team')} @ {game.get('home_team')}")
        print(f"  Status: {game.get('status')}")
        print(f"  Game ID: {game.get('game_id')}")
        break

# Step 3: Subscribe to games
print("\n" + "="*70)
print("STEP 3: SUBSCRIBING TO GAMES")
print("="*70)

subscriptions_added = []

if miami_vt_game:
    print("\nSubscribing to Miami @ Virginia Tech (NCAA)...")
    miami_vt_game['sport'] = 'CFB'  # CRITICAL: Add sport field
    success = watchlist_manager.add_game_to_watchlist(user_id, miami_vt_game, selected_team=None)
    if success:
        print("  SUCCESS - Subscribed to NCAA game")
        subscriptions_added.append("Miami @ Virginia Tech (NCAA)")
    else:
        print("  FAILED - Check logs above")
else:
    print("\nSKIPPED: Miami @ VT game not found")

if miami_nfl_game:
    print("\nSubscribing to Miami Dolphins (NFL)...")
    miami_nfl_game['sport'] = 'NFL'  # CRITICAL: Add sport field
    success = watchlist_manager.add_game_to_watchlist(user_id, miami_nfl_game, selected_team=None)
    if success:
        print("  SUCCESS - Subscribed to NFL game")
        subscriptions_added.append(f"{miami_nfl_game.get('away_team')} @ {miami_nfl_game.get('home_team')} (NFL)")
    else:
        print("  FAILED - Check logs above")
else:
    print("\nSKIPPED: Miami NFL game not found")

# Step 4: Verify in database
print("\n" + "="*70)
print("STEP 4: VERIFYING SUBSCRIPTIONS IN DATABASE")
print("="*70)

watchlist = watchlist_manager.get_user_watchlist(user_id)
print(f"\nTotal subscriptions for user {user_id}: {len(watchlist)}")

for idx, sub in enumerate(watchlist, 1):
    print(f"\n{idx}. {sub.get('away_team')} @ {sub.get('home_team')}")
    print(f"   Sport: {sub.get('sport')}")
    print(f"   Game ID: {sub.get('game_id')}")
    print(f"   Active: {sub.get('is_active')}")

# Step 5: Test Telegram
print("\n" + "="*70)
print("STEP 5: TELEGRAM NOTIFICATION TEST")
print("="*70)

try:
    from src.telegram_notifier import TelegramNotifier
    notifier = TelegramNotifier()

    test_message = "TEST SUBSCRIPTION CONFIRMATION\n\n"
    test_message += f"User ID: {user_id}\n"
    test_message += f"Subscriptions added: {len(subscriptions_added)}\n\n"
    for sub in subscriptions_added:
        test_message += f"- {sub}\n"
    test_message += "\nYou should receive updates for these games!"

    notifier.send_custom_message(test_message)
    print("\nTelegram test message sent!")
    print("CHECK YOUR TELEGRAM for confirmation")
except Exception as e:
    print(f"\nTelegram test FAILED: {e}")

# Step 6: Summary
print("\n" + "="*70)
print("SUMMARY")
print("="*70)

print(f"\nSubscriptions added: {len(subscriptions_added)}")
for sub in subscriptions_added:
    print(f"  - {sub}")

print("\nNext steps:")
print("  1. CHECK TELEGRAM - You should have received notifications")
print("  2. Refresh Game Hub page")
print("  3. Go to Settings tab - verify subscriptions show up")
print("  4. Monitor Telegram during live games for updates")

print("\n" + "="*70)
print("TEST COMPLETE")
print("="*70)

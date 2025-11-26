"""
Force refresh and verify subscriptions are showing correctly
"""
import os
import streamlit as st
from dotenv import load_dotenv
from src.game_watchlist_manager import GameWatchlistManager
from src.kalshi_db_manager import KalshiDBManager

load_dotenv()

print("="*70)
print("FORCE REFRESH - CLEARING STREAMLIT CACHE")
print("="*70)

# Initialize
db = KalshiDBManager()
watchlist_manager = GameWatchlistManager(db)

# Get user ID from environment
user_id = os.getenv('TELEGRAM_AUTHORIZED_USERS', '').split(',')[0]
print(f"\nUser ID: {user_id}")

# Get fresh data from database (no cache)
watchlist = watchlist_manager.get_user_watchlist(user_id)

print(f"\nDatabase Query Results:")
print(f"Total subscriptions: {len(watchlist)}")

# Group by sport
nfl_games = [w for w in watchlist if w.get('sport') == 'NFL']
ncaa_games = [w for w in watchlist if w.get('sport') == 'CFB']
nba_games = [w for w in watchlist if w.get('sport') == 'NBA']

print(f"\nNFL Games: {len(nfl_games)}")
for game in nfl_games:
    print(f"  - {game.get('away_team')} @ {game.get('home_team')}")
    print(f"    Game ID: {game.get('game_id')}")
    print(f"    Active: {game.get('is_active')}")

print(f"\nNCAA Games: {len(ncaa_games)}")
for game in ncaa_games:
    print(f"  - {game.get('away_team')} @ {game.get('home_team')}")
    print(f"    Game ID: {game.get('game_id')}")
    print(f"    Active: {game.get('is_active')}")

print(f"\nNBA Games: {len(nba_games)}")

print("\n" + "="*70)
print("INSTRUCTIONS TO FIX UI:")
print("="*70)
print("\n1. In your browser (Game Hub page):")
print("   - Press 'C' key to clear cache")
print("   - Or press Ctrl+Shift+R to hard refresh")
print("")
print("2. Then refresh the Settings tab:")
print("   - Click away from Settings tab")
print("   - Click back on Settings tab")
print("")
print("3. You should now see:")
print(f"   - NFL Games: {len(nfl_games)}")
print(f"   - NCAA Games: {len(ncaa_games)}")
print(f"   - Total: {len(watchlist)}")

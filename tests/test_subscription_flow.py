"""
Test Subscription Flow
Verify that game data is correctly passed through subscription
"""
from dotenv import load_dotenv
load_dotenv(override=True)

from src.game_watchlist_manager import GameWatchlistManager
from src.telegram_notifier import TelegramNotifier

print("=" * 70)
print("  Testing Subscription Flow")
print("=" * 70)

# Create a realistic game object (like what ESPN returns)
miami_game = {
    'game_id': 'test_miami_123',
    'sport': 'NFL',
    'away_team': 'Miami Dolphins',
    'home_team': 'Buffalo Bills',
    'away_score': 14,
    'home_score': 21,
    'status': 'Live',
    'status_detail': '3rd Quarter',
    'is_live': True,
    'game_time': '2025-11-24 13:00:00',
    'ai_prediction': {
        'predicted_winner': 'Buffalo Bills',
        'point_spread': '-7.5',
        'win_probability': 68,
        'confidence': 'High'
    }
}

print("\nGame Data:")
print(f"  Away: {miami_game['away_team']}")
print(f"  Home: {miami_game['home_team']}")
print(f"  Score: {miami_game['away_score']} - {miami_game['home_score']}")
print(f"  Status: {miami_game['status_detail']}")

# Test subscription
print("\n" + "-" * 70)
print("  Adding to watchlist...")
print("-" * 70)

gwm = GameWatchlistManager()
user_id = "test_user"

success = gwm.add_game_to_watchlist(user_id, miami_game, selected_team="Miami Dolphins")

if success:
    print("\n✅ Game added to watchlist successfully!")
    print("📱 Telegram alert should have been sent")
    print("\nCheck your Telegram to verify:")
    print("  - Teams are correct (Miami Dolphins @ Buffalo Bills)")
    print("  - NOT showing 'Test Team'")
    print("  - Score is shown correctly")
else:
    print("\n❌ Failed to add game to watchlist")

print("\n" + "=" * 70)
print("  Test Complete")
print("=" * 70)

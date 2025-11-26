"""Final test of Telegram integration"""
from dotenv import load_dotenv
load_dotenv(override=True)

from src.telegram_notifier import TelegramNotifier
from src.game_watchlist_manager import GameWatchlistManager

print("=" * 70)
print("  Final Telegram Integration Test")
print("=" * 70)

# Test 1: Direct TelegramNotifier
print("\n1. Testing TelegramNotifier directly...")
telegram = TelegramNotifier()
msg_id = telegram.send_custom_message("Direct test from TelegramNotifier class")
print(f"   Result: {'SUCCESS - Message ID: ' + str(msg_id) if msg_id else 'FAILED'}")

# Test 2: Game subscription alert
print("\n2. Testing game subscription alert...")
gwm = GameWatchlistManager()
test_game = {
    'away_team': 'Oklahoma Sooners',
    'home_team': 'Missouri Tigers',
    'status': 'Live',
    'away_score': 21,
    'home_score': 17,
    'game_date': '11/22 - 3:30 PM EST',
    'sport': 'NCAA',
    'ai_prediction': {
        'predicted_winner': 'Oklahoma',
        'point_spread': '-6.5',
        'win_probability': 68,
        'confidence': 'High'
    }
}

try:
    gwm._send_subscription_alert(test_game)
    print("   Result: SUCCESS - Subscription alert sent")
except Exception as e:
    print(f"   Result: FAILED - {e}")

print("\n" + "=" * 70)
print("  Test Complete - Check your Telegram app!")
print("=" * 70)

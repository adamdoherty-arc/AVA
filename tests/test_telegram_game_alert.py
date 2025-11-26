"""
Test Telegram Game Alert System
================================

Quick test script to send a test game alert to Telegram.

Usage:
    python test_telegram_game_alert.py

Author: Magnus Trading Platform
"""

import os
import sys
from dotenv import load_dotenv
from src.telegram_notifier import TelegramNotifier

# Load environment variables
load_dotenv()

def send_test_game_alert():
    """Send a test game alert to Telegram"""

    print("=" * 60)
    print("  Telegram Game Alert System - Test")
    print("=" * 60)

    # Check if Telegram is configured
    bot_token = os.getenv('TELEGRAM_BOT_TOKEN')
    chat_id = os.getenv('TELEGRAM_CHAT_ID')
    enabled = os.getenv('TELEGRAM_ENABLED', 'false').lower() == 'true'

    print(f"\nBot Token: {'Configured' if bot_token and bot_token != 'your_bot_token_here' else 'NOT configured'}")
    print(f"Chat ID: {'Configured' if chat_id and chat_id != 'your_chat_id_here' else 'NOT configured'}")
    print(f"Enabled: {enabled}")

    if not bot_token or bot_token == 'your_bot_token_here':
        print("\nERROR: TELEGRAM_BOT_TOKEN not configured in .env")
        print("\nTo configure:")
        print("1. Open Telegram and search for @BotFather")
        print("2. Send /newbot and follow instructions")
        print("3. Copy the bot token")
        print("4. Add to .env file: TELEGRAM_BOT_TOKEN=your_token_here")
        sys.exit(1)

    if not chat_id or chat_id == 'your_chat_id_here':
        print("\nERROR: TELEGRAM_CHAT_ID not configured in .env")
        print("\nTo configure:")
        print("1. Start a chat with your bot on Telegram")
        print("2. Open Telegram and search for @userinfobot")
        print("3. Send any message to get your chat ID")
        print("4. Add to .env file: TELEGRAM_CHAT_ID=your_chat_id")
        sys.exit(1)

    # Initialize notifier
    print("\n" + "-" * 60)
    print("Initializing Telegram Notifier...")
    print("-" * 60)

    notifier = TelegramNotifier()

    # Test connection
    print("\nTesting connection...")
    if not notifier.test_connection():
        print("Connection test failed!")
        sys.exit(1)

    print("Connection successful!")

    # Send test game alert
    print("\n" + "-" * 60)
    print("Sending Test Game Alert...")
    print("-" * 60)

    test_message = """
🏈 **GAME ALERT SUBSCRIPTION CONFIRMED**

You've successfully subscribed to game updates!

**Test Game:**
🏆 #8 Oklahoma Sooners (8-2)
     vs
🐅 #22 Missouri Tigers (7-3)

📅 Date: 11/22 - 12:00 PM EST
📺 Status: Live Now - 0

You'll receive notifications for:
• Score updates
• Quarter changes
• Game status changes
• AI prediction updates

🤖 Multi-Agent AI Analysis
🎯 Prediction: Oklahoma -13.8
✅ 72% win probability
💡 Medium Confidence

**Powered by Magnus Trading Platform**
"""

    message_id = notifier.send_custom_message(test_message)

    if message_id:
        print(f"\nTest game alert sent successfully!")
        print(f"   Message ID: {message_id}")
        print("\nCheck your Telegram to see the alert!")
    else:
        print("\nFailed to send test alert")
        sys.exit(1)

    print("\n" + "=" * 60)
    print("Test Complete!")
    print("=" * 60)
    print("\nTelegram game alerts are working correctly!")
    print("   You can now use the Subscribe button in the Game Cards page.")
    print()

if __name__ == "__main__":
    try:
        send_test_game_alert()
    except KeyboardInterrupt:
        print("\n\nTest interrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"\nFatal error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

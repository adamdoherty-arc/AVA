"""
Test Telegram Connection and Get Chat ID for AVA Bot
"""
import os
import sys
import requests
from dotenv import load_dotenv

# Fix Windows console encoding
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

load_dotenv()

def get_telegram_updates():
    """Get recent Telegram messages to find chat ID"""
    token = os.getenv("TELEGRAM_BOT_TOKEN")

    if not token or token == "YOUR_BOT_TOKEN_HERE":
        print("[ERROR] TELEGRAM_BOT_TOKEN not configured in .env")
        return

    print("=" * 70)
    print("TELEGRAM CONNECTION TEST")
    print("=" * 70)
    print()

    # Get bot info
    url = f"https://api.telegram.org/bot{token}/getMe"
    response = requests.get(url)

    if response.status_code == 200:
        bot_info = response.json()
        if bot_info['ok']:
            bot = bot_info['result']
            print(f"[OK] Bot Connected: @{bot['username']}")
            print(f"     Bot Name: {bot['first_name']}")
            print(f"     Bot ID: {bot['id']}")
            print()
        else:
            print("[ERROR] Bot connection failed")
            return
    else:
        print("[ERROR] Failed to connect to Telegram API")
        return

    # Get recent messages
    print("Checking for recent messages...")
    print("(If no messages found, send any message to your bot first)")
    print()

    url = f"https://api.telegram.org/bot{token}/getUpdates"
    response = requests.get(url)

    if response.status_code == 200:
        data = response.json()

        if data['ok'] and data['result']:
            print(f"[OK] Found {len(data['result'])} message(s)")
            print()

            # Get the most recent chat ID
            latest = data['result'][-1]

            if 'message' in latest:
                chat_id = latest['message']['chat']['id']
                username = latest['message']['chat'].get('username', 'N/A')
                first_name = latest['message']['chat'].get('first_name', 'N/A')

                print("=" * 70)
                print("YOUR CHAT INFORMATION:")
                print("=" * 70)
                print(f"Chat ID: {chat_id}")
                print(f"Username: @{username}")
                print(f"Name: {first_name}")
                print()
                print("=" * 70)
                print("NEXT STEPS:")
                print("=" * 70)
                print(f"1. Copy this to your .env file:")
                print(f"   TELEGRAM_CHAT_ID={chat_id}")
                print()
                print("2. Restart the dashboard")
                print()
                print("3. Game watchlist alerts will now work!")
                print("=" * 70)

                # Test sending a message
                print()
                test_send = input("Send a test message to verify? (y/n): ")
                if test_send.lower() == 'y':
                    send_test_message(token, chat_id)
            else:
                print("[WARN] No message found in updates")
                print("       Please send a message to your bot first")
        else:
            print("[WARN] No messages found")
            print()
            print("TO FIX:")
            print(f"1. Open Telegram and search for your bot")
            print("2. Send any message (e.g., 'hello')")
            print("3. Run this script again")
    else:
        print("[ERROR] Failed to get updates")

def send_test_message(token, chat_id):
    """Send a test message"""
    url = f"https://api.telegram.org/bot{token}/sendMessage"

    message = """
🎉 AVA Bot Test Message

✅ Telegram connection is working!
✅ Chat ID configured correctly
✅ Game watchlist alerts are ready

You will now receive notifications when:
• Games are added to your watchlist
• Scores change
• Game status changes
• AI predictions update
• Kalshi odds shift

Background monitor running every 5 minutes.
Enjoy your real-time game alerts! 🏈
"""

    data = {
        "chat_id": chat_id,
        "text": message,
        "parse_mode": "Markdown"
    }

    response = requests.post(url, json=data)

    if response.status_code == 200:
        print()
        print("✅ Test message sent successfully!")
        print("   Check your Telegram app")
    else:
        print()
        print("❌ Failed to send test message")
        print(f"   Error: {response.text}")

if __name__ == "__main__":
    get_telegram_updates()

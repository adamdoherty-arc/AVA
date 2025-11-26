"""
Setup Telegram Alerts - Automatic Configuration
================================================

This script will:
1. Check your Telegram bot token
2. Get recent messages and extract your chat ID
3. Test sending an alert
"""

import os
import asyncio
from dotenv import load_dotenv, set_key
from telegram import Bot

load_dotenv()

async def setup_telegram():
    """Setup Telegram alerts automatically"""

    print("=" * 70)
    print("  Telegram Alerts Setup - Automatic Configuration")
    print("=" * 70)

    bot_token = os.getenv('TELEGRAM_BOT_TOKEN')
    chat_id = os.getenv('TELEGRAM_CHAT_ID')

    if not bot_token or bot_token == 'your_bot_token_here':
        print("\nERROR: TELEGRAM_BOT_TOKEN not configured in .env")
        return False

    print(f"\nBot Token: Configured")
    print(f"Current Chat ID: {chat_id}")

    bot = Bot(token=bot_token)

    try:
        # Get bot info
        bot_info = await bot.get_me()
        print(f"\nBot Name: {bot_info.first_name}")
        print(f"Bot Username: @{bot_info.username}")

        # Get recent messages to find chat ID
        print("\nFetching recent messages...")
        updates = await bot.get_updates()

        if not updates:
            print("\n" + "=" * 70)
            print("  Action Required!")
            print("=" * 70)
            print(f"\n1. Open Telegram")
            print(f"2. Search for @{bot_info.username}")
            print(f"3. Send any message (e.g., /start or Hello)")
            print(f"4. Run this script again")
            await bot.close()
            return False

        # Get the latest chat ID
        latest_chat_id = updates[-1].message.chat.id
        from_user = updates[-1].message.from_user

        print(f"\nFound your chat!")
        print(f"Chat ID: {latest_chat_id}")
        if from_user:
            print(f"User: {from_user.first_name}")
            if from_user.username:
                print(f"Username: @{from_user.username}")

        # Update .env file
        env_file = '.env'
        if os.path.exists(env_file):
            set_key(env_file, 'TELEGRAM_CHAT_ID', str(latest_chat_id))
            print(f"\n[SUCCESS] Updated .env file with your chat ID!")

        # Test sending a message
        print("\n" + "=" * 70)
        print("  Sending Test Alert...")
        print("=" * 70)

        test_message = """
🎉 **TELEGRAM ALERTS ACTIVATED**

Your Magnus game alerts are now configured!

When you click Subscribe on any game card, you'll receive:
• Real-time score updates
• Quarter/period changes
• AI prediction updates
• Kalshi odds changes

**Test Game Alert:**
🏈 Oklahoma Sooners @ Missouri Tigers
📅 11/22 - 12:00 PM EST
📺 Live Now - 0:0

🤖 AI Analysis: Oklahoma -13.8
✅ 72% win probability

**Setup Complete!** 🚀
"""

        result = await bot.send_message(
            chat_id=latest_chat_id,
            text=test_message,
            parse_mode='Markdown'
        )

        print(f"\n[SUCCESS] Test alert sent successfully!")
        print(f"  Message ID: {result.message_id}")
        print("\n>> Check your Telegram app!")

        print("\n" + "=" * 70)
        print("  Setup Complete!")
        print("=" * 70)
        print("\n[SUCCESS] Telegram alerts are ready!")
        print("  Go to Game Cards page and click Subscribe on any game.")
        print()

        await bot.close()
        return True

    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        await bot.close()
        return False

if __name__ == "__main__":
    success = asyncio.run(setup_telegram())

    if success:
        print("\n[SUCCESS] You're all set! Subscribe to games to get instant alerts.")
    else:
        print("\n[WARNING] Setup incomplete. Follow the instructions above and try again.")

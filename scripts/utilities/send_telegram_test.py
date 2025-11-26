"""
Quick Telegram Test - Async Version
====================================

Simple async test to send a Telegram message.
"""

import os
import asyncio
from dotenv import load_dotenv
from telegram import Bot

load_dotenv()

async def send_test_message():
    """Send a test message via Telegram"""
    bot_token = os.getenv('TELEGRAM_BOT_TOKEN')
    chat_id = os.getenv('TELEGRAM_CHAT_ID')

    print("=" * 60)
    print("Telegram Test - Sending Message...")
    print("=" * 60)

    bot = Bot(token=bot_token)

    message = """
🏈 **GAME SUBSCRIPTION CONFIRMED**

**Test Game: Oklahoma Sooners @ Missouri Tigers**

📅 Date: 11/22 - 12:00 PM EST
📺 Status: Live Now

You'll receive notifications for:
• Score updates
• Quarter changes
• Game status changes
• AI prediction updates

🤖 **Multi-Agent AI Analysis**
🎯 Prediction: Oklahoma -13.8
✅ 72% win probability
💡 Medium Confidence

**Powered by Magnus Trading Platform**
"""

    try:
        result = await bot.send_message(
            chat_id=chat_id,
            text=message,
            parse_mode='Markdown'
        )
        print(f"\nSUCCESS! Message sent.")
        print(f"Message ID: {result.message_id}")
        print("\nCheck your Telegram app!")

        await bot.close()  # Important: close the bot when done

    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(send_test_message())

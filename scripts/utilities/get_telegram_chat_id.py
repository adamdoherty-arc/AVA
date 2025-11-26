"""
Get Your Telegram Chat ID
==========================

This script helps you get your Telegram chat ID.

Steps:
1. Open Telegram
2. Search for your bot (the one you created with @BotFather)
3. Send any message to your bot (e.g., "/start" or "Hello")
4. Run this script to see your chat ID
5. Copy the chat ID and add it to your .env file

Usage:
    python get_telegram_chat_id.py
"""

import os
import asyncio
from dotenv import load_dotenv
from telegram import Bot

load_dotenv()

async def get_chat_id():
    """Get chat ID from recent messages"""
    bot_token = os.getenv('TELEGRAM_BOT_TOKEN')

    if not bot_token or bot_token == 'your_bot_token_here':
        print("\nERROR: TELEGRAM_BOT_TOKEN not configured in .env")
        print("\nPlease configure your bot token first:")
        print("1. Open Telegram and search for @BotFather")
        print("2. Send /newbot and follow instructions")
        print("3. Copy the bot token")
        print("4. Add to .env file: TELEGRAM_BOT_TOKEN=your_token_here")
        return

    print("=" * 70)
    print("  Get Your Telegram Chat ID")
    print("=" * 70)

    bot = Bot(token=bot_token)

    try:
        # Get bot info
        bot_info = await bot.get_me()
        print(f"\nBot Name: {bot_info.first_name}")
        print(f"Bot Username: @{bot_info.username}")

        print("\n" + "-" * 70)
        print("Instructions:")
        print("-" * 70)
        print("1. Open Telegram")
        print(f"2. Search for @{bot_info.username}")
        print("3. Send any message to your bot (e.g., /start or Hello)")
        print("4. Come back here and press Enter to get your chat ID")
        print("-" * 70)

        input("\nPress Enter after sending a message to your bot...")

        # Get updates (recent messages)
        updates = await bot.get_updates()

        if not updates:
            print("\nNo messages found!")
            print("Make sure you sent a message to your bot first.")
            print(f"Search for @{bot_info.username} in Telegram and send /start")
        else:
            print("\nFound messages! Here are your chat IDs:")
            print("=" * 70)

            seen_chats = set()
            for update in updates:
                if update.message:
                    chat_id = update.message.chat.id
                    chat_type = update.message.chat.type
                    from_user = update.message.from_user

                    if chat_id not in seen_chats:
                        seen_chats.add(chat_id)
                        print(f"\nChat ID: {chat_id}")
                        print(f"Chat Type: {chat_type}")
                        if from_user:
                            print(f"User: {from_user.first_name}")
                            if from_user.username:
                                print(f"Username: @{from_user.username}")

            print("\n" + "=" * 70)
            print("Add this to your .env file:")
            print("=" * 70)

            # Get the most recent chat ID
            latest_chat_id = updates[-1].message.chat.id
            print(f"\nTELEGRAM_CHAT_ID={latest_chat_id}")
            print("\n" + "=" * 70)

        await bot.close()

    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(get_chat_id())

"""Quick check of Telegram configuration"""
from dotenv import load_dotenv
import os

load_dotenv()

bot_token = os.getenv('TELEGRAM_BOT_TOKEN')
chat_id = os.getenv('TELEGRAM_CHAT_ID')

print("=" * 60)
print("Telegram Configuration Check")
print("=" * 60)
print(f"\nBot Token: {'Configured' if bot_token and bot_token != 'your_bot_token_here' else 'NOT configured'}")
print(f"Chat ID: {chat_id if chat_id and chat_id != 'YOUR_CHAT_ID_HERE' else 'NOT configured'}")
print()

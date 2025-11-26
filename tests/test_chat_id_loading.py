"""Test chat ID loading"""
from dotenv import load_dotenv
import os

load_dotenv(override=True)

chat_id = os.getenv('TELEGRAM_CHAT_ID')
print(f'Loaded chat ID: {chat_id}')
print(f'Type: {type(chat_id)}')
print(f'Length: {len(chat_id) if chat_id else 0}')
print(f'Is default placeholder: {chat_id == "YOUR_CHAT_ID_HERE"}')
print(f'Is valid: {chat_id and chat_id != "YOUR_CHAT_ID_HERE" and chat_id != "your_chat_id_here"}')

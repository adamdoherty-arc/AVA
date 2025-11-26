"""
Debug what user_id the Settings page is using
"""
import os
from dotenv import load_dotenv

load_dotenv()

# What the app SHOULD be using
telegram_users = os.getenv('TELEGRAM_AUTHORIZED_USERS', 'default_user')
print(f"TELEGRAM_AUTHORIZED_USERS env var: {telegram_users}")
print(f"First user (should be): {telegram_users.split(',')[0]}")

# What we subscribed with
print(f"\nSubscriptions created with user_id: 7957298119")

# Check what the app initialization code does
print("\nApp initialization code:")
print("  telegram_user_id = os.getenv('TELEGRAM_AUTHORIZED_USERS', 'default_user').split(',')[0]")
print("  st.session_state.user_id = telegram_user_id")

print("\nPotential issue:")
print("  If session state wasn't initialized, it might be using 'default_user'")
print("  instead of '7957298119'")

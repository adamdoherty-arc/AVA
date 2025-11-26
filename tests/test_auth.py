#!/usr/bin/env python
"""Test basic Robinhood authentication"""

import robin_stocks.robinhood as rh
import os
from dotenv import load_dotenv
import pickle
from pathlib import Path

# Load environment variables
load_dotenv()

print("="*60)
print("ROBINHOOD AUTHENTICATION TEST")
print("="*60)

username = os.getenv('ROBINHOOD_USERNAME')
password = os.getenv('ROBINHOOD_PASSWORD')

print(f"\nCredentials from .env:")
print(f"  Username: {username[:3]}...{username[-3:] if username else 'NOT FOUND'}")
print(f"  Password: {'*' * 8 if password else 'NOT FOUND'}")

if not username or not password:
    print("\nERROR: Missing credentials in .env file")
    exit(1)

print("\nAttempting direct login with robin_stocks...")
print("-"*40)

try:
    # Try login with store_session
    login = rh.authentication.login(
        username=username,
        password=password,
        expiresIn=86400,
        store_session=True
    )

    if login:
        print("\nSUCCESS! Logged in successfully")
        print(f"Login response keys: {login.keys() if isinstance(login, dict) else 'Not a dict'}")

        # Try to get account info
        try:
            profile = rh.profiles.load_account_profile()
            print(f"\nAccount Number: {profile.get('account_number', 'N/A')[:4]}****")
            print(f"Buying Power: ${float(profile.get('buying_power', 0)):,.2f}")
        except Exception as e:
            print(f"\nError getting account info: {e}")

        # Logout
        rh.authentication.logout()
        print("\nLogged out successfully")
    else:
        print("\nFAILED: Login returned None or False")

except Exception as e:
    print(f"\nERROR during login: {e}")
    print("\nThis could be due to:")
    print("  1. Invalid credentials")
    print("  2. Account requires MFA (check your phone/email)")
    print("  3. Network/API issues")

    # If MFA is needed, user will see a prompt
    print("\nIf you see an MFA prompt, enter the code from your authenticator app")
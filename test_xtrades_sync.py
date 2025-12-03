#!/usr/bin/env python3
"""Test script for Xtrades sync - runs in non-headless mode to capture login cookies."""

import os
import sys
import time
from dotenv import load_dotenv

load_dotenv()

from src.xtrades_scraper import XtradesScraper
from src.xtrades_db_manager import XtradesDBManager

print('='*60)
print('XTRADES SYNC TEST (Non-Headless Mode)')
print('='*60)

# Initialize scraper in non-headless mode
print('\n1. Initializing scraper (visible browser)...')
scraper = XtradesScraper(headless=False)
print('   Chrome started successfully')

# Attempt login
print('\n2. Attempting login to Xtrades.net...')
try:
    success = scraper.login()
    if success:
        print('   LOGIN SUCCESSFUL!')
        print('   Cookies saved for future headless runs')
    else:
        print('   Login returned False')
except Exception as e:
    print(f'   Login failed: {e}')
    scraper.close()
    sys.exit(1)

# Get active profiles
print('\n3. Getting active profiles...')
db = XtradesDBManager()
profiles = db.get_active_profiles()
print(f'   Found {len(profiles)} active profiles:')
for p in profiles:
    print(f'   - {p["username"]}')

# Sync first profile as test
if profiles:
    profile = profiles[0]
    print(f'\n4. Syncing profile: {profile["username"]}...')
    try:
        alerts = scraper.get_profile_alerts(profile['username'], max_alerts=10)
        print(f'   Found {len(alerts)} alerts')

        # Show what was parsed
        valid_trades = 0
        for alert in alerts[:5]:
            ticker = alert.get('ticker')
            strategy = alert.get('strategy')
            action = alert.get('action')
            price = alert.get('entry_price')
            if strategy or action or price:
                valid_trades += 1
                print(f'   - {ticker}: {strategy or "N/A"} | {action or "N/A"} | ${price or "N/A"}')
            else:
                print(f'   - {ticker}: [FILTERED - not a trade alert]')

        print(f'\n   Valid trade alerts: {valid_trades}/{len(alerts)}')
    except Exception as e:
        print(f'   Error syncing: {e}')

# Cleanup
print('\n5. Closing browser...')
scraper.close()
print('   Done!')
print('='*60)

#!/usr/bin/env python3
"""
Manual Login Helper for Xtrades.net
====================================

This script opens Chrome and waits for you to manually log in to Xtrades.net via Discord.
Once logged in, it saves your session cookies for future automated runs.

Usage:
    python manual_login_xtrades.py

After running:
    1. The browser will open to the Xtrades login page
    2. Click "Sign in with Discord"
    3. Complete the Discord login in the browser
    4. Once you see the Xtrades dashboard, press ENTER in this terminal
    5. Cookies will be saved for future automated runs
"""

import os
import sys
import time
import pickle
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src'))

import undetected_chromedriver as uc

print('='*70)
print('XTRADES MANUAL LOGIN HELPER')
print('='*70)
print('''
This script will open Chrome for you to manually log in to Xtrades.net.

Instructions:
1. Click "Sign in with Discord" on the login page
2. Complete the Discord login in the browser
3. Wait until you see the Xtrades dashboard
4. Return to this terminal and press ENTER
5. Your session cookies will be saved for automated runs
''')
print('='*70)

# Setup cache directory
cache_dir = Path.home() / '.xtrades_cache'
cache_dir.mkdir(exist_ok=True)
cookies_file = cache_dir / 'cookies.pkl'

# Initialize Chrome
print('\nStarting Chrome...')
options = uc.ChromeOptions()
options.add_argument('--start-maximized')
options.add_argument('--no-sandbox')
options.add_argument('--disable-dev-shm-usage')

# Use a persistent profile directory to retain Discord login
profile_dir = cache_dir / 'chrome_profile'
options.add_argument(f'--user-data-dir={profile_dir}')

driver = uc.Chrome(options=options, headless=False)

try:
    # Navigate to login page
    print('Opening Xtrades login page...')
    driver.get('https://app.xtrades.net/login')
    time.sleep(3)

    print('\n' + '='*70)
    print('WAITING FOR MANUAL LOGIN')
    print('='*70)
    print('''
Please complete these steps in the browser:

1. Click "Sign in with Discord" button
2. Log in with your Discord credentials if prompted
3. Click "Authorize" if Discord asks for permission
4. Wait until you see the Xtrades dashboard/feed

When you see the Xtrades dashboard, press ENTER here to save cookies...
''')

    input('>>> Press ENTER after you are logged in and see the dashboard... ')

    # Verify login
    print('\nVerifying login status...')
    current_url = driver.current_url
    print(f'Current URL: {current_url}')

    if 'login' in current_url.lower() or 'auth' in current_url.lower():
        print('\nWARNING: You still appear to be on the login page!')
        confirm = input('Are you sure you want to save cookies anyway? (y/n): ')
        if confirm.lower() != 'y':
            print('Aborting without saving cookies.')
            sys.exit(1)

    # Save cookies
    print('\nSaving cookies...')
    cookies = driver.get_cookies()

    if cookies:
        with open(cookies_file, 'wb') as f:
            pickle.dump(cookies, f)
        print(f'SUCCESS! Saved {len(cookies)} cookies to: {cookies_file}')

        # Show cookie details
        print('\nCookies saved:')
        for c in cookies:
            expiry = c.get('expiry', 'session')
            print(f"  - {c['name']} (domain: {c['domain']}, expiry: {expiry})")
    else:
        print('WARNING: No cookies found to save!')

    print('\n' + '='*70)
    print('DONE!')
    print('='*70)
    print(f'''
Your session has been saved. You can now run the automated sync:

  python src/ava/xtrades_background_sync.py --once

Or set up the scheduled sync:

  python src/ava/xtrades_background_sync.py --interval 5
''')

finally:
    input('\nPress ENTER to close the browser...')
    driver.quit()

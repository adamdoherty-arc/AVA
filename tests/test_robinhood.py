#!/usr/bin/env python
"""Test script for Robinhood integration"""

import os
from dotenv import load_dotenv
from src.robinhood_integration import RobinhoodClient

# Load environment variables
load_dotenv()

print("="*60)
print("ROBINHOOD INTEGRATION TEST")
print("="*60)

# Check if credentials are in environment
username = os.getenv('ROBINHOOD_USERNAME')
password = os.getenv('ROBINHOOD_PASSWORD')
mfa = os.getenv('ROBINHOOD_MFA_CODE')

if not username or not password:
    print("\nNo credentials found in .env file")
    print("Please enter credentials manually:\n")

    username = input("Robinhood Username/Email: ")
    password = input("Robinhood Password: ")
    mfa = input("MFA Secret Key (optional, press Enter to skip): ")
else:
    print(f"\nUsing credentials from .env file")
    print(f"Username: {username[:3]}...{username[-3:]}")

# Create client
print("\nConnecting to Robinhood...")
client = RobinhoodClient(username, password, mfa)

# Try to login
if client.login(store_session=True):
    print("SUCCESS: Logged in to Robinhood!")

    try:
        # Get account info
        print("\n" + "="*40)
        print("ACCOUNT INFORMATION")
        print("="*40)
        account = client.get_account_info()

        if account:
            print(f"Buying Power: ${account.get('buying_power', 0):,.2f}")
            print(f"Total Value: ${account.get('total_value', 0):,.2f}")
            print(f"Cash: ${account.get('cash', 0):,.2f}")
            print(f"Day Trades: {account.get('day_trade_count', 0)}/3")

        # Get wheel positions
        print("\n" + "="*40)
        print("WHEEL STRATEGY POSITIONS")
        print("="*40)
        wheel_positions = client.get_wheel_positions()

        if wheel_positions:
            for pos in wheel_positions:
                if pos['strategy'] == 'csp':
                    print(f"\nCash-Secured Put: {pos['symbol']}")
                    print(f"  Strike: ${pos['strike']}")
                    print(f"  Expiration: {pos['expiration']}")
                    print(f"  Premium: ${pos['premium_collected']:.2f}")
                    print(f"  Days to Expiry: {pos['days_to_expiry']}")

                elif pos['strategy'] == 'cc':
                    print(f"\nCovered Call: {pos['symbol']}")
                    print(f"  Strike: ${pos['strike']}")
                    print(f"  Expiration: {pos['expiration']}")
                    print(f"  Premium: ${pos['premium_collected']:.2f}")
                    print(f"  Days to Expiry: {pos['days_to_expiry']}")

                elif pos['strategy'] == 'potential_cc':
                    print(f"\nStock Position: {pos['symbol']}")
                    print(f"  Shares: {pos['shares']}")
                    print(f"  Cost Basis: ${pos['cost_basis']:.2f}")
                    print(f"  Current Price: ${pos['current_price']:.2f}")
                    print(f"  P&L: ${pos['unrealized_pnl']:.2f} ({pos['unrealized_pnl_pct']:.1f}%)")
                    print(f"  Covered Calls Available: {pos['contracts_available']}")
        else:
            print("No wheel strategy positions found")

        # Get all positions summary
        print("\n" + "="*40)
        print("ALL POSITIONS SUMMARY")
        print("="*40)
        all_positions = client.get_all_positions()

        print(f"Stock Positions: {len(all_positions['stocks'])}")
        print(f"Option Positions: {len(all_positions['options'])}")

        # Show stock symbols
        if all_positions['stocks']:
            symbols = [s['symbol'] for s in all_positions['stocks']]
            print(f"Stocks Owned: {', '.join(symbols[:10])}")

    except Exception as e:
        print(f"\nError getting data: {e}")

    finally:
        # Logout
        client.logout()
        print("\n" + "="*40)
        print("Logged out from Robinhood")
        print("="*40)

else:
    print("\nFAILED: Could not login to Robinhood")
    print("\nTroubleshooting:")
    print("1. Check your username and password")
    print("2. If you have 2FA enabled, you need to provide the MFA secret key")
    print("3. The MFA secret is the long string when setting up 2FA, not the 6-digit code")
    print("4. You may need to approve the login from your email")

print("\nTo use in the dashboard:")
print("1. Run: python launch.py")
print("2. Go to the Positions page")
print("3. Click 'Connect to Robinhood'")
print("4. Enter your credentials")
print("5. Your real positions will load automatically")
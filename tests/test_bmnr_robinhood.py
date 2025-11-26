"""
Test if BMNR has options available on Robinhood
"""

import os
import robin_stocks.robinhood as rh

print("="*80)
print("Testing BMNR Options on Robinhood")
print("="*80)

# Login to Robinhood
print("\n1. Logging into Robinhood...")
try:
    username = os.getenv('ROBINHOOD_USERNAME')
    password = os.getenv('ROBINHOOD_PASSWORD')

    if not username or not password:
        print("   ERROR: Robinhood credentials not found in environment")
        exit(1)

    rh.authentication.login(
        username=username,
        password=password,
        expiresIn=86400,
        store_session=True
    )
    print("   SUCCESS: Logged into Robinhood")

except Exception as e:
    print(f"   ERROR: Login failed: {e}")
    exit(1)

# Check if BMNR has options chains
print("\n2. Checking if BMNR has options available...")
try:
    chains = rh.options.get_chains('BMNR')
    print(f"   Result: {chains}")

    if chains and 'expiration_dates' in chains:
        exp_dates = chains['expiration_dates']
        print(f"   SUCCESS: Found {len(exp_dates)} expiration dates")
        print(f"   First 5 expirations: {exp_dates[:5]}")

        # Try to get options for first expiration
        if exp_dates:
            first_exp = exp_dates[0]
            print(f"\n3. Getting options for {first_exp}...")

            puts = rh.options.find_options_by_expiration(
                'BMNR',
                first_exp,
                optionType='put'
            )

            if puts:
                print(f"   SUCCESS: Found {len(puts)} put options")
                print(f"   Sample strikes: {[opt.get('strike_price') for opt in puts[:5]]}")
            else:
                print("   ERROR: No put options found for this expiration")
    else:
        print("   ERROR: No expiration dates found - BMNR likely doesn't have options trading")
        print("   This means BMNR is not optionable on Robinhood")

except Exception as e:
    print(f"   ERROR: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "="*80)
print("Diagnostic Complete")
print("="*80)

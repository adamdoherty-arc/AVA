"""Test script to verify Kalshi API connectivity"""

import requests
import json

def test_kalshi_api():
    """Test different Kalshi API endpoints"""

    # Try different base URLs
    urls_to_test = [
        "https://api.elections.kalshi.com/trade-api/v2",
        "https://trading-api.kalshi.com/trade-api/v2",
        "https://api.kalshi.com/trade-api/v2"
    ]

    for base_url in urls_to_test:
        print(f"\n{'='*80}")
        print(f"Testing: {base_url}")
        print('='*80)

        try:
            # Test markets endpoint
            url = f"{base_url}/markets"
            params = {'limit': 5, 'status': 'active'}

            print(f"GET {url}")
            print(f"Params: {params}")

            response = requests.get(url, params=params, timeout=10)
            print(f"Status Code: {response.status_code}")
            print(f"Headers: {dict(response.headers)}")

            if response.status_code == 200:
                data = response.json()
                print(f"✓ SUCCESS!")
                print(f"Response keys: {list(data.keys())}")

                markets = data.get('markets', [])
                print(f"Markets found: {len(markets)}")

                if markets:
                    print(f"\nFirst market sample:")
                    print(json.dumps(markets[0], indent=2))
                    return base_url  # Found working URL
            else:
                print(f"✗ FAILED")
                print(f"Response: {response.text[:500]}")

        except Exception as e:
            print(f"✗ ERROR: {e}")

    return None

if __name__ == "__main__":
    print("Kalshi API Connectivity Test")
    print("="*80)

    working_url = test_kalshi_api()

    if working_url:
        print(f"\n{'='*80}")
        print(f"✓ WORKING URL FOUND: {working_url}")
        print('='*80)
    else:
        print(f"\n{'='*80}")
        print("✗ NO WORKING URL FOUND - Kalshi API may be down or changed")
        print('='*80)

"""Quick test of FRED API key"""
import os
os.environ['FRED_API_KEY'] = '5745785754da757bae8c70bcccfd2c1c'

import requests

# Test FRED API directly
api_key = '5745785754da757bae8c70bcccfd2c1c'
url = f"https://api.stlouisfed.org/fred/series/observations?series_id=UNRATE&api_key={api_key}&file_type=json&limit=1&sort_order=desc"

print("Testing FRED API with your key...")
print(f"API Key: {api_key[:8]}...")
print()

try:
    response = requests.get(url, timeout=10)
    print(f"Status Code: {response.status_code}")

    if response.status_code == 200:
        data = response.json()
        if 'observations' in data and len(data['observations']) > 0:
            obs = data['observations'][0]
            print(f"SUCCESS! FRED API is working!")
            print()
            print(f"Latest Unemployment Rate: {obs['value']}%")
            print(f"Date: {obs['date']}")
            print()
            print("=" * 60)
            print("YOUR FRED API KEY IS ACTIVE AND WORKING!")
            print("=" * 60)
            print()
            print("You now have access to:")
            print("  - 820,000+ economic indicators")
            print("  - UNLIMITED API calls")
            print("  - $0 cost forever")
            print()
            print("AVA can now use:")
            print("  - GDP, inflation, unemployment data")
            print("  - Federal Funds Rate")
            print("  - VIX (market volatility)")
            print("  - Recession risk indicators")
            print("  - Market regime detection")
            print()
            print("Next step: Test the full integration!")
            print("  python src/ava/world_class_ava_integration.py")
        else:
            print("ERROR: Unexpected response format")
            print(data)
    else:
        print(f"ERROR: Request failed with status {response.status_code}")
        print(response.text)

except Exception as e:
    print(f"ERROR: {e}")

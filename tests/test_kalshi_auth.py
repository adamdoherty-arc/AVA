"""
Test Kalshi API Authentication
Diagnose why login is failing with provided credentials
"""
import os
import requests
import json
from dotenv import load_dotenv

load_dotenv(override=True)

print("=" * 80)
print("TESTING KALSHI AUTHENTICATION")
print("=" * 80)

# Get credentials
email = os.getenv('KALSHI_EMAIL')
password = os.getenv('KALSHI_PASSWORD')

print(f"\n1. Credentials Check:")
print(f"   Email: {email}")
print(f"   Password: {'*' * len(password) if password else 'NOT SET'}")

# Test trading API endpoint
print(f"\n2. Testing Trading API Login:")
print(f"   URL: https://trading-api.kalshi.com/trade-api/v2/login")

try:
    url = "https://trading-api.kalshi.com/trade-api/v2/login"
    headers = {
        "accept": "application/json",
        "content-type": "application/json"
    }
    body = {
        "email": email,
        "password": password
    }

    response = requests.post(url, headers=headers, json=body, timeout=10)

    print(f"   Status Code: {response.status_code}")
    print(f"   Response Headers: {dict(response.headers)}")

    if response.status_code == 200:
        print("   [SUCCESS] Login successful!")
        data = response.json()
        print(f"   Token received: {data.get('token', 'No token')[:30]}...")
    else:
        print(f"   [FAILED] Login failed")
        print(f"   Response: {response.text}")

except Exception as e:
    print(f"   [ERROR] Exception: {e}")

# Test public elections API
print(f"\n3. Testing Public Elections API:")
print(f"   URL: https://api.elections.kalshi.com/trade-api/v2/markets")

try:
    url = "https://api.elections.kalshi.com/trade-api/v2/markets"
    params = {'limit': 5}

    response = requests.get(url, params=params, timeout=10)

    print(f"   Status Code: {response.status_code}")

    if response.status_code == 200:
        print("   [SUCCESS] Public API accessible!")
        data = response.json()
        markets = data.get('markets', [])
        print(f"   Markets fetched: {len(markets)}")
        if markets:
            print(f"   Sample market: {markets[0].get('title', 'Unknown')[:60]}...")
    else:
        print(f"   [FAILED] Public API error")
        print(f"   Response: {response.text[:200]}")

except Exception as e:
    print(f"   [ERROR] Exception: {e}")

# Test API key method (if available)
print(f"\n4. Testing API Key Method:")
api_key = os.getenv('KALSHI_API_KEY')
private_key_path = os.getenv('KALSHI_PRIVATE_KEY_PATH')

print(f"   API Key: {api_key[:20] if api_key else 'NOT SET'}...")
print(f"   Private Key Path: {private_key_path if private_key_path else 'NOT SET'}")

if api_key and private_key_path:
    print("   [INFO] API key method available - could implement if email/password fails")
else:
    print("   [INFO] API key method not configured")

print("\n" + "=" * 80)
print("DIAGNOSIS COMPLETE")
print("=" * 80)

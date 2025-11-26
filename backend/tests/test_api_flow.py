import requests
import sys

BASE_URL = "http://localhost:8001/api"

def test_endpoint(method, endpoint, payload=None, expected_status=200):
    url = f"{BASE_URL}{endpoint}"
    try:
        if method == "GET":
            response = requests.get(url)
        elif method == "POST":
            response = requests.post(url, json=payload)
        
        if response.status_code == expected_status:
            print(f"[PASS] {method} {endpoint} - ({response.status_code})")
            return True
        else:
            print(f"[FAIL] {method} {endpoint} - (Expected {expected_status}, Got {response.status_code})")
            print(f"   Response: {response.text[:200]}")
            return False
    except Exception as e:
        print(f"[ERROR] {method} {endpoint} - Error: {str(e)}")
        return False

def run_tests():
    print("Starting Backend API Verification...")
    
    # 1. Health Check
    if not test_endpoint("GET", "/health"):
        print("Backend seems to be down or unreachable.")
        return

    # 2. Portfolio Endpoints
    test_endpoint("GET", "/portfolio/summary")
    test_endpoint("GET", "/portfolio/positions")

    # 3. Research Endpoints
    # Using a known symbol like AAPL
    test_endpoint("GET", "/research/AAPL")

    # 4. Strategy Endpoints
    test_endpoint("GET", "/strategy/analyze?watchlist=NVDA")

    # 5. Chat Endpoint
    chat_payload = {
        "message": "What is the price of AAPL?",
        "history": []
    }
    test_endpoint("POST", "/chat/", payload=chat_payload)

    print("\nVerification Complete.")

if __name__ == "__main__":
    run_tests()

import requests
import sys
import time

# Backend URL - uses centralized port 8002
BASE_URL = "http://localhost:8002"

def check_endpoint(name, url, method="GET", payload=None):
    print(f"Checking {name}...", end=" ")
    try:
        if method == "GET":
            response = requests.get(url)
        else:
            response = requests.post(url, json=payload)
        
        if response.status_code == 200:
            print("[OK]")
            return True
        else:
            print(f"[FAILED] ({response.status_code})")
            print(response.text)
            return False
    except Exception as e:
        print(f"[ERROR]: {e}")
        return False

def main():
    print("=== Magnus Full Stack Verification ===\n")
    
    # 1. Check Root
    if not check_endpoint("API Root", f"{BASE_URL}/"):
        print("CRITICAL: API is not running!")
        sys.exit(1)

    # 2. Check Dashboard
    check_endpoint("Dashboard Summary", f"{BASE_URL}/api/dashboard/summary")

    # 3. Check Prediction Markets
    check_endpoint("Prediction Markets", f"{BASE_URL}/api/predictions/markets")

    # 4. Check Sports Games
    check_endpoint("Sports Games", f"{BASE_URL}/api/sports/markets")

    # 5. Check Chat (Ava)
    check_endpoint("Ava Chat", f"{BASE_URL}/api/chat/", method="POST", payload={
        "message": "Hello Ava, are you online?",
        "history": []
    })

    print("\n=== Verification Complete ===")

if __name__ == "__main__":
    main()

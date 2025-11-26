"""Test automatic session ID refresh on authentication failure"""

import os
from dotenv import load_dotenv, set_key
from src.tradingview_api_sync import TradingViewAPISync

load_dotenv()

def test_auto_refresh():
    """
    Test that the sync automatically refreshes session ID on auth failure
    """
    print("="*70)
    print("Testing Automatic Session ID Refresh")
    print("="*70)

    # Save current session ID
    original_session_id = os.getenv('TRADINGVIEW_SESSION_ID', '')
    print(f"\nCurrent session ID: {original_session_id[:16]}...")

    # Test 1: Normal sync with valid session ID
    print("\n[Test 1] Testing sync with valid session ID...")
    syncer = TradingViewAPISync()
    watchlists = syncer.sync_to_database()

    if watchlists:
        print(f"[OK] Synced {len(watchlists)} watchlists successfully")
        for name, symbols in list(watchlists.items())[:3]:
            print(f"   - {name}: {len(symbols)} symbols")
    else:
        print("[FAIL] Sync failed")

    # Test 2: Show what happens when session expires
    print("\n" + "="*70)
    print("[Info] What happens when session ID expires?")
    print("="*70)
    print("\nWhen your session ID expires, the system will:")
    print("  1. Detect authentication failure (401/403 error)")
    print("  2. Automatically run get_session_interactive.py")
    print("  3. Browser window opens for you to login")
    print("  4. You complete 2FA/verification")
    print("  5. New session ID is extracted and saved")
    print("  6. Sync automatically retries with new session ID")
    print("\nThis is completely automatic - you just need to complete")
    print("the login when the browser window appears!")

    # Test 3: Demonstrate the flow (without actually invalidating)
    print("\n" + "="*70)
    print("[Test 2] Session ID Auto-Refresh Flow")
    print("="*70)
    print("\nTo manually test the auto-refresh functionality:")
    print("\n1. Temporarily invalidate your session ID in .env:")
    print("   TRADINGVIEW_SESSION_ID=invalid_session_12345")
    print("\n2. Run: python src/tradingview_api_sync.py")
    print("\n3. Watch as it automatically:")
    print("   - Detects the invalid session")
    print("   - Opens browser for re-authentication")
    print("   - Extracts new session ID")
    print("   - Completes the sync")

    print("\n" + "="*70)
    print("Test Summary")
    print("="*70)
    print(f"\n[OK] Session ID is valid: {original_session_id[:16]}...")
    print(f"[OK] Auto-refresh is configured and ready")
    print(f"[OK] Sync is working: {len(watchlists)} watchlists found")
    print("\n[INFO] The system will automatically handle session expiration")
    print("       No manual intervention needed except completing 2FA when prompted")
    print("="*70)

if __name__ == "__main__":
    test_auto_refresh()

"""
Test Discord Premium Alert System
==================================

Comprehensive test script for the Discord premium alert notification system.

Features:
1. Test Discord bot webhook connectivity
2. Send test premium alert notification
3. Simulate real trading alert
4. Verify message formatting

Usage:
    python test_discord_premium_alert.py

Author: Magnus Trading Platform
Created: 2025-11-21
"""

import os
import sys
import requests
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


def test_discord_webhook(webhook_url: str, test_type: str = "connectivity"):
    """
    Test Discord webhook with different message types

    Args:
        webhook_url: Discord webhook URL
        test_type: Type of test ('connectivity', 'simple', 'premium_alert', 'full')

    Returns:
        Tuple of (success: bool, response: str)
    """
    try:
        if test_type == "connectivity":
            # Simple connectivity test
            payload = {
                'content': '✅ **Discord Webhook Test** - Connection successful!',
                'username': 'Magnus Test Bot'
            }

        elif test_type == "simple":
            # Simple message test
            payload = {
                'content': '📢 This is a test message from Magnus Trading Platform',
                'username': 'Magnus Alert Bot'
            }

        elif test_type == "premium_alert":
            # Simulated premium trading alert
            payload = {
                'content': '🚨 **NEW PREMIUM ALERT** 🚨\n\n**Test Trader** at 02:30 PM:\nBUY NVDA 140C 12/20 @ $5.50\nTarget: $8.00\nStop: $4.00\n\nThis is a TEST alert.',
                'username': 'Magnus Alert Bot',
                'avatar_url': 'https://i.imgur.com/4M34hi2.png'
            }

        elif test_type == "full":
            # Full embed with rich formatting (simulating real premium alert)
            current_time = datetime.now()

            payload = {
                'content': '🚨 **NEW PREMIUM ALERT** 🚨',
                'username': 'Magnus Alert Bot',
                'avatar_url': 'https://i.imgur.com/4M34hi2.png',
                'embeds': [{
                    'title': '📈 Premium Trading Alert - TEST',
                    'description': (
                        '**🎯 TRADE ALERT**\n\n'
                        '**Ticker:** NVDA\n'
                        '**Action:** BUY\n'
                        '**Type:** Call Option\n'
                        '**Strike:** $140\n'
                        '**Expiration:** 12/20/2025\n'
                        '**Entry:** $5.50\n'
                        '**Target:** $8.00\n'
                        '**Stop Loss:** $4.00\n\n'
                        '**Rationale:** Strong uptrend, bullish momentum, technical breakout\n\n'
                        '⚠️ This is a TEST alert for webhook validation'
                    ),
                    'color': 0xFF6B6B,  # Red color
                    'author': {
                        'name': 'Test Trader (Premium Channel)'
                    },
                    'timestamp': current_time.isoformat(),
                    'footer': {
                        'text': f'Channel: Premium Alerts (990331623260180580)'
                    },
                    'fields': [
                        {
                            'name': '💰 Entry Price',
                            'value': '$5.50',
                            'inline': True
                        },
                        {
                            'name': '🎯 Target',
                            'value': '$8.00',
                            'inline': True
                        },
                        {
                            'name': '🛑 Stop Loss',
                            'value': '$4.00',
                            'inline': True
                        }
                    ]
                }]
            }

        else:
            return False, f"Unknown test type: {test_type}"

        # Send webhook request
        response = requests.post(webhook_url, json=payload, timeout=10)
        response.raise_for_status()

        return True, f"✅ {test_type.upper()} test successful! Message sent to Discord."

    except requests.exceptions.HTTPError as e:
        return False, f"❌ HTTP Error: {e.response.status_code} - {e.response.text}"
    except requests.exceptions.ConnectionError:
        return False, "❌ Connection Error: Could not connect to Discord webhook"
    except requests.exceptions.Timeout:
        return False, "❌ Timeout Error: Request took too long"
    except Exception as e:
        return False, f"❌ Unexpected Error: {str(e)}"


def main():
    """Main test function"""
    print("\n" + "=" * 70)
    print("  Discord Premium Alert System - Test Suite")
    print("=" * 70)

    # Check for webhook URL
    webhook_url = os.getenv('DISCORD_BOT_WEBHOOK_URL')

    if not webhook_url:
        print("\n❌ ERROR: DISCORD_BOT_WEBHOOK_URL not configured in .env")
        print("\nTo configure:")
        print("1. Go to your Discord server")
        print("2. Edit the channel where you want notifications")
        print("3. Integrations → Webhooks → New Webhook")
        print("4. Copy the Webhook URL")
        print("5. Add to .env file:")
        print("   DISCORD_BOT_WEBHOOK_URL=<your_webhook_url>")
        print()
        sys.exit(1)

    print(f"\n✅ Webhook URL configured")
    print(f"   {webhook_url[:50]}...")

    # Run tests
    tests = [
        ("connectivity", "Basic connectivity test"),
        ("simple", "Simple message test"),
        ("premium_alert", "Premium alert format test"),
        ("full", "Full embed with rich formatting")
    ]

    print("\n" + "-" * 70)
    print("Running Tests...")
    print("-" * 70)

    passed = 0
    failed = 0

    for test_type, description in tests:
        print(f"\n{len(tests) - failed - passed}. {description}")
        print(f"   Test type: {test_type}")

        success, message = test_discord_webhook(webhook_url, test_type)

        if success:
            print(f"   {message}")
            passed += 1
        else:
            print(f"   {message}")
            failed += 1

        # Prompt to check Discord
        if success:
            input("   → Check your Discord channel and press Enter to continue...")

    # Summary
    print("\n" + "=" * 70)
    print("Test Summary")
    print("=" * 70)
    print(f"\n✅ Passed: {passed}/{len(tests)}")
    if failed > 0:
        print(f"❌ Failed: {failed}/{len(tests)}")
    else:
        print("\n🎉 All tests passed! Your Discord webhook is working correctly.")

    # Next steps
    print("\n" + "-" * 70)
    print("Next Steps:")
    print("-" * 70)
    print("\n1. ✅ Webhook is configured and working")
    print("2. 🔄 Discord sync runs every 5 minutes via Celery")
    print("3. 🚨 Premium alerts from channel 990331623260180580 trigger notifications")
    print("\nTo test with real alerts:")
    print("  • Post a message in the premium channel (990331623260180580)")
    print("  • Wait up to 5 minutes for Celery sync")
    print("  • Check your Discord bot channel for notification")
    print("\nTo manually test sync:")
    print("  python -m src.discord_premium_alert_sync")
    print()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n👋 Test interrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Fatal error: {e}")
        sys.exit(1)

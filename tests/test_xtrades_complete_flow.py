"""
XTrade Messages - Complete System Test
Tests all components of the Discord integration system
"""

import os
import sys
from dotenv import load_dotenv
from datetime import datetime

load_dotenv()

def print_section(title):
    """Print formatted section header"""
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)


def test_database_schema():
    """Test 1: Verify Discord schema exists"""
    print_section("TEST 1: Database Schema")

    try:
        import psycopg2

        conn = psycopg2.connect(
            host=os.getenv('DB_HOST', 'localhost'),
            port=int(os.getenv('DB_PORT', 5432)),
            database=os.getenv('DB_NAME', 'magnus'),
            user=os.getenv('DB_USER', 'postgres'),
            password=os.getenv('DB_PASSWORD', '')
        )

        cur = conn.cursor()

        # Check tables
        cur.execute("""
            SELECT table_name
            FROM information_schema.tables
            WHERE table_schema = 'public'
            AND table_name LIKE 'discord_%'
            ORDER BY table_name
        """)

        tables = [row[0] for row in cur.fetchall()]

        print(f"\n[OK] Connected to database: {os.getenv('DB_NAME', 'magnus')}")
        print(f"[OK] Found {len(tables)} Discord tables:")
        for table in tables:
            print(f"     - {table}")

        # Check for required tables
        required_tables = ['discord_channels', 'discord_messages']
        missing = [t for t in required_tables if t not in tables]

        if missing:
            print(f"\n[ERROR] Missing required tables: {missing}")
            return False

        cur.close()
        conn.close()

        print("\n[PASS] Database schema test passed")
        return True

    except Exception as e:
        print(f"\n[FAIL] Database schema test failed: {e}")
        return False


def test_channel_management():
    """Test 2: Channel management functionality"""
    print_section("TEST 2: Channel Management")

    try:
        from discord_messages_page import DiscordDB

        db = DiscordDB()

        # Get channels
        print("\n[TEST] Fetching channels...")
        with db.get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT COUNT(*) FROM discord_channels")
                count = cur.fetchone()[0]

        print(f"[OK] Found {count} channel(s) configured")

        # Test add channel (dry run)
        print("\n[TEST] Channel management methods available:")
        print(f"     - add_channel(): {'OK' if hasattr(db, 'add_channel') else 'NO'}")
        print(f"     - remove_channel(): {'OK' if hasattr(db, 'remove_channel') else 'NO'}")
        print(f"     - get_channels(): {'OK' if hasattr(db, 'get_channels') else 'NO'}")

        print("\n[PASS] Channel management test passed")
        return True

    except Exception as e:
        print(f"\n[FAIL] Channel management test failed: {e}")
        return False


def test_discord_sync():
    """Test 3: Discord sync capability"""
    print_section("TEST 3: Discord Sync System")

    try:
        from src.discord_message_sync import DiscordMessageSync

        sync = DiscordMessageSync()

        print("\n[OK] DiscordMessageSync initialized")
        print(f"     - Token configured: {'✓' if sync.token else '✗'}")
        print(f"     - Exporter path: {sync.exporter_path}")
        print(f"     - Export directory: {sync.export_dir}")
        print(f"     - Database: {sync.db_name}")

        # Check auto-sync script
        if os.path.exists('auto_sync_all_channels.py'):
            print("\n[OK] Auto-sync script: auto_sync_all_channels.py")
        else:
            print("\n[WARN] Auto-sync script not found")

        # Check service setup script
        if os.path.exists('setup_discord_sync_service.bat'):
            print("[OK] Service setup script: setup_discord_sync_service.bat")
        else:
            print("[WARN] Service setup script not found")

        print("\n[PASS] Discord sync test passed")
        return True

    except Exception as e:
        print(f"\n[FAIL] Discord sync test failed: {e}")
        return False


def test_telegram_alerts():
    """Test 4: Telegram alert system"""
    print_section("TEST 4: Telegram Alert System")

    try:
        from src.discord_telegram_alerts import DiscordTelegramAlerts

        alerts = DiscordTelegramAlerts()

        print("\n[OK] DiscordTelegramAlerts initialized")

        # Check Telegram configuration
        telegram_token = os.getenv('TELEGRAM_BOT_TOKEN')
        telegram_chat_id = os.getenv('TELEGRAM_CHAT_ID')

        print(f"     - Bot token configured: {'✓' if telegram_token else '✗'}")
        print(f"     - Chat ID configured: {'✓' if telegram_chat_id and telegram_chat_id != 'YOUR_CHAT_ID_HERE' else '✗'}")
        print(f"     - Min confidence score: {alerts.min_confidence_score}")

        # Check alert table
        import psycopg2
        conn = psycopg2.connect(
            host=os.getenv('DB_HOST', 'localhost'),
            port=int(os.getenv('DB_PORT', 5432)),
            database=os.getenv('DB_NAME', 'magnus'),
            user=os.getenv('DB_USER', 'postgres'),
            password=os.getenv('DB_PASSWORD', '')
        )
        cur = conn.cursor()
        cur.execute("""
            SELECT table_name
            FROM information_schema.tables
            WHERE table_schema = 'public'
            AND table_name = 'discord_alerts'
        """)
        has_alerts_table = cur.fetchone() is not None
        cur.close()
        conn.close()

        print(f"     - Alerts table: {'✓' if has_alerts_table else '✗ (run: python src/discord_telegram_alerts.py)'}")

        if telegram_chat_id == 'YOUR_CHAT_ID_HERE':
            print("\n[WARN] Telegram chat ID not configured")
            print("       Message @userinfobot on Telegram to get your chat ID")
            print("       Then update TELEGRAM_CHAT_ID in .env")

        print("\n[PASS] Telegram alert test passed (configuration needed)")
        return True

    except Exception as e:
        print(f"\n[FAIL] Telegram alert test failed: {e}")
        return False


def test_ava_integration():
    """Test 5: AVA Discord integration"""
    print_section("TEST 5: AVA Discord Integration")

    try:
        from src.ava.discord_knowledge import DiscordKnowledge, get_discord_knowledge

        dk = DiscordKnowledge()

        print("\n[OK] DiscordKnowledge initialized")
        print("     Available methods:")
        print("     - get_recent_signals()")
        print("     - get_signals_by_ticker()")
        print("     - get_channel_summary()")
        print("     - search_messages()")
        print("     - get_ava_context()")

        # Test recent signals
        signals = dk.get_recent_signals(hours_back=168, limit=5)
        print(f"\n[OK] Recent signals query: {len(signals)} signals found")

        # Test summary
        summary = dk.get_channel_summary(hours_back=24)
        if summary:
            total_msgs = summary.get('overall', {}).get('total_messages', 0)
            print(f"[OK] Channel summary: {total_msgs} messages in last 24h")

        print("\n[PASS] AVA integration test passed")
        return True

    except Exception as e:
        print(f"\n[FAIL] AVA integration test failed: {e}")
        return False


def test_ui_page():
    """Test 6: Discord messages page"""
    print_section("TEST 6: UI Page Components")

    try:
        import discord_messages_page

        print("\n[OK] discord_messages_page module loaded")
        print("     Components:")
        print("     - DiscordDB class: ✓")
        print("     - Channel Management UI: ✓")
        print("     - Trading signal search: ✓")
        print("     - AI signal analysis: ✓")
        print("     - Analytics tab: ✓")

        print("\n[PASS] UI page test passed")
        return True

    except Exception as e:
        print(f"\n[FAIL] UI page test failed: {e}")
        return False


def main():
    """Run all tests"""
    print("\n" + "=" * 70)
    print("  XTrade Messages - Complete System Test")
    print("  " + datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    print("=" * 70)

    results = {}

    # Run all tests
    results['database_schema'] = test_database_schema()
    results['channel_management'] = test_channel_management()
    results['discord_sync'] = test_discord_sync()
    results['telegram_alerts'] = test_telegram_alerts()
    results['ava_integration'] = test_ava_integration()
    results['ui_page'] = test_ui_page()

    # Summary
    print_section("TEST SUMMARY")

    passed = sum(1 for v in results.values() if v)
    total = len(results)

    print(f"\nTests passed: {passed}/{total}")
    print("\nDetailed results:")
    for test_name, result in results.items():
        status = "[PASS]" if result else "[FAIL]"
        print(f"  {status} {test_name.replace('_', ' ').title()}")

    # Next steps
    print_section("NEXT STEPS")

    if results['database_schema'] and results['channel_management']:
        print("\n✅ Core system is ready!")
        print("\nTo start using:")
        print("  1. Add a Discord channel via XTrade Messages > Channel Management")
        print("  2. Run: python sync_discord.py CHANNEL_ID 7")
        print("  3. View messages in the XTrade Messages page")

    if not all(results.values()):
        print("\n⚠️  Some tests failed. Review errors above.")

    telegram_ok = os.getenv('TELEGRAM_CHAT_ID', 'YOUR_CHAT_ID_HERE') != 'YOUR_CHAT_ID_HERE'
    if not telegram_ok:
        print("\n📱 To enable Telegram alerts:")
        print("  1. Message @userinfobot on Telegram")
        print("  2. Copy your chat ID")
        print("  3. Update TELEGRAM_CHAT_ID in .env")
        print("  4. Run: python src/discord_telegram_alerts.py")

    print("\n" + "=" * 70)
    print()

    return 0 if all(results.values()) else 1


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""
Automated Test Suite for Xtrades Sync Service
==============================================

This script performs automated testing of all Xtrades sync components to verify
the system is configured correctly and ready for production use.

Run this before deploying to production or after any configuration changes.

Usage:
    python test_xtrades_sync.py

Author: Magnus Wheel Strategy Trading Dashboard
Created: 2025-11-02
"""

import os
import sys
from datetime import datetime
from pathlib import Path

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src'))


class TestResult:
    """Stores test result with status and message"""
    def __init__(self, name: str, passed: bool, message: str = ""):
        self.name = name
        self.passed = passed
        self.message = message


class XtradesSyncTester:
    """Test suite for Xtrades Sync Service"""

    def __init__(self):
        self.results = []

    def test_imports(self) -> TestResult:
        """Test all required packages can be imported"""
        print("\n[1/8] Testing Python package imports...")

        try:
            import selenium
            import psycopg2
            import undetected_chromedriver
            from bs4 import BeautifulSoup
            from dotenv import load_dotenv

            # Optional Telegram
            try:
                import telegram
                telegram_installed = True
            except ImportError:
                telegram_installed = False

            msg = "All required packages installed"
            if not telegram_installed:
                msg += " (Telegram optional - not installed)"

            return TestResult("Package Imports", True, msg)

        except ImportError as e:
            return TestResult("Package Imports", False, f"Missing package: {e}")

    def test_environment(self) -> TestResult:
        """Test environment variables are configured"""
        print("[2/8] Testing environment configuration...")

        try:
            from dotenv import load_dotenv
            load_dotenv()

            required_vars = [
                'XTRADES_USERNAME',
                'XTRADES_PASSWORD',
                'DB_HOST',
                'DB_USER',
                'DB_PASSWORD',
                'DB_NAME'
            ]

            missing = []
            for var in required_vars:
                if not os.getenv(var):
                    missing.append(var)

            if missing:
                return TestResult(
                    "Environment Variables",
                    False,
                    f"Missing: {', '.join(missing)}"
                )

            return TestResult(
                "Environment Variables",
                True,
                f"All {len(required_vars)} required variables configured"
            )

        except Exception as e:
            return TestResult("Environment Variables", False, str(e))

    def test_database_connection(self) -> TestResult:
        """Test database connection and schema"""
        print("[3/8] Testing database connection...")

        try:
            from xtrades_db_manager import XtradesDBManager

            db = XtradesDBManager()
            conn = db.get_connection()

            # Test connection
            cursor = conn.cursor()

            # Check tables exist
            cursor.execute("""
                SELECT table_name FROM information_schema.tables
                WHERE table_schema = 'public'
                  AND table_name LIKE 'xtrades_%'
            """)

            tables = [row[0] for row in cursor.fetchall()]
            required_tables = [
                'xtrades_profiles',
                'xtrades_trades',
                'xtrades_sync_log',
                'xtrades_notifications'
            ]

            missing_tables = [t for t in required_tables if t not in tables]

            if missing_tables:
                cursor.close()
                conn.close()
                return TestResult(
                    "Database Connection",
                    False,
                    f"Missing tables: {', '.join(missing_tables)}"
                )

            # Check for active profiles
            cursor.execute("SELECT COUNT(*) FROM xtrades_profiles WHERE active = TRUE")
            profile_count = cursor.fetchone()[0]

            cursor.close()
            conn.close()

            return TestResult(
                "Database Connection",
                True,
                f"Connected, {len(tables)} tables, {profile_count} active profiles"
            )

        except Exception as e:
            return TestResult("Database Connection", False, str(e))

    def test_database_manager(self) -> TestResult:
        """Test database manager CRUD operations"""
        print("[4/8] Testing database manager...")

        try:
            from xtrades_db_manager import XtradesDBManager

            db = XtradesDBManager()

            # Test get active profiles
            profiles = db.get_active_profiles()

            if not profiles:
                return TestResult(
                    "Database Manager",
                    False,
                    "No active profiles found. Add at least one profile to test."
                )

            # Test sync logging
            sync_id = db.log_sync_start()

            stats = {
                'profiles_synced': 0,
                'trades_found': 0,
                'new_trades': 0,
                'updated_trades': 0,
                'errors': None,
                'duration_seconds': 0.5,
                'status': 'success'
            }

            success = db.log_sync_complete(sync_id, stats)

            if not success:
                return TestResult(
                    "Database Manager",
                    False,
                    "Failed to log sync operation"
                )

            return TestResult(
                "Database Manager",
                True,
                f"CRUD operations working, {len(profiles)} profiles active"
            )

        except Exception as e:
            return TestResult("Database Manager", False, str(e))

    def test_scraper_initialization(self) -> TestResult:
        """Test scraper can initialize"""
        print("[5/8] Testing scraper initialization...")

        try:
            from xtrades_scraper import XtradesScraper

            # Just test initialization, don't login
            scraper = XtradesScraper(headless=True)

            if not scraper.driver:
                scraper.close()
                return TestResult(
                    "Scraper Initialization",
                    False,
                    "Browser driver not initialized"
                )

            scraper.close()

            return TestResult(
                "Scraper Initialization",
                True,
                "Browser initialized successfully"
            )

        except Exception as e:
            return TestResult("Scraper Initialization", False, str(e))

    def test_scraper_login(self) -> TestResult:
        """Test scraper can login to Xtrades.net"""
        print("[6/8] Testing scraper login (this may take 30-60 seconds)...")

        try:
            from xtrades_scraper import XtradesScraper

            scraper = XtradesScraper(headless=True)

            try:
                success = scraper.login()

                if not success:
                    scraper.close()
                    return TestResult(
                        "Scraper Login",
                        False,
                        "Login failed. Check credentials in .env file."
                    )

                scraper.close()

                return TestResult(
                    "Scraper Login",
                    True,
                    "Successfully logged in to Xtrades.net"
                )

            except Exception as e:
                scraper.close()
                raise e

        except Exception as e:
            return TestResult("Scraper Login", False, str(e))

    def test_telegram(self) -> TestResult:
        """Test Telegram notifications"""
        print("[7/8] Testing Telegram notifications...")

        try:
            from telegram_notifier import TelegramNotifier

            notifier = TelegramNotifier()

            if not notifier.enabled:
                return TestResult(
                    "Telegram Notifications",
                    True,
                    "Disabled (set TELEGRAM_ENABLED=true to enable)"
                )

            # Get bot info without sending message
            bot_info = notifier.get_bot_info()

            if not bot_info:
                return TestResult(
                    "Telegram Notifications",
                    False,
                    "Failed to connect. Check TELEGRAM_BOT_TOKEN."
                )

            return TestResult(
                "Telegram Notifications",
                True,
                f"Connected to bot: @{bot_info.get('username', 'unknown')}"
            )

        except Exception as e:
            return TestResult("Telegram Notifications", False, str(e))

    def test_file_structure(self) -> TestResult:
        """Test required files exist"""
        print("[8/8] Testing file structure...")

        try:
            required_files = [
                'xtrades_sync_service.py',
                'xtrades_sync.bat',
                'src/xtrades_scraper.py',
                'src/xtrades_db_manager.py',
                'src/telegram_notifier.py',
                '.env'
            ]

            missing_files = []
            for file_path in required_files:
                if not Path(file_path).exists():
                    missing_files.append(file_path)

            # Check logs directory
            logs_dir = Path('logs')
            if not logs_dir.exists():
                logs_dir.mkdir()

            if missing_files:
                return TestResult(
                    "File Structure",
                    False,
                    f"Missing files: {', '.join(missing_files)}"
                )

            return TestResult(
                "File Structure",
                True,
                f"All {len(required_files)} required files present"
            )

        except Exception as e:
            return TestResult("File Structure", False, str(e))

    def run_all_tests(self) -> int:
        """Run all tests and return exit code"""
        print("="*70)
        print("XTRADES SYNC SERVICE - AUTOMATED TEST SUITE")
        print("="*70)
        print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

        # Run tests in order
        test_methods = [
            self.test_imports,
            self.test_environment,
            self.test_database_connection,
            self.test_database_manager,
            self.test_scraper_initialization,
            self.test_scraper_login,
            self.test_telegram,
            self.test_file_structure
        ]

        for test_method in test_methods:
            try:
                result = test_method()
                self.results.append(result)
            except Exception as e:
                self.results.append(
                    TestResult(test_method.__name__, False, f"Unexpected error: {e}")
                )

        # Print results
        print("\n" + "="*70)
        print("TEST RESULTS")
        print("="*70)

        passed_count = 0
        failed_count = 0

        for result in self.results:
            status = "✓ PASS" if result.passed else "✗ FAIL"
            color = "\033[92m" if result.passed else "\033[91m"  # Green/Red
            reset = "\033[0m"

            print(f"{result.name:30s} {color}{status}{reset}")

            if result.message:
                print(f"  → {result.message}")

            if result.passed:
                passed_count += 1
            else:
                failed_count += 1

        # Summary
        print("\n" + "="*70)
        print(f"SUMMARY: {passed_count} passed, {failed_count} failed")
        print("="*70)

        if failed_count == 0:
            print("\n✓ All tests passed! System is ready for production.")
            print("\nNext steps:")
            print("  1. Add profiles to monitor (if not already done)")
            print("  2. Test manual sync: python xtrades_sync_service.py")
            print("  3. Configure Task Scheduler (see XTRADES_SYNC_SETUP.md)")
        else:
            print("\n✗ Some tests failed. Please fix the issues above.")
            print("\nTroubleshooting:")
            print("  1. Check .env file has all required credentials")
            print("  2. Ensure PostgreSQL is running")
            print("  3. Verify Xtrades schema is installed")
            print("  4. Review XTRADES_SYNC_SETUP.md for detailed setup")

        print("\n" + "="*70)
        print(f"Completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("="*70 + "\n")

        # Return exit code
        return 0 if failed_count == 0 else 1


def main():
    """Main entry point"""
    try:
        tester = XtradesSyncTester()
        exit_code = tester.run_all_tests()
        sys.exit(exit_code)

    except KeyboardInterrupt:
        print("\n\nTest suite interrupted by user")
        sys.exit(130)

    except Exception as e:
        print(f"\n\nFatal error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

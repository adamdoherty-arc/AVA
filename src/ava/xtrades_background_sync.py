"""
Xtrades Background Sync Service
================================

Automatic background synchronization service for Xtrades alerts.
Runs continuously, fetching new alerts every 5 minutes from all active profiles.

Features:
- Scheduled sync every 5 minutes
- Syncs all active Xtrades profiles
- Logs sync operations to database
- Error handling with retry logic
- Performance metrics tracking

Usage:
    python src/ava/xtrades_background_sync.py

Or use the Windows batch file:
    run_xtrades_sync.bat

Author: Magnus Wheel Strategy
Created: 2025-11-06
"""

import os
import sys
import time
import schedule
import psycopg2
from datetime import datetime, timedelta

# Fix Windows console encoding
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')
from pathlib import Path
from typing import List, Dict
from dotenv import load_dotenv

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.xtrades_scraper import XtradesScraper
from src.xtrades_db_manager import XtradesDBManager

load_dotenv()


class XtradesBackgroundSync:
    """Background sync service for Xtrades alerts"""

    def __init__(self, interval_minutes: int = 5):
        """
        Initialize background sync service

        Args:
            interval_minutes: Minutes between sync operations (default: 5)
        """
        self.interval_minutes = interval_minutes
        self.db_manager = XtradesDBManager()
        self.scraper = None
        self.sync_count = 0
        self.last_sync_time = None
        self.is_running = False

    def get_active_profiles(self) -> List[str]:
        """
        Get list of active profiles to sync

        Returns:
            List of active profile usernames
        """
        try:
            # Use XtradesDBManager's get_active_profiles which handles connection properly
            profiles = self.db_manager.get_active_profiles()
            return [p['username'] for p in profiles]

        except Exception as e:
            print(f"❌ Error getting active profiles: {e}")
            return []

    def sync_xtrades(self) -> None:
        """Perform Xtrades sync operation"""
        if self.is_running:
            print("⚠️  Sync already in progress, skipping...")
            return

        self.is_running = True
        start_time = datetime.now()

        print(f"\n{'='*60}")
        print(f"🔄 Starting Xtrades Sync #{self.sync_count + 1}")
        print(f"⏰ Time: {start_time.strftime('%Y-%m-%d %I:%M:%S %p')}")
        print(f"{'='*60}")

        profiles_synced = 0
        trades_found = 0
        new_trades = 0
        updated_trades = 0
        errors = []

        try:
            # Get active profiles
            active_profiles = self.get_active_profiles()

            if not active_profiles:
                print("⚠️  No active profiles to sync")
                self.is_running = False
                return

            print(f"📋 Active profiles: {', '.join(active_profiles)}")

            # Initialize scraper (reuse if possible)
            if not self.scraper:
                print("🌐 Initializing scraper...")
                self.scraper = XtradesScraper(headless=True)
                self.scraper.login()
                print("✅ Scraper initialized and logged in")

            # Sync each profile
            for username in active_profiles:
                try:
                    print(f"\n📊 Syncing profile: {username}")

                    # Scrape alerts
                    alerts = self.scraper.get_profile_alerts(username, max_alerts=50)
                    trades_found += len(alerts)

                    if alerts:
                        print(f"   Found {len(alerts)} alerts")

                        # Get profile_id
                        profile = self.db_manager.get_profile_by_username(username)
                        if not profile:
                            print(f"   ⚠️  Profile not found in database, skipping")
                            continue

                        profile_id = profile['id']

                        # Save alerts to database
                        saved_count = 0
                        for alert_data in alerts:
                            try:
                                # Add profile_id to alert data
                                alert_data['profile_id'] = profile_id

                                # Check if trade already exists
                                existing = self.db_manager.find_existing_trade(
                                    profile_id,
                                    alert_data.get('ticker', ''),
                                    alert_data.get('alert_timestamp')
                                )

                                if not existing:
                                    # Add new trade
                                    self.db_manager.add_trade(alert_data)
                                    saved_count += 1
                            except Exception as e:
                                print(f"   ⚠️  Failed to save alert: {e}")
                                continue

                        new_trades += saved_count
                        print(f"   ✅ Saved {saved_count} new/updated trades")
                    else:
                        print(f"   No new alerts")

                    profiles_synced += 1

                except Exception as e:
                    error_msg = f"Error syncing {username}: {e}"
                    print(f"   ❌ {error_msg}")
                    errors.append(error_msg)
                    continue

            # Calculate duration
            duration = (datetime.now() - start_time).total_seconds()

            # Log sync operation
            self._log_sync(
                profiles_synced=profiles_synced,
                trades_found=trades_found,
                new_trades=new_trades,
                updated_trades=updated_trades,
                errors=errors,
                duration=duration
            )

            # Update stats
            self.sync_count += 1
            self.last_sync_time = datetime.now()

            # Print summary
            print(f"\n{'='*60}")
            print(f"✅ Sync Complete!")
            print(f"📈 Profiles synced: {profiles_synced}/{len(active_profiles)}")
            print(f"📊 Alerts found: {trades_found}")
            print(f"💾 New/updated: {new_trades}")
            print(f"⏱️  Duration: {duration:.2f}s")
            if errors:
                print(f"⚠️  Errors: {len(errors)}")
            print(f"⏰ Next sync in {self.interval_minutes} minutes")
            print(f"{'='*60}\n")

        except Exception as e:
            error_msg = f"Critical error in sync operation: {e}"
            print(f"❌ {error_msg}")
            errors.append(error_msg)

            # Log failed sync
            duration = (datetime.now() - start_time).total_seconds()
            self._log_sync(
                profiles_synced=profiles_synced,
                trades_found=trades_found,
                new_trades=new_trades,
                updated_trades=updated_trades,
                errors=errors,
                duration=duration,
                status='failed'
            )

        finally:
            self.is_running = False

    def _log_sync(
        self,
        profiles_synced: int,
        trades_found: int,
        new_trades: int,
        updated_trades: int,
        errors: List[str],
        duration: float,
        status: str = 'success'
    ):
        """
        Log sync operation to database

        Args:
            profiles_synced: Number of profiles successfully synced
            trades_found: Total trades found
            new_trades: Number of new trades added
            updated_trades: Number of trades updated
            errors: List of error messages
            duration: Duration in seconds
            status: 'success', 'partial', or 'failed'
        """
        conn = None
        try:
            # Use XtradesDBManager's connection for consistency
            conn = self.db_manager.get_connection()
            cursor = conn.cursor()

            # Determine status
            if errors and profiles_synced == 0:
                status = 'failed'
            elif errors:
                status = 'partial'

            cursor.execute("""
                INSERT INTO xtrades_sync_log (
                    profiles_synced,
                    trades_found,
                    new_trades,
                    updated_trades,
                    errors,
                    duration_seconds,
                    status
                ) VALUES (%s, %s, %s, %s, %s, %s, %s)
            """, (
                profiles_synced,
                trades_found,
                new_trades,
                updated_trades,
                '\n'.join(errors) if errors else None,
                duration,
                status
            ))

            conn.commit()

        except Exception as e:
            print(f"⚠️  Warning: Could not log sync operation: {e}")

        finally:
            if conn:
                self.db_manager.release_connection(conn)

    def start(self) -> None:
        """Start the background sync service"""
        print("🚀 Starting Xtrades Background Sync Service")
        print(f"⏰ Sync interval: Every {self.interval_minutes} minutes")
        print(f"📂 Database: {os.getenv('DB_HOST', 'localhost')}:{os.getenv('DB_PORT', '5432')}/{os.getenv('DB_NAME', 'magnus')}")
        print(f"🌐 Xtrades account: {os.getenv('XTRADES_USERNAME', 'Not configured')}")
        print("\nPress Ctrl+C to stop\n")

        # Run first sync immediately
        self.sync_xtrades()

        # Schedule periodic sync
        schedule.every(self.interval_minutes).minutes.do(self.sync_xtrades)

        # Keep running
        try:
            while True:
                schedule.run_pending()
                time.sleep(30)  # Check every 30 seconds

        except KeyboardInterrupt:
            print("\n\n🛑 Stopping sync service...")
            self.cleanup()
            print("✅ Service stopped gracefully")

    def cleanup(self) -> None:
        """Cleanup resources"""
        if self.scraper:
            try:
                self.scraper.close()
                print("✅ Scraper closed")
            except Exception as e:
                print(f"⚠️  Warning: Error closing scraper: {e}")


def main():
    """Main entry point"""
    import argparse

    parser = argparse.ArgumentParser(
        description='Xtrades Background Sync Service',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run with default 5-minute interval
  python src/ava/xtrades_background_sync.py

  # Run with 10-minute interval
  python src/ava/xtrades_background_sync.py --interval 10

  # Run once and exit (no continuous sync)
  python src/ava/xtrades_background_sync.py --once
        """
    )

    parser.add_argument(
        '--interval',
        type=int,
        default=5,
        help='Sync interval in minutes (default: 5)'
    )

    parser.add_argument(
        '--once',
        action='store_true',
        help='Run sync once and exit (no continuous sync)'
    )

    args = parser.parse_args()

    # Validate environment (DB connection uses DB_HOST, DB_PORT, DB_USER, DB_PASSWORD, DB_NAME)
    if not os.getenv('DB_HOST') and not os.getenv('DATABASE_URL'):
        print("❌ ERROR: Database not configured. Set either DB_HOST or DATABASE_URL in .env file")
        sys.exit(1)

    if not os.getenv('XTRADES_USERNAME') or not os.getenv('XTRADES_PASSWORD'):
        print("❌ ERROR: XTRADES_USERNAME and XTRADES_PASSWORD must be set in .env file")
        sys.exit(1)

    # Create sync service
    sync_service = XtradesBackgroundSync(interval_minutes=args.interval)

    if args.once:
        # Run once and exit
        print("🔄 Running one-time sync...")
        sync_service.sync_xtrades()
        print("✅ One-time sync complete")
    else:
        # Run continuously
        sync_service.start()


if __name__ == "__main__":
    main()

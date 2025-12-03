"""
Xtrades Database Synchronization Service
=========================================
Manages synchronization between Xtrades.net scraper and PostgreSQL database.

Features:
- Profile management
- Trade storage and updates
- Duplicate detection
- Sync logging
- Batch operations
- Performance tracking
"""

import os
import sys
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from datetime import datetime, timedelta
from decimal import Decimal

import psycopg2
from psycopg2.extras import execute_values, RealDictCursor
from dotenv import load_dotenv

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from xtrades_scraper import (
    XtradesScraper,
    LoginFailedException,
    ProfileNotFoundException
)

load_dotenv()


class XtradesDBSync:
    """
    Synchronizes Xtrades.net scraper data with PostgreSQL database.

    Usage:
        sync = XtradesDBSync()
        sync.sync_profile("behappy")
        sync.sync_all_active_profiles()
    """

    def __init__(self, db_config: Optional[Dict] = None):
        """
        Initialize database sync service.

        Args:
            db_config: Database configuration dict (uses env vars if None)
        """
        self.db_config = db_config or {
            'host': os.getenv('DB_HOST', 'localhost'),
            'port': int(os.getenv('DB_PORT', '5432')),
            'database': os.getenv('DB_NAME', 'magnus'),
            'user': os.getenv('DB_USER', 'postgres'),
            'password': os.getenv('DB_PASSWORD')
        }

        self.conn = None
        self.scraper: Optional[XtradesScraper] = None
        self._connect()

    def _connect(self) -> None:
        """Establish database connection"""
        try:
            self.conn = psycopg2.connect(**self.db_config)
            print("✓ Connected to database")
        except psycopg2.Error as e:
            raise Exception(f"Database connection failed: {e}")

    def _ensure_scraper(self) -> None:
        """Initialize scraper if not already done"""
        if not self.scraper:
            self.scraper = XtradesScraper()
            self.scraper.login()

    def add_profile(
        self,
        username: str,
        display_name: Optional[str] = None,
        active: bool = True,
        notes: Optional[str] = None
    ) -> int:
        """
        Add a profile to monitor.

        Args:
            username: Xtrades.net username
            display_name: Display name (defaults to username)
            active: Whether to actively monitor
            notes: Optional notes

        Returns:
            Profile ID

        Raises:
            psycopg2.Error: Database error
        """
        cursor = self.conn.cursor()

        try:
            cursor.execute("""
                INSERT INTO xtrades_profiles (username, display_name, active, notes)
                VALUES (%s, %s, %s, %s)
                ON CONFLICT (username) DO UPDATE
                    SET display_name = EXCLUDED.display_name,
                        active = EXCLUDED.active,
                        notes = EXCLUDED.notes
                RETURNING id
            """, (username, display_name or username, active, notes))

            profile_id = cursor.fetchone()[0]
            self.conn.commit()

            print(f"✓ Profile added: {username} (ID: {profile_id})")
            return profile_id

        except psycopg2.Error as e:
            self.conn.rollback()
            raise Exception(f"Failed to add profile: {e}")
        finally:
            cursor.close()

    def get_profile_id(self, username: str) -> Optional[int]:
        """Get profile ID by username"""
        cursor = self.conn.cursor()
        try:
            cursor.execute(
                "SELECT id FROM xtrades_profiles WHERE username = %s",
                (username,)
            )
            result = cursor.fetchone()
            return result[0] if result else None
        finally:
            cursor.close()

    def sync_profile(
        self,
        username: str,
        max_alerts: Optional[int] = None
    ) -> Tuple[int, int]:
        """
        Sync a single profile.

        Args:
            username: Xtrades.net username
            max_alerts: Maximum alerts to retrieve

        Returns:
            Tuple of (new_trades, updated_trades)

        Raises:
            ProfileNotFoundException: Profile doesn't exist
            Exception: Other errors
        """
        start_time = datetime.now()
        print(f"\nSyncing profile: {username}")
        print("="*60)

        try:
            # Ensure profile exists in database
            profile_id = self.get_profile_id(username)
            if not profile_id:
                profile_id = self.add_profile(username)

            # Initialize scraper
            self._ensure_scraper()

            # Scrape profile
            print("Scraping alerts...")
            alerts = self.scraper.get_profile_alerts(username, max_alerts)
            print(f"Found {len(alerts)} alerts")

            # Store alerts
            print("Storing in database...")
            new_trades, updated_trades = self._store_trades(profile_id, alerts)

            # Update profile status
            self._update_profile_status(
                profile_id,
                'success',
                len(alerts)
            )

            # Log sync
            duration = (datetime.now() - start_time).total_seconds()
            self._log_sync(
                profiles_synced=1,
                trades_found=len(alerts),
                new_trades=new_trades,
                updated_trades=updated_trades,
                duration=duration,
                status='success'
            )

            print(f"✓ Sync complete: {new_trades} new, {updated_trades} updated")
            return new_trades, updated_trades

        except ProfileNotFoundException:
            self._update_profile_status(profile_id, 'error')
            raise
        except Exception as e:
            if profile_id:
                self._update_profile_status(profile_id, 'error')
            raise Exception(f"Sync failed: {e}")

    def sync_all_active_profiles(
        self,
        max_alerts: Optional[int] = None
    ) -> Dict[str, Tuple[int, int]]:
        """
        Sync all active profiles.

        Args:
            max_alerts: Maximum alerts per profile

        Returns:
            Dict mapping username to (new_trades, updated_trades)
        """
        start_time = datetime.now()
        print("\nSyncing all active profiles...")
        print("="*60)

        # Get active profiles
        cursor = self.conn.cursor(cursor_factory=RealDictCursor)
        cursor.execute("""
            SELECT username FROM xtrades_profiles
            WHERE active = TRUE
            ORDER BY username
        """)
        profiles = cursor.fetchall()
        cursor.close()

        print(f"Found {len(profiles)} active profiles\n")

        # Initialize scraper once
        self._ensure_scraper()

        # Sync each profile
        results = {}
        total_new = 0
        total_updated = 0
        total_found = 0

        for i, profile in enumerate(profiles, 1):
            username = profile['username']
            print(f"\n[{i}/{len(profiles)}] {username}")
            print("-"*60)

            try:
                new, updated = self.sync_profile(username, max_alerts)
                results[username] = (new, updated)
                total_new += new
                total_updated += updated
            except Exception as e:
                print(f"✗ Error: {e}")
                results[username] = (0, 0)

        # Overall sync log
        duration = (datetime.now() - start_time).total_seconds()
        self._log_sync(
            profiles_synced=len(profiles),
            trades_found=total_found,
            new_trades=total_new,
            updated_trades=total_updated,
            duration=duration,
            status='success'
        )

        # Summary
        print("\n" + "="*60)
        print("SYNC SUMMARY")
        print("="*60)
        print(f"Profiles synced:  {len(profiles)}")
        print(f"New trades:       {total_new}")
        print(f"Updated trades:   {total_updated}")
        print(f"Duration:         {duration:.1f}s")

        return results

    def _store_trades(
        self,
        profile_id: int,
        alerts: List[Dict]
    ) -> Tuple[int, int]:
        """
        Store trades in database.

        Args:
            profile_id: Profile ID
            alerts: List of alert dictionaries

        Returns:
            Tuple of (new_count, updated_count)
        """
        cursor = self.conn.cursor()
        new_count = 0
        updated_count = 0

        try:
            for alert in alerts:
                # Generate unique ID from alert text and timestamp
                alert_id = self._generate_alert_id(alert)

                # Check if trade exists
                cursor.execute("""
                    SELECT id FROM xtrades_trades
                    WHERE profile_id = %s AND xtrades_alert_id = %s
                """, (profile_id, alert_id))

                exists = cursor.fetchone()

                if exists:
                    # Update existing trade
                    cursor.execute("""
                        UPDATE xtrades_trades
                        SET exit_price = COALESCE(%s, exit_price),
                            exit_date = COALESCE(%s, exit_date),
                            pnl = COALESCE(%s, pnl),
                            pnl_percent = COALESCE(%s, pnl_percent),
                            status = %s,
                            updated_at = NOW()
                        WHERE id = %s
                    """, (
                        alert.get('exit_price'),
                        alert.get('exit_date'),
                        alert.get('pnl'),
                        alert.get('pnl_percent'),
                        alert['status'],
                        exists[0]
                    ))
                    updated_count += 1
                else:
                    # Insert new trade
                    cursor.execute("""
                        INSERT INTO xtrades_trades (
                            profile_id, ticker, strategy, action,
                            entry_price, entry_date, exit_price, exit_date,
                            quantity, pnl, pnl_percent, status,
                            strike_price, expiration_date,
                            alert_text, alert_timestamp, xtrades_alert_id
                        ) VALUES (
                            %s, %s, %s, %s, %s, %s, %s, %s,
                            %s, %s, %s, %s, %s, %s, %s, %s, %s
                        )
                    """, (
                        profile_id,
                        alert.get('ticker'),
                        alert.get('strategy'),
                        alert.get('action'),
                        alert.get('entry_price'),
                        alert.get('entry_date'),
                        alert.get('exit_price'),
                        alert.get('exit_date'),
                        alert.get('quantity'),
                        alert.get('pnl'),
                        alert.get('pnl_percent'),
                        alert['status'],
                        alert.get('strike_price'),
                        alert.get('expiration_date'),
                        alert['alert_text'],
                        alert.get('alert_timestamp'),
                        alert_id
                    ))
                    new_count += 1

            self.conn.commit()
            return new_count, updated_count

        except psycopg2.Error as e:
            self.conn.rollback()
            raise Exception(f"Failed to store trades: {e}")
        finally:
            cursor.close()

    def _generate_alert_id(self, alert: Dict) -> str:
        """Generate unique ID for alert"""
        import hashlib

        # Use alert text and timestamp
        text = alert.get('alert_text', '')
        timestamp = alert.get('alert_timestamp', '')

        content = f"{text}_{timestamp}"
        return hashlib.md5(content.encode()).hexdigest()

    def _update_profile_status(
        self,
        profile_id: int,
        status: str,
        trades_count: int = 0
    ):
        """Update profile sync status"""
        cursor = self.conn.cursor()
        try:
            cursor.execute("""
                UPDATE xtrades_profiles
                SET last_sync = NOW(),
                    last_sync_status = %s,
                    total_trades_scraped = total_trades_scraped + %s
                WHERE id = %s
            """, (status, trades_count, profile_id))
            self.conn.commit()
        finally:
            cursor.close()

    def _log_sync(
        self,
        profiles_synced: int,
        trades_found: int,
        new_trades: int,
        updated_trades: int,
        duration: float,
        status: str,
        errors: Optional[str] = None
    ):
        """Log sync operation"""
        cursor = self.conn.cursor()
        try:
            cursor.execute("""
                INSERT INTO xtrades_sync_log (
                    profiles_synced, trades_found, new_trades,
                    updated_trades, duration_seconds, status, errors
                ) VALUES (%s, %s, %s, %s, %s, %s, %s)
            """, (
                profiles_synced, trades_found, new_trades,
                updated_trades, duration, status, errors
            ))
            self.conn.commit()
        except psycopg2.Error as e:
            print(f"Warning: Failed to log sync: {e}")
        finally:
            cursor.close()

    def get_profile_stats(self, username: str) -> Dict:
        """Get statistics for a profile"""
        cursor = self.conn.cursor(cursor_factory=RealDictCursor)

        try:
            cursor.execute("""
                SELECT
                    p.username,
                    p.display_name,
                    p.active,
                    p.total_trades_scraped,
                    p.last_sync,
                    COUNT(t.id) as total_trades,
                    COUNT(CASE WHEN t.status = 'open' THEN 1 END) as open_trades,
                    COUNT(CASE WHEN t.status = 'closed' THEN 1 END) as closed_trades,
                    SUM(CASE WHEN t.status = 'closed' THEN t.pnl ELSE 0 END) as total_pnl,
                    AVG(CASE WHEN t.status = 'closed' AND t.pnl IS NOT NULL
                        THEN t.pnl_percent ELSE NULL END) as avg_pnl_percent
                FROM xtrades_profiles p
                LEFT JOIN xtrades_trades t ON p.id = t.profile_id
                WHERE p.username = %s
                GROUP BY p.id
            """, (username,))

            return dict(cursor.fetchone() or {})

        finally:
            cursor.close()

    def get_recent_trades(
        self,
        username: Optional[str] = None,
        limit: int = 20,
        status: Optional[str] = None
    ) -> List[Dict]:
        """Get recent trades"""
        cursor = self.conn.cursor(cursor_factory=RealDictCursor)

        try:
            query = """
                SELECT
                    t.*,
                    p.username,
                    p.display_name
                FROM xtrades_trades t
                JOIN xtrades_profiles p ON t.profile_id = p.id
                WHERE 1=1
            """
            params = []

            if username:
                query += " AND p.username = %s"
                params.append(username)

            if status:
                query += " AND t.status = %s"
                params.append(status)

            query += " ORDER BY t.alert_timestamp DESC LIMIT %s"
            params.append(limit)

            cursor.execute(query, params)
            return [dict(row) for row in cursor.fetchall()]

        finally:
            cursor.close()

    def close(self) -> None:
        """Close connections"""
        if self.conn:
            self.conn.close()
            print("✓ Database connection closed")

        if self.scraper:
            self.scraper.close()


def main():
    """Example usage"""
    import argparse

    parser = argparse.ArgumentParser(description='Xtrades Database Sync')
    parser.add_argument('command', choices=['add', 'sync', 'sync-all', 'stats'])
    parser.add_argument('--username', help='Username')
    parser.add_argument('--max-alerts', type=int, help='Max alerts to retrieve')
    args = parser.parse_args()

    sync = XtradesDBSync()

    try:
        if args.command == 'add':
            if not args.username:
                print("Error: --username required")
                return 1
            sync.add_profile(args.username)

        elif args.command == 'sync':
            if not args.username:
                print("Error: --username required")
                return 1
            sync.sync_profile(args.username, args.max_alerts)

        elif args.command == 'sync-all':
            sync.sync_all_active_profiles(args.max_alerts)

        elif args.command == 'stats':
            if not args.username:
                print("Error: --username required")
                return 1
            stats = sync.get_profile_stats(args.username)
            print(f"\nStats for {args.username}:")
            for key, value in stats.items():
                print(f"  {key}: {value}")

        return 0

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    finally:
        sync.close()


if __name__ == "__main__":
    sys.exit(main())

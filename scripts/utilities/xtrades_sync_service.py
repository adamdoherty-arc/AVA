#!/usr/bin/env python3
"""
Xtrades Sync Service - Automated 5-minute sync of trade alerts from Xtrades.net profiles
========================================================================================

This service runs every 5 minutes (via Windows Task Scheduler) to:
1. Load active profiles from the database
2. Scrape latest alerts from each profile
3. Compare with existing database trades
4. Add new trades and update existing ones
5. Send Telegram notifications for new/updated trades
6. Log all sync operations to database and file

Features:
- Continuous browser session (login once per run)
- Duplicate detection via timestamps
- Graceful error handling (continue on profile errors)
- Comprehensive logging to database and file
- Telegram notifications with retry logic
- Profile-level sync tracking
- Performance monitoring

Designed for Windows Task Scheduler execution every 5 minutes.

Author: Magnus Wheel Strategy Trading Dashboard
Created: 2025-11-02
"""

import os
import sys
import time
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
from dotenv import load_dotenv

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src'))

from xtrades_scraper import XtradesScraper, LoginFailedException, ProfileNotFoundException
from xtrades_db_manager import XtradesDBManager
from telegram_notifier import TelegramNotifier


# Configure logging
def setup_logging(log_dir: str = 'logs') -> logging.Logger:
    """
    Setup logging to both file and console.

    Args:
        log_dir: Directory for log files

    Returns:
        Configured logger instance
    """
    # Create logs directory if it doesn't exist
    log_path = Path(log_dir)
    log_path.mkdir(exist_ok=True)

    # Create logger
    logger = logging.getLogger('xtrades_sync')
    logger.setLevel(logging.INFO)

    # Remove existing handlers to avoid duplicates
    logger.handlers.clear()

    # File handler - rotating daily
    log_file = log_path / f"xtrades_sync_{datetime.now().strftime('%Y%m%d')}.log"
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setLevel(logging.INFO)
    file_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    file_handler.setFormatter(file_formatter)

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%H:%M:%S'
    )
    console_handler.setFormatter(console_formatter)

    # Add handlers
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger


# Initialize logger
logger = setup_logging()


class XtradesSyncService:
    """
    Service for synchronizing Xtrades profile data to database.

    Handles the complete sync workflow including scraping, database updates,
    notifications, and error handling.
    """

    def __init__(self, headless: bool = True):
        """
        Initialize the sync service.

        Args:
            headless: Run browser in headless mode (default True for scheduled tasks)
        """
        self.headless = headless
        self.db = XtradesDBManager()
        self.scraper: Optional[XtradesScraper] = None
        self.notifier = TelegramNotifier()

        # Statistics
        self.stats = {
            'profiles_synced': 0,
            'trades_found': 0,
            'new_trades': 0,
            'updated_trades': 0,
            'errors': [],
            'start_time': None,
            'end_time': None
        }

    def sync_profile(self, profile: Dict[str, Any]) -> Dict[str, Any]:
        """
        Sync a single profile's trades.

        Args:
            profile: Profile dictionary from database

        Returns:
            Dictionary with sync statistics for this profile
        """
        profile_id = profile['id']
        username = profile['username']

        logger.info(f"Syncing profile: {username} (ID: {profile_id})")

        stats = {
            'username': username,
            'trades_found': 0,
            'new_trades': 0,
            'updated_trades': 0,
            'errors': []
        }

        try:
            # Scrape alerts from profile
            alerts = self.scraper.get_profile_alerts(username, max_alerts=50)
            stats['trades_found'] = len(alerts)

            logger.info(f"  Found {len(alerts)} alerts for {username}")

            if not alerts:
                self.db.update_profile_sync_status(profile_id, 'success', 0)
                return stats

            # Process each alert
            for alert in alerts:
                try:
                    result = self._process_alert(profile_id, username, alert)

                    if result == 'new':
                        stats['new_trades'] += 1
                    elif result == 'updated':
                        stats['updated_trades'] += 1

                except Exception as e:
                    error_msg = f"Error processing alert for {username}: {e}"
                    logger.error(f"  {error_msg}")
                    stats['errors'].append(error_msg)

            # Update profile sync status
            self.db.update_profile_sync_status(
                profile_id,
                'success',
                stats['new_trades']
            )

            logger.info(
                f"  Completed {username}: "
                f"{stats['new_trades']} new, {stats['updated_trades']} updated"
            )

        except ProfileNotFoundException as e:
            error_msg = f"Profile not found: {username}"
            logger.error(f"  {error_msg}")
            stats['errors'].append(error_msg)
            self.db.update_profile_sync_status(profile_id, 'error')

        except Exception as e:
            error_msg = f"Error syncing {username}: {e}"
            logger.error(f"  {error_msg}")
            stats['errors'].append(error_msg)
            self.db.update_profile_sync_status(profile_id, 'error')

        return stats

    def _process_alert(
        self,
        profile_id: int,
        username: str,
        alert: Dict[str, Any]
    ) -> Optional[str]:
        """
        Process a single alert - add new or update existing trade.

        Args:
            profile_id: Profile database ID
            username: Profile username
            alert: Parsed alert dictionary from scraper

        Returns:
            'new' if new trade added, 'updated' if existing updated, None if skipped
        """
        ticker = alert.get('ticker')
        alert_timestamp = alert.get('alert_timestamp')

        if not ticker or not alert_timestamp:
            logger.debug(f"    Skipping alert: missing ticker or timestamp")
            return None

        # Parse timestamp
        if isinstance(alert_timestamp, str):
            try:
                alert_timestamp = datetime.fromisoformat(alert_timestamp)
            except ValueError:
                logger.warning(f"    Invalid timestamp format: {alert_timestamp}")
                return None

        # Check for duplicate
        existing_trade_id = self.db.find_existing_trade(
            profile_id,
            ticker,
            alert_timestamp
        )

        if existing_trade_id:
            # Check if we need to update (e.g., trade was closed)
            if alert.get('status') == 'closed' or alert.get('exit_price'):
                update_data = {}

                if alert.get('exit_price'):
                    update_data['exit_price'] = alert['exit_price']

                if alert.get('exit_date'):
                    update_data['exit_date'] = alert['exit_date']

                if alert.get('pnl') is not None:
                    update_data['pnl'] = alert['pnl']

                if alert.get('pnl_percent') is not None:
                    update_data['pnl_percent'] = alert['pnl_percent']

                if alert.get('status'):
                    update_data['status'] = alert['status']

                if update_data:
                    success = self.db.update_trade(existing_trade_id, update_data)
                    if success:
                        logger.info(
                            f"    Updated {ticker} - "
                            f"{alert.get('strategy', 'N/A')} (ID: {existing_trade_id})"
                        )

                        # Send notification for closed trade
                        if update_data.get('status') == 'closed':
                            trade_data = self.db.get_trade_by_id(existing_trade_id)
                            if trade_data:
                                trade_data['profile_username'] = username
                                self.notifier.send_trade_closed_alert(trade_data)
                                self.db.log_notification(
                                    existing_trade_id,
                                    'trade_closed',
                                    status='sent'
                                )

                        return 'updated'

            logger.debug(f"    Duplicate found for {ticker}, skipping")
            return None

        # Add new trade
        trade_data = {
            'profile_id': profile_id,
            'ticker': ticker,
            'strategy': alert.get('strategy'),
            'action': alert.get('action'),
            'entry_price': alert.get('entry_price'),
            'entry_date': alert.get('entry_date'),
            'quantity': alert.get('quantity', 1),
            'strike_price': alert.get('strike_price'),
            'expiration_date': alert.get('expiration_date'),
            'alert_text': alert.get('alert_text'),
            'alert_timestamp': alert_timestamp,
            'status': alert.get('status', 'open')
        }

        # Add exit data if present
        if alert.get('exit_price'):
            trade_data['exit_price'] = alert['exit_price']
        if alert.get('exit_date'):
            trade_data['exit_date'] = alert['exit_date']
        if alert.get('pnl') is not None:
            trade_data['pnl'] = alert['pnl']
        if alert.get('pnl_percent') is not None:
            trade_data['pnl_percent'] = alert['pnl_percent']

        try:
            trade_id = self.db.add_trade(trade_data)
            logger.info(
                f"    Added {ticker} - {alert.get('strategy', 'N/A')} "
                f"@ ${alert.get('entry_price', 'N/A')} (ID: {trade_id})"
            )

            # Send notification for new trade
            trade_data['profile_username'] = username
            message_id = self.notifier.send_new_trade_alert(trade_data)

            if message_id:
                self.db.log_notification(
                    trade_id,
                    'new_trade',
                    telegram_msg_id=message_id,
                    status='sent'
                )

            return 'new'

        except Exception as e:
            logger.error(f"    Failed to add trade: {e}")
            return None

    def run_sync(self) -> int:
        """
        Execute the complete sync workflow.

        Returns:
            Exit code (0 for success, 1 for errors)
        """
        self.stats['start_time'] = datetime.now()

        logger.info("="*70)
        logger.info(f"XTRADES SYNC - {self.stats['start_time'].strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info("="*70)

        # Initialize database connection
        conn = self.db.get_connection()
        conn.close()

        # Get active profiles
        profiles = self.db.get_active_profiles()

        if not profiles:
            logger.info("No active profiles to sync")
            return 0

        logger.info(f"Found {len(profiles)} active profile(s) to sync\n")

        # Start sync log
        sync_id = self.db.log_sync_start()
        logger.info(f"Sync log ID: {sync_id}\n")

        try:
            # Initialize scraper
            logger.info("Initializing browser...")
            self.scraper = XtradesScraper(headless=self.headless)

            # Login once for all profiles
            logger.info("Logging in to Xtrades.net...")
            if not self.scraper.login():
                error_msg = "Failed to login to Xtrades.net"
                logger.error(error_msg)
                self.stats['errors'].append(error_msg)

                # Log failed sync
                self.db.log_sync_complete(sync_id, {
                    'profiles_synced': 0,
                    'trades_found': 0,
                    'new_trades': 0,
                    'updated_trades': 0,
                    'errors': error_msg,
                    'duration_seconds': 0,
                    'status': 'failed'
                })

                return 1

            logger.info("Login successful!\n")

            # Sync each profile
            for i, profile in enumerate(profiles, 1):
                logger.info(f"[{i}/{len(profiles)}] Processing {profile['username']}...")

                try:
                    profile_stats = self.sync_profile(profile)

                    # Aggregate stats
                    self.stats['profiles_synced'] += 1
                    self.stats['trades_found'] += profile_stats['trades_found']
                    self.stats['new_trades'] += profile_stats['new_trades']
                    self.stats['updated_trades'] += profile_stats['updated_trades']
                    self.stats['errors'].extend(profile_stats['errors'])

                except Exception as e:
                    error_msg = f"Error syncing {profile['username']}: {e}"
                    logger.error(error_msg)
                    self.stats['errors'].append(error_msg)

                logger.info("")  # Blank line between profiles

        except Exception as e:
            error_msg = f"Critical error during sync: {e}"
            logger.error(error_msg)
            self.stats['errors'].append(error_msg)

            # Send error notification
            self.notifier.send_sync_error_alert(
                error_msg,
                [p['username'] for p in profiles]
            )

        finally:
            # Cleanup
            if self.scraper:
                self.scraper.close()

            # Calculate duration
            self.stats['end_time'] = datetime.now()
            duration = (self.stats['end_time'] - self.stats['start_time']).total_seconds()

            # Determine final status
            if self.stats['errors']:
                if self.stats['profiles_synced'] > 0:
                    status = 'partial'
                else:
                    status = 'failed'
            else:
                status = 'success'

            # Log sync completion
            self.db.log_sync_complete(sync_id, {
                'profiles_synced': self.stats['profiles_synced'],
                'trades_found': self.stats['trades_found'],
                'new_trades': self.stats['new_trades'],
                'updated_trades': self.stats['updated_trades'],
                'errors': '\n'.join(self.stats['errors']) if self.stats['errors'] else None,
                'duration_seconds': duration,
                'status': status
            })

            # Print summary
            logger.info("="*70)
            logger.info("SYNC SUMMARY")
            logger.info("="*70)
            logger.info(f"Profiles Synced:   {self.stats['profiles_synced']}/{len(profiles)}")
            logger.info(f"Total Alerts:      {self.stats['trades_found']}")
            logger.info(f"New Trades:        {self.stats['new_trades']}")
            logger.info(f"Updated Trades:    {self.stats['updated_trades']}")
            logger.info(f"Errors:            {len(self.stats['errors'])}")
            logger.info(f"Duration:          {duration:.1f}s")
            logger.info(f"Status:            {status.upper()}")
            logger.info("="*70)

            if self.stats['errors']:
                logger.warning("\nErrors encountered:")
                for error in self.stats['errors']:
                    logger.warning(f"  - {error}")

            logger.info("")

        # Return exit code
        return 0 if status in ['success', 'partial'] else 1


def main():
    """Main entry point for the sync service."""
    # Load environment variables
    load_dotenv()

    # Parse command line arguments
    headless = '--no-headless' not in sys.argv

    try:
        # Create and run sync service
        service = XtradesSyncService(headless=headless)
        exit_code = service.run_sync()

        sys.exit(exit_code)

    except KeyboardInterrupt:
        logger.info("\n\nSync interrupted by user")
        sys.exit(130)

    except Exception as e:
        logger.exception(f"Fatal error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()

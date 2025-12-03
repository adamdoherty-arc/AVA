#!/usr/bin/env python3
"""
Xtrades Telegram Integration

This script integrates Xtrades trade tracking with Telegram notifications.
It monitors the database for new trades, updates, and closures, then sends
real-time notifications via Telegram.

Usage:
    python xtrades_telegram_integration.py

    Or run as a background service:
    python xtrades_telegram_integration.py --daemon

Author: Magnus Trading Dashboard
Created: 2025-11-02
"""

import os
import sys
import argparse
import logging
import time
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
from decimal import Decimal

import psycopg2
from psycopg2.extras import RealDictCursor
from dotenv import load_dotenv

from telegram_notifier import TelegramNotifier


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('xtrades_telegram.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


class XtradesTelegramIntegration:
    """
    Manages Telegram notifications for Xtrades trading alerts.

    This class monitors the Xtrades database for new trades, updates, and
    closures, then sends real-time notifications via Telegram.
    """

    def __init__(self, check_interval: int = 60):
        """
        Initialize the integration.

        Args:
            check_interval: Seconds between database checks (default: 60)
        """
        load_dotenv()

        self.notifier = TelegramNotifier()
        self.db_url = os.getenv('DATABASE_URL')
        self.check_interval = check_interval
        self.running = False

        # Validate configuration
        if not self.db_url:
            raise ValueError("DATABASE_URL not set in environment")

        logger.info("Xtrades Telegram Integration initialized")

    def start(self) -> None:
        """Start the integration service."""

        logger.info("Starting Xtrades Telegram Integration...")

        # Test Telegram connection
        if self.notifier.enabled:
            if self.notifier.test_connection():
                logger.info("Telegram connection successful")
            else:
                logger.warning("Telegram connection failed - notifications disabled")
        else:
            logger.warning("Telegram notifications are disabled")

        # Test database connection
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT 1")
                logger.info("Database connection successful")
        except Exception as e:
            logger.error(f"Database connection failed: {e}")
            raise

        # Start monitoring loop
        self.running = True
        self._monitoring_loop()

    def stop(self) -> None:
        """Stop the integration service."""
        logger.info("Stopping integration...")
        self.running = False

    def _monitoring_loop(self) -> None:
        """Main monitoring loop."""

        logger.info(f"Monitoring started (check interval: {self.check_interval}s)")

        while self.running:
            try:
                # Process new trades
                new_count = self.process_new_trades()

                # Process trade updates
                update_count = self.process_trade_updates()

                # Process closed trades
                closed_count = self.process_closed_trades()

                # Log activity
                if new_count > 0 or update_count > 0 or closed_count > 0:
                    logger.info(
                        f"Processed: {new_count} new, "
                        f"{update_count} updates, {closed_count} closed"
                    )

                # Wait before next check
                time.sleep(self.check_interval)

            except KeyboardInterrupt:
                logger.info("Received shutdown signal")
                self.stop()
                break

            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}", exc_info=True)
                self.notifier.send_sync_error_alert(
                    f"Monitoring loop error: {str(e)}"
                )
                # Wait before retrying
                time.sleep(self.check_interval)

    def process_new_trades(self) -> int:
        """
        Process and notify about new trades.

        Returns:
            Number of new trades processed
        """

        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()

                # Find trades without new_trade notifications
                cursor.execute("""
                    SELECT
                        t.id, t.ticker, t.strategy, t.action,
                        t.entry_price, t.quantity, t.strike_price,
                        t.expiration_date, t.alert_timestamp, t.alert_text,
                        p.username as profile_username
                    FROM xtrades_trades t
                    JOIN xtrades_profiles p ON t.profile_id = p.id
                    LEFT JOIN xtrades_notifications n
                        ON t.id = n.trade_id
                        AND n.notification_type = 'new_trade'
                    WHERE n.id IS NULL
                    AND t.status = 'open'
                    ORDER BY t.alert_timestamp DESC
                    LIMIT 10
                """)

                trades = cursor.fetchall()

                for trade in trades:
                    try:
                        # Send notification
                        message_id = self.notifier.send_new_trade_alert(
                            dict(trade)
                        )

                        # Log notification
                        self._log_notification(
                            cursor,
                            trade['id'],
                            'new_trade',
                            message_id,
                            'sent' if message_id else 'failed'
                        )

                        logger.info(
                            f"New trade notification sent: "
                            f"{trade['ticker']} ({trade['profile_username']})"
                        )

                    except Exception as e:
                        logger.error(
                            f"Failed to send notification for trade {trade['id']}: {e}"
                        )
                        self._log_notification(
                            cursor,
                            trade['id'],
                            'new_trade',
                            None,
                            'failed',
                            str(e)
                        )

                conn.commit()
                return len(trades)

        except Exception as e:
            logger.error(f"Error processing new trades: {e}", exc_info=True)
            return 0

    def process_trade_updates(self) -> int:
        """
        Process and notify about trade updates.

        Returns:
            Number of updates processed
        """

        # TODO: Implement change detection logic
        # This would require tracking previous states
        return 0

    def process_closed_trades(self) -> int:
        """
        Process and notify about closed trades.

        Returns:
            Number of closed trades processed
        """

        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()

                # Find recently closed trades without closure notifications
                cursor.execute("""
                    SELECT
                        t.id, t.ticker, t.strategy, t.pnl, t.pnl_percent,
                        t.entry_date, t.exit_date,
                        p.username as profile_username
                    FROM xtrades_trades t
                    JOIN xtrades_profiles p ON t.profile_id = p.id
                    LEFT JOIN xtrades_notifications n
                        ON t.id = n.trade_id
                        AND n.notification_type = 'trade_closed'
                    WHERE n.id IS NULL
                    AND t.status = 'closed'
                    AND t.exit_date > NOW() - INTERVAL '24 hours'
                    ORDER BY t.exit_date DESC
                    LIMIT 10
                """)

                trades = cursor.fetchall()

                for trade in trades:
                    try:
                        # Send notification
                        message_id = self.notifier.send_trade_closed_alert(
                            dict(trade)
                        )

                        # Log notification
                        self._log_notification(
                            cursor,
                            trade['id'],
                            'trade_closed',
                            message_id,
                            'sent' if message_id else 'failed'
                        )

                        logger.info(
                            f"Trade closed notification sent: "
                            f"{trade['ticker']} P&L: {trade['pnl']} "
                            f"({trade['profile_username']})"
                        )

                    except Exception as e:
                        logger.error(
                            f"Failed to send closure notification for "
                            f"trade {trade['id']}: {e}"
                        )
                        self._log_notification(
                            cursor,
                            trade['id'],
                            'trade_closed',
                            None,
                            'failed',
                            str(e)
                        )

                conn.commit()
                return len(trades)

        except Exception as e:
            logger.error(f"Error processing closed trades: {e}", exc_info=True)
            return 0

    def send_summary(self) -> bool:
        """
        Send daily trading summary.

        Returns:
            True if successful, False otherwise
        """

        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()

                cursor.execute("""
                    SELECT
                        COUNT(*) FILTER (
                            WHERE entry_date::date = CURRENT_DATE
                        ) as new_trades,
                        COUNT(*) FILTER (
                            WHERE exit_date::date = CURRENT_DATE
                        ) as closed_trades,
                        COUNT(*) FILTER (
                            WHERE status = 'open'
                        ) as total_trades,
                        COALESCE(SUM(pnl) FILTER (
                            WHERE exit_date::date = CURRENT_DATE
                        ), 0) as total_pnl,
                        ROUND(
                            100.0 * COUNT(*) FILTER (
                                WHERE pnl > 0
                                AND exit_date::date = CURRENT_DATE
                            ) / NULLIF(COUNT(*) FILTER (
                                WHERE exit_date::date = CURRENT_DATE
                            ), 0),
                            2
                        ) as win_rate,
                        (
                            SELECT COUNT(*)
                            FROM xtrades_profiles
                            WHERE active = TRUE
                        ) as active_profiles
                    FROM xtrades_trades
                """)

                stats = cursor.fetchone()

                summary_data = {
                    'new_trades': stats[0],
                    'closed_trades': stats[1],
                    'total_trades': stats[2],
                    'total_pnl': Decimal(str(stats[3])),
                    'win_rate': float(stats[4] or 0),
                    'active_profiles': stats[5]
                }

                message_id = self.notifier.send_daily_summary(summary_data)

                if message_id:
                    logger.info("Daily summary sent successfully")
                    return True
                else:
                    logger.warning("Failed to send daily summary")
                    return False

        except Exception as e:
            logger.error(f"Error sending summary: {e}", exc_info=True)
            return False

    def _get_connection(self) -> None:
        """Get database connection with RealDictCursor."""
        return psycopg2.connect(
            self.db_url,
            cursor_factory=RealDictCursor
        )

    def _log_notification(
        self,
        cursor,
        trade_id: int,
        notification_type: str,
        message_id: Optional[str],
        status: str,
        error_message: Optional[str] = None
    ):
        """
        Log notification to database.

        Args:
            cursor: Database cursor
            trade_id: Trade ID
            notification_type: Type of notification
            message_id: Telegram message ID (if successful)
            status: 'sent' or 'failed'
            error_message: Error message (if failed)
        """

        cursor.execute("""
            INSERT INTO xtrades_notifications
            (trade_id, notification_type, telegram_message_id, status, error_message)
            VALUES (%s, %s, %s, %s, %s)
        """, (trade_id, notification_type, message_id, status, error_message))

    def get_stats(self) -> Dict[str, Any]:
        """
        Get notification statistics.

        Returns:
            Dictionary with statistics
        """

        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()

                cursor.execute("""
                    SELECT
                        COUNT(*) as total_notifications,
                        COUNT(*) FILTER (WHERE status = 'sent') as sent,
                        COUNT(*) FILTER (WHERE status = 'failed') as failed,
                        COUNT(*) FILTER (
                            WHERE notification_type = 'new_trade'
                        ) as new_trade_count,
                        COUNT(*) FILTER (
                            WHERE notification_type = 'trade_closed'
                        ) as closed_trade_count,
                        MAX(sent_at) as last_notification
                    FROM xtrades_notifications
                    WHERE sent_at > NOW() - INTERVAL '24 hours'
                """)

                stats = cursor.fetchone()

                return {
                    'total_notifications': stats[0],
                    'sent': stats[1],
                    'failed': stats[2],
                    'new_trades': stats[3],
                    'closed_trades': stats[4],
                    'last_notification': stats[5]
                }

        except Exception as e:
            logger.error(f"Error getting stats: {e}")
            return {}


def main():
    """Main entry point."""

    parser = argparse.ArgumentParser(
        description='Xtrades Telegram Integration'
    )
    parser.add_argument(
        '--interval',
        type=int,
        default=60,
        help='Check interval in seconds (default: 60)'
    )
    parser.add_argument(
        '--summary',
        action='store_true',
        help='Send daily summary and exit'
    )
    parser.add_argument(
        '--test',
        action='store_true',
        help='Test connection and exit'
    )
    parser.add_argument(
        '--stats',
        action='store_true',
        help='Show notification statistics and exit'
    )

    args = parser.parse_args()

    # Initialize integration
    integration = XtradesTelegramIntegration(check_interval=args.interval)

    try:
        if args.test:
            # Test mode
            print("Testing Telegram connection...")
            if integration.notifier.test_connection():
                print("Connection successful!")
                sys.exit(0)
            else:
                print("Connection failed!")
                sys.exit(1)

        elif args.summary:
            # Send summary and exit
            print("Sending daily summary...")
            if integration.send_summary():
                print("Summary sent successfully!")
                sys.exit(0)
            else:
                print("Failed to send summary")
                sys.exit(1)

        elif args.stats:
            # Show stats and exit
            stats = integration.get_stats()
            print("\nNotification Statistics (Last 24 Hours):")
            print(f"  Total: {stats.get('total_notifications', 0)}")
            print(f"  Sent: {stats.get('sent', 0)}")
            print(f"  Failed: {stats.get('failed', 0)}")
            print(f"  New Trades: {stats.get('new_trades', 0)}")
            print(f"  Closed Trades: {stats.get('closed_trades', 0)}")
            print(f"  Last: {stats.get('last_notification', 'N/A')}")
            sys.exit(0)

        else:
            # Normal mode - start monitoring
            integration.start()

    except KeyboardInterrupt:
        logger.info("Received interrupt signal, shutting down...")
        integration.stop()
        sys.exit(0)

    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == '__main__':
    main()

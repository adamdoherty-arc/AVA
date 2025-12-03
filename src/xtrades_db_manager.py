"""
Xtrades Database Manager - CRUD operations for Xtrades monitoring system

This module manages interactions with the Xtrades tables in the Magnus database:
- xtrades_profiles: Trader profiles to monitor
- xtrades_trades: Trade data scraped from profiles
- xtrades_sync_log: Synchronization audit log
- xtrades_notifications: Notification tracking

Security fixes applied:
- Connection pooling to prevent leaks
- All queries use parameterization to prevent SQL injection
- Transaction management for atomic operations
"""

import psycopg2
from psycopg2.extras import RealDictCursor
import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from dotenv import load_dotenv
import logging

# Import connection pool
try:
    from src.database.connection_pool import DatabaseConnectionPool
    USE_CONNECTION_POOL = True
except ImportError:
    logger.warning("Connection pool not available, using direct connections")
    USE_CONNECTION_POOL = False

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv(override=True)


class XtradesDBManager:
    """Manages Xtrades data in Magnus PostgreSQL database with connection pooling"""

    def __init__(self) -> None:
        self.db_config = {
            'host': os.getenv('DB_HOST', 'localhost'),
            'port': os.getenv('DB_PORT', '5432'),
            'user': os.getenv('DB_USER', 'postgres'),
            'password': os.getenv('DB_PASSWORD', 'postgres123!'),
            'database': os.getenv('DB_NAME', 'magnus')
        }

        # Initialize connection pool if available
        if USE_CONNECTION_POOL:
            try:
                # Initialize the singleton pool
                DatabaseConnectionPool()
                self.pool = DatabaseConnectionPool()._pool
                logger.info("XtradesDBManager initialized with connection pooling")
            except Exception as e:
                logger.warning(f"Failed to initialize connection pool: {e}, falling back to direct connections")
                self.pool = None
        else:
            self.pool = None

    def get_connection(self) -> None:
        """
        Get database connection from pool or create direct connection.

        Note: When using pool, connections must be returned via putconn()
        or use context manager for automatic cleanup.
        """
        if self.pool:
            return self.pool.getconn()
        else:
            return psycopg2.connect(**self.db_config)

    def release_connection(self, conn):
        """Release connection back to pool"""
        if self.pool:
            self.pool.putconn(conn)
        else:
            try:
                conn.close()
            except Exception:
                pass

    # =========================================================================
    # PROFILE MANAGEMENT
    # =========================================================================

    def add_profile(self, username: str, display_name: Optional[str] = None,
                    notes: Optional[str] = None) -> int:
        """
        Add a new Xtrades profile to monitor with transaction safety.

        Args:
            username: Xtrades.net username (must be unique)
            display_name: Optional display name for the profile
            notes: Optional notes about this profile

        Returns:
            profile_id: ID of the created profile

        Raises:
            Exception: If profile creation fails
        """
        conn = None
        try:
            conn = self.get_connection()
            with conn:
                with conn.cursor() as cur:
                    # Parameterized query - safe from SQL injection
                    cur.execute("""
                        INSERT INTO xtrades_profiles (username, display_name, notes)
                        VALUES (%s, %s, %s)
                        ON CONFLICT (username)
                        DO UPDATE SET
                            display_name = EXCLUDED.display_name,
                            notes = EXCLUDED.notes,
                            active = TRUE
                        RETURNING id
                    """, (username.lower().strip(), display_name, notes))

                    profile_id = cur.fetchone()[0]
                    # Transaction commits automatically with 'with conn' context
                    logger.info(f"Added/updated profile: {username} (ID: {profile_id})")
                    return profile_id

        except Exception as e:
            logger.error(f"Error adding profile {username}: {e}")
            # Transaction automatically rolled back by context manager
            raise
        finally:
            if conn:
                self.release_connection(conn)

    def get_active_profiles(self) -> List[Dict[str, Any]]:
        """
        Get all active profiles being monitored with safe connection handling.

        Returns:
            List of profile dictionaries with all fields
        """
        conn = None
        try:
            conn = self.get_connection()
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                # Parameterized query (active=TRUE is hardcoded constant, not user input)
                cur.execute("""
                    SELECT id, username, display_name, active, added_date,
                           last_sync, last_sync_status, total_trades_scraped, notes
                    FROM xtrades_profiles
                    WHERE active = %s
                    ORDER BY username
                """, (True,))

                profiles = cur.fetchall()
                return [dict(profile) for profile in profiles]

        except Exception as e:
            logger.error(f"Error fetching active profiles: {e}")
            return []
        finally:
            if conn:
                self.release_connection(conn)

    def get_all_profiles(self, include_inactive: bool = False) -> List[Dict[str, Any]]:
        """
        Get all profiles (optionally including inactive ones)

        Args:
            include_inactive: If True, include deactivated profiles

        Returns:
            List of profile dictionaries
        """
        conn = None
        try:
            conn = self.get_connection()
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                query = """
                    SELECT id, username, display_name, active, added_date,
                           last_sync, last_sync_status, total_trades_scraped, notes
                    FROM xtrades_profiles
                """

                if not include_inactive:
                    query += " WHERE active = TRUE"

                query += " ORDER BY username"

                cur.execute(query)
                profiles = cur.fetchall()
                return [dict(profile) for profile in profiles]

        except Exception as e:
            logger.error(f"Error fetching profiles: {e}")
            return []
        finally:
            if conn:
                self.release_connection(conn)

    def get_profile_by_id(self, profile_id: int) -> Optional[Dict[str, Any]]:
        """
        Get a specific profile by ID

        Args:
            profile_id: Profile ID

        Returns:
            Profile dictionary or None if not found
        """
        conn = None
        try:
            conn = self.get_connection()
            cur = conn.cursor(cursor_factory=RealDictCursor)

            cur.execute("""
                SELECT id, username, display_name, active, added_date,
                       last_sync, last_sync_status, total_trades_scraped, notes
                FROM xtrades_profiles
                WHERE id = %s
            """, (profile_id,))

            profile = cur.fetchone()
            return dict(profile) if profile else None

        except Exception as e:
            logger.error(f"Error fetching profile {profile_id}: {e}")
            return None
        finally:
            if conn:
                self.release_connection(conn)

    def get_profile_by_username(self, username: str) -> Optional[Dict[str, Any]]:
        """
        Get a specific profile by username

        Args:
            username: Xtrades username

        Returns:
            Profile dictionary or None if not found
        """
        conn = None
        try:
            conn = self.get_connection()
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute("""
                    SELECT id, username, display_name, active, added_date,
                           last_sync, last_sync_status, total_trades_scraped, notes
                    FROM xtrades_profiles
                    WHERE username = %s
                """, (username.lower().strip(),))

                profile = cur.fetchone()
                return dict(profile) if profile else None

        except Exception as e:
            logger.error(f"Error fetching profile {username}: {e}")
            return None
        finally:
            if conn:
                self.release_connection(conn)

    def update_profile_sync_status(self, profile_id: int, status: str,
                                   trades_count: Optional[int] = None) -> bool:
        """
        Update profile's last sync timestamp and status

        Args:
            profile_id: Profile ID
            status: Sync status ('success', 'error', 'pending')
            trades_count: Optional new total trades count (will increment if provided)

        Returns:
            True if successful, False otherwise
        """
        conn = None
        try:
            conn = self.get_connection()
            cur = conn.cursor()

            if trades_count is not None:
                cur.execute("""
                    UPDATE xtrades_profiles
                    SET last_sync = NOW(),
                        last_sync_status = %s,
                        total_trades_scraped = total_trades_scraped + %s
                    WHERE id = %s
                """, (status, trades_count, profile_id))
            else:
                cur.execute("""
                    UPDATE xtrades_profiles
                    SET last_sync = NOW(),
                        last_sync_status = %s
                    WHERE id = %s
                """, (status, profile_id))

            conn.commit()
            logger.info(f"Updated sync status for profile {profile_id}: {status}")
            return True

        except Exception as e:
            logger.error(f"Error updating profile sync status: {e}")
            if conn:
                conn.rollback()
            return False
        finally:
            if conn:
                self.release_connection(conn)

    def deactivate_profile(self, profile_id: int) -> bool:
        """
        Deactivate a profile (soft delete)

        Args:
            profile_id: Profile ID to deactivate

        Returns:
            True if successful, False otherwise
        """
        conn = None
        try:
            conn = self.get_connection()
            cur = conn.cursor()

            cur.execute("""
                UPDATE xtrades_profiles
                SET active = FALSE
                WHERE id = %s
            """, (profile_id,))

            conn.commit()
            logger.info(f"Deactivated profile ID {profile_id}")
            return True

        except Exception as e:
            logger.error(f"Error deactivating profile {profile_id}: {e}")
            if conn:
                conn.rollback()
            return False
        finally:
            if conn:
                self.release_connection(conn)

    def reactivate_profile(self, profile_id: int) -> bool:
        """
        Reactivate a previously deactivated profile

        Args:
            profile_id: Profile ID to reactivate

        Returns:
            True if successful, False otherwise
        """
        conn = None
        try:
            conn = self.get_connection()
            cur = conn.cursor()

            cur.execute("""
                UPDATE xtrades_profiles
                SET active = TRUE
                WHERE id = %s
            """, (profile_id,))

            conn.commit()
            logger.info(f"Reactivated profile ID {profile_id}")
            return True

        except Exception as e:
            logger.error(f"Error reactivating profile {profile_id}: {e}")
            if conn:
                conn.rollback()
            return False
        finally:
            if conn:
                self.release_connection(conn)

    # =========================================================================
    # TRADE MANAGEMENT
    # =========================================================================

    def add_trade(self, trade_data: Dict[str, Any]) -> int:
        """
        Insert a new trade from Xtrades alert

        Args:
            trade_data: Dictionary containing trade fields:
                - profile_id (required)
                - ticker (required)
                - alert_timestamp (required)
                - strategy, action, entry_price, entry_date, quantity,
                - strike_price, expiration_date, alert_text, xtrades_alert_id

        Returns:
            trade_id: ID of the created trade

        Raises:
            Exception: If trade creation fails
        """
        conn = None
        try:
            conn = self.get_connection()
            cur = conn.cursor()

            cur.execute("""
                INSERT INTO xtrades_trades (
                    profile_id, ticker, strategy, action, entry_price, entry_date,
                    quantity, strike_price, expiration_date, alert_text,
                    alert_timestamp, xtrades_alert_id, status, pnl_percent
                )
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                RETURNING id
            """, (
                trade_data.get('profile_id'),
                trade_data.get('ticker', '').upper(),
                trade_data.get('strategy'),
                trade_data.get('action'),
                trade_data.get('entry_price'),
                trade_data.get('entry_date'),
                trade_data.get('quantity', 1),
                trade_data.get('strike_price'),
                trade_data.get('expiration_date'),
                trade_data.get('alert_text'),
                trade_data.get('alert_timestamp'),
                trade_data.get('xtrades_alert_id'),
                trade_data.get('status', 'open'),
                trade_data.get('pnl_percent')
            ))

            trade_id = cur.fetchone()[0]
            conn.commit()
            logger.info(f"Added trade: {trade_data.get('ticker')} - {trade_data.get('strategy')} (ID: {trade_id})")
            return trade_id

        except Exception as e:
            logger.error(f"Error adding trade: {e}")
            if conn:
                conn.rollback()
            raise
        finally:
            if conn:
                self.release_connection(conn)

    def update_trade(self, trade_id: int, update_data: Dict[str, Any]) -> bool:
        """
        Update an existing trade (typically to close it or update P&L)

        Args:
            trade_id: Trade ID to update
            update_data: Dictionary with fields to update:
                - exit_price, exit_date, pnl, pnl_percent, status, etc.

        Returns:
            True if successful, False otherwise
        """
        conn = None
        try:
            conn = self.get_connection()
            cur = conn.cursor()

            # Build dynamic UPDATE query based on provided fields
            update_fields = []
            update_values = []

            allowed_fields = {
                'exit_price', 'exit_date', 'pnl', 'pnl_percent', 'status',
                'entry_price', 'entry_date', 'quantity', 'strategy', 'action',
                'strike_price', 'expiration_date', 'alert_text'
            }

            for field, value in update_data.items():
                if field in allowed_fields:
                    update_fields.append(f"{field} = %s")
                    update_values.append(value)

            if not update_fields:
                logger.warning(f"No valid fields to update for trade {trade_id}")
                return False

            # Always update updated_at
            update_fields.append("updated_at = NOW()")

            query = f"""
                UPDATE xtrades_trades
                SET {', '.join(update_fields)}
                WHERE id = %s
            """
            update_values.append(trade_id)

            cur.execute(query, update_values)
            conn.commit()

            if cur.rowcount > 0:
                logger.info(f"Updated trade ID {trade_id}: {list(update_data.keys())}")
                return True
            else:
                logger.warning(f"Trade ID {trade_id} not found for update")
                return False

        except Exception as e:
            logger.error(f"Error updating trade {trade_id}: {e}")
            if conn:
                conn.rollback()
            return False
        finally:
            if conn:
                self.release_connection(conn)

    def get_trade_by_id(self, trade_id: int) -> Optional[Dict[str, Any]]:
        """
        Get a specific trade by ID

        Args:
            trade_id: Trade ID

        Returns:
            Trade dictionary or None if not found
        """
        conn = None
        try:
            conn = self.get_connection()
            cur = conn.cursor(cursor_factory=RealDictCursor)

            cur.execute("""
                SELECT * FROM xtrades_trades
                WHERE id = %s
            """, (trade_id,))

            trade = cur.fetchone()
            return dict(trade) if trade else None

        except Exception as e:
            logger.error(f"Error fetching trade {trade_id}: {e}")
            return None
        finally:
            if conn:
                self.release_connection(conn)

    def get_trades_by_profile(self, profile_id: int,
                              status: Optional[str] = None,
                              limit: int = 100) -> List[Dict[str, Any]]:
        """
        Get trades for a specific profile with optional status filter

        Args:
            profile_id: Profile ID
            status: Optional status filter ('open', 'closed', 'expired')
            limit: Maximum number of trades to return

        Returns:
            List of trade dictionaries
        """
        conn = None
        try:
            conn = self.get_connection()
            cur = conn.cursor(cursor_factory=RealDictCursor)

            query = """
                SELECT * FROM xtrades_trades
                WHERE profile_id = %s
            """
            params = [profile_id]

            if status:
                query += " AND status = %s"
                params.append(status)

            query += " ORDER BY alert_timestamp DESC LIMIT %s"
            params.append(limit)

            cur.execute(query, params)
            trades = cur.fetchall()
            return [dict(trade) for trade in trades]

        except Exception as e:
            logger.error(f"Error fetching trades for profile {profile_id}: {e}")
            return []
        finally:
            if conn:
                self.release_connection(conn)

    def get_all_trades(self, status: Optional[str] = None,
                       ticker: Optional[str] = None,
                       limit: int = 100) -> List[Dict[str, Any]]:
        """
        Get all trades with optional filters

        Args:
            status: Optional status filter ('open', 'closed', 'expired')
            ticker: Optional ticker filter
            limit: Maximum number of trades to return

        Returns:
            List of trade dictionaries
        """
        conn = None
        try:
            conn = self.get_connection()
            cur = conn.cursor(cursor_factory=RealDictCursor)

            query = "SELECT * FROM xtrades_trades WHERE 1=1"
            params = []

            if status:
                query += " AND status = %s"
                params.append(status)

            if ticker:
                query += " AND ticker = %s"
                params.append(ticker.upper())

            query += " ORDER BY alert_timestamp DESC LIMIT %s"
            params.append(limit)

            cur.execute(query, params)
            trades = cur.fetchall()
            return [dict(trade) for trade in trades]

        except Exception as e:
            logger.error(f"Error fetching trades: {e}")
            return []
        finally:
            if conn:
                self.release_connection(conn)

    def find_existing_trade(self, profile_id: int, ticker: str,
                           alert_timestamp: datetime) -> Optional[int]:
        """
        Check if a trade already exists to avoid duplicates

        Args:
            profile_id: Profile ID
            ticker: Stock ticker
            alert_timestamp: Timestamp of the alert (can be datetime object or ISO string)

        Returns:
            trade_id if found, None otherwise
        """
        conn = None
        try:
            conn = self.get_connection()
            cur = conn.cursor()

            # Convert string timestamp to datetime if needed
            if isinstance(alert_timestamp, str):
                try:
                    # Handle ISO format with or without timezone info
                    alert_timestamp = datetime.fromisoformat(
                        alert_timestamp.replace('Z', '+00:00')
                    )
                except (ValueError, AttributeError) as e:
                    logger.error(f"Failed to parse alert_timestamp '{alert_timestamp}': {e}")
                    return None

            # Allow 1-minute tolerance for timestamp matching
            cur.execute("""
                SELECT id FROM xtrades_trades
                WHERE profile_id = %s
                  AND ticker = %s
                  AND alert_timestamp BETWEEN %s - INTERVAL '1 minute'
                                           AND %s + INTERVAL '1 minute'
                LIMIT 1
            """, (profile_id, ticker.upper(), alert_timestamp, alert_timestamp))

            result = cur.fetchone()
            return result[0] if result else None

        except Exception as e:
            logger.error(f"Error finding existing trade: {e}")
            return None
        finally:
            if conn:
                self.release_connection(conn)

    def get_open_trades_by_profile(self, profile_id: int) -> List[Dict[str, Any]]:
        """
        Get all open trades for a profile (convenience method)

        Args:
            profile_id: Profile ID

        Returns:
            List of open trade dictionaries
        """
        return self.get_trades_by_profile(profile_id, status='open', limit=1000)

    def close_trade(self, trade_id: int, exit_price: float,
                    exit_date: Optional[datetime] = None,
                    status: str = 'closed') -> bool:
        """
        Close a trade and calculate P&L

        Args:
            trade_id: Trade ID to close
            exit_price: Exit price
            exit_date: Exit date (default: now)
            status: Final status ('closed' or 'expired')

        Returns:
            True if successful, False otherwise
        """
        conn = None
        try:
            conn = self.get_connection()
            cur = conn.cursor(cursor_factory=RealDictCursor)

            # Get current trade data
            cur.execute("SELECT * FROM xtrades_trades WHERE id = %s", (trade_id,))
            trade = cur.fetchone()

            if not trade:
                logger.error(f"Trade {trade_id} not found")
                return False

            # Calculate P&L
            entry_price = float(trade['entry_price']) if trade['entry_price'] else 0
            quantity = trade['quantity'] or 1

            pnl = (exit_price - entry_price) * quantity * 100  # Options are per 100 shares
            pnl_percent = ((exit_price - entry_price) / entry_price * 100) if entry_price > 0 else 0

            # Update trade
            update_data = {
                'exit_price': exit_price,
                'exit_date': exit_date or datetime.now(),
                'pnl': pnl,
                'pnl_percent': pnl_percent,
                'status': status
            }

            return self.update_trade(trade_id, update_data)

        except Exception as e:
            logger.error(f"Error closing trade {trade_id}: {e}")
            return False
        finally:
            if conn:
                self.release_connection(conn)

    # =========================================================================
    # SYNC LOGGING
    # =========================================================================

    def log_sync_start(self) -> int:
        """
        Create a new sync log entry at the start of a sync operation

        Returns:
            sync_log_id: ID of the created sync log entry

        Raises:
            Exception: If log creation fails
        """
        conn = None
        try:
            conn = self.get_connection()
            cur = conn.cursor()

            cur.execute("""
                INSERT INTO xtrades_sync_log (status)
                VALUES ('running')
                RETURNING id
            """)

            sync_log_id = cur.fetchone()[0]
            conn.commit()
            logger.info(f"Started sync operation (Log ID: {sync_log_id})")
            return sync_log_id

        except Exception as e:
            logger.error(f"Error creating sync log: {e}")
            if conn:
                conn.rollback()
            raise
        finally:
            if conn:
                self.release_connection(conn)

    def log_sync_complete(self, sync_log_id: int, stats: Dict[str, Any]) -> bool:
        """
        Update sync log with completion statistics

        Args:
            sync_log_id: Sync log ID to update
            stats: Dictionary containing:
                - profiles_synced: Number of profiles processed
                - trades_found: Total trades found
                - new_trades: New trades added
                - updated_trades: Existing trades updated
                - errors: Error messages (if any)
                - duration_seconds: How long the sync took
                - status: Final status ('success', 'partial', 'failed')

        Returns:
            True if successful, False otherwise
        """
        conn = None
        try:
            conn = self.get_connection()
            cur = conn.cursor()

            cur.execute("""
                UPDATE xtrades_sync_log
                SET profiles_synced = %s,
                    trades_found = %s,
                    new_trades = %s,
                    updated_trades = %s,
                    errors = %s,
                    duration_seconds = %s,
                    status = %s
                WHERE id = %s
            """, (
                stats.get('profiles_synced', 0),
                stats.get('trades_found', 0),
                stats.get('new_trades', 0),
                stats.get('updated_trades', 0),
                stats.get('errors'),
                stats.get('duration_seconds'),
                stats.get('status', 'success'),
                sync_log_id
            ))

            conn.commit()
            logger.info(f"Sync log {sync_log_id} completed: {stats.get('status', 'success')}")
            return True

        except Exception as e:
            logger.error(f"Error updating sync log {sync_log_id}: {e}")
            if conn:
                conn.rollback()
            return False
        finally:
            if conn:
                self.release_connection(conn)

    def get_sync_history(self, limit: int = 50) -> List[Dict[str, Any]]:
        """
        Get recent sync operation history

        Args:
            limit: Maximum number of sync logs to return

        Returns:
            List of sync log dictionaries
        """
        conn = None
        try:
            conn = self.get_connection()
            cur = conn.cursor(cursor_factory=RealDictCursor)

            cur.execute("""
                SELECT * FROM xtrades_sync_log
                ORDER BY sync_timestamp DESC
                LIMIT %s
            """, (limit,))

            logs = cur.fetchall()
            return [dict(log) for log in logs]

        except Exception as e:
            logger.error(f"Error fetching sync history: {e}")
            return []
        finally:
            if conn:
                self.release_connection(conn)

    def get_latest_sync(self) -> Optional[Dict[str, Any]]:
        """
        Get the most recent sync operation

        Returns:
            Latest sync log dictionary or None
        """
        logs = self.get_sync_history(limit=1)
        return logs[0] if logs else None

    # =========================================================================
    # NOTIFICATIONS
    # =========================================================================

    def log_notification(self, trade_id: int, notification_type: str,
                        telegram_msg_id: Optional[str] = None,
                        status: str = 'sent',
                        error_message: Optional[str] = None) -> int:
        """
        Record a notification that was sent for a trade

        Args:
            trade_id: Trade ID the notification is for
            notification_type: Type ('new_trade', 'trade_update', 'trade_closed')
            telegram_msg_id: Telegram message ID (for tracking/editing)
            status: Notification status ('sent' or 'failed')
            error_message: Error message if status is 'failed'

        Returns:
            notification_id: ID of the notification record

        Raises:
            Exception: If notification logging fails
        """
        conn = None
        try:
            conn = self.get_connection()
            cur = conn.cursor()

            cur.execute("""
                INSERT INTO xtrades_notifications (
                    trade_id, notification_type, telegram_message_id,
                    status, error_message
                )
                VALUES (%s, %s, %s, %s, %s)
                RETURNING id
            """, (trade_id, notification_type, telegram_msg_id, status, error_message))

            notification_id = cur.fetchone()[0]
            conn.commit()
            logger.info(f"Logged notification for trade {trade_id}: {notification_type}")
            return notification_id

        except Exception as e:
            logger.error(f"Error logging notification: {e}")
            if conn:
                conn.rollback()
            raise
        finally:
            if conn:
                self.release_connection(conn)

    def get_unsent_notifications(self) -> List[Dict[str, Any]]:
        """
        Get trades that need notifications (trades without notification records)

        Returns:
            List of trade dictionaries that need notifications
        """
        conn = None
        try:
            conn = self.get_connection()
            cur = conn.cursor(cursor_factory=RealDictCursor)

            cur.execute("""
                SELECT t.* FROM xtrades_trades t
                LEFT JOIN xtrades_notifications n ON t.id = n.trade_id
                WHERE n.id IS NULL
                  AND t.status = 'open'
                ORDER BY t.alert_timestamp DESC
            """)

            trades = cur.fetchall()
            return [dict(trade) for trade in trades]

        except Exception as e:
            logger.error(f"Error fetching unsent notifications: {e}")
            return []
        finally:
            if conn:
                self.release_connection(conn)

    def get_notifications_for_trade(self, trade_id: int) -> List[Dict[str, Any]]:
        """
        Get all notifications sent for a specific trade

        Args:
            trade_id: Trade ID

        Returns:
            List of notification dictionaries
        """
        conn = None
        try:
            conn = self.get_connection()
            cur = conn.cursor(cursor_factory=RealDictCursor)

            cur.execute("""
                SELECT * FROM xtrades_notifications
                WHERE trade_id = %s
                ORDER BY sent_at DESC
            """, (trade_id,))

            notifications = cur.fetchall()
            return [dict(notif) for notif in notifications]

        except Exception as e:
            logger.error(f"Error fetching notifications for trade {trade_id}: {e}")
            return []
        finally:
            if conn:
                self.release_connection(conn)

    # =========================================================================
    # ANALYTICS & STATISTICS
    # =========================================================================

    def get_profile_stats(self, profile_id: int) -> Dict[str, Any]:
        """
        Get statistics for a specific profile

        Args:
            profile_id: Profile ID

        Returns:
            Dictionary with profile statistics:
                - total_trades, open_trades, closed_trades
                - total_pnl, avg_pnl, win_rate
                - best_trade, worst_trade
        """
        conn = None
        try:
            conn = self.get_connection()
            cur = conn.cursor(cursor_factory=RealDictCursor)

            # Get aggregate stats
            cur.execute("""
                SELECT
                    COUNT(*) as total_trades,
                    COUNT(CASE WHEN status = 'open' THEN 1 END) as open_trades,
                    COUNT(CASE WHEN status IN ('closed', 'expired') THEN 1 END) as closed_trades,
                    COALESCE(SUM(CASE WHEN status IN ('closed', 'expired') THEN pnl END), 0) as total_pnl,
                    COALESCE(AVG(CASE WHEN status IN ('closed', 'expired') THEN pnl END), 0) as avg_pnl,
                    COUNT(CASE WHEN pnl > 0 THEN 1 END)::FLOAT /
                        NULLIF(COUNT(CASE WHEN status IN ('closed', 'expired') THEN 1 END), 0) * 100 as win_rate,
                    MAX(pnl) as best_trade_pnl,
                    MIN(pnl) as worst_trade_pnl
                FROM xtrades_trades
                WHERE profile_id = %s
            """, (profile_id,))

            stats = dict(cur.fetchone())

            # Get best trade details
            cur.execute("""
                SELECT ticker, pnl, pnl_percent
                FROM xtrades_trades
                WHERE profile_id = %s AND pnl IS NOT NULL
                ORDER BY pnl DESC
                LIMIT 1
            """, (profile_id,))

            best_trade = cur.fetchone()
            stats['best_trade'] = dict(best_trade) if best_trade else None

            # Get worst trade details
            cur.execute("""
                SELECT ticker, pnl, pnl_percent
                FROM xtrades_trades
                WHERE profile_id = %s AND pnl IS NOT NULL
                ORDER BY pnl ASC
                LIMIT 1
            """, (profile_id,))

            worst_trade = cur.fetchone()
            stats['worst_trade'] = dict(worst_trade) if worst_trade else None

            return stats

        except Exception as e:
            logger.error(f"Error getting profile stats for {profile_id}: {e}")
            return {
                'total_trades': 0,
                'open_trades': 0,
                'closed_trades': 0,
                'total_pnl': 0.0,
                'avg_pnl': 0.0,
                'win_rate': 0.0,
                'best_trade': None,
                'worst_trade': None
            }
        finally:
            if conn:
                self.release_connection(conn)

    def get_overall_stats(self) -> Dict[str, Any]:
        """
        Get system-wide statistics across all profiles

        Returns:
            Dictionary with overall statistics
        """
        conn = None
        try:
            conn = self.get_connection()
            cur = conn.cursor(cursor_factory=RealDictCursor)

            # Get aggregate stats
            cur.execute("""
                SELECT
                    COUNT(DISTINCT profile_id) as total_profiles,
                    COUNT(*) as total_trades,
                    COUNT(CASE WHEN status = 'open' THEN 1 END) as open_trades,
                    COUNT(CASE WHEN status IN ('closed', 'expired') THEN 1 END) as closed_trades,
                    COALESCE(SUM(CASE WHEN status IN ('closed', 'expired') THEN pnl END), 0) as total_pnl,
                    COALESCE(AVG(CASE WHEN status IN ('closed', 'expired') THEN pnl END), 0) as avg_pnl,
                    COUNT(CASE WHEN pnl > 0 THEN 1 END)::FLOAT /
                        NULLIF(COUNT(CASE WHEN status IN ('closed', 'expired') THEN 1 END), 0) * 100 as win_rate
                FROM xtrades_trades
            """)

            stats = dict(cur.fetchone())

            # Get most active ticker
            cur.execute("""
                SELECT ticker, COUNT(*) as trade_count
                FROM xtrades_trades
                GROUP BY ticker
                ORDER BY trade_count DESC
                LIMIT 1
            """)

            top_ticker = cur.fetchone()
            stats['most_active_ticker'] = dict(top_ticker) if top_ticker else None

            # Get most profitable profile
            cur.execute("""
                SELECT p.username, SUM(t.pnl) as total_pnl
                FROM xtrades_trades t
                JOIN xtrades_profiles p ON t.profile_id = p.id
                WHERE t.status IN ('closed', 'expired')
                GROUP BY p.username
                ORDER BY total_pnl DESC
                LIMIT 1
            """)

            top_profile = cur.fetchone()
            stats['top_profile'] = dict(top_profile) if top_profile else None

            return stats

        except Exception as e:
            logger.error(f"Error getting overall stats: {e}")
            return {
                'total_profiles': 0,
                'total_trades': 0,
                'open_trades': 0,
                'closed_trades': 0,
                'total_pnl': 0.0,
                'avg_pnl': 0.0,
                'win_rate': 0.0,
                'most_active_ticker': None,
                'top_profile': None
            }
        finally:
            if conn:
                self.release_connection(conn)

    def get_trades_by_ticker(self, ticker: str, limit: int = 50) -> List[Dict[str, Any]]:
        """
        Get all trades for a specific ticker across all profiles

        Args:
            ticker: Stock ticker symbol
            limit: Maximum trades to return

        Returns:
            List of trade dictionaries
        """
        conn = None
        try:
            conn = self.get_connection()
            cur = conn.cursor(cursor_factory=RealDictCursor)

            cur.execute("""
                SELECT t.*, p.username, p.display_name
                FROM xtrades_trades t
                JOIN xtrades_profiles p ON t.profile_id = p.id
                WHERE t.ticker = %s
                ORDER BY t.alert_timestamp DESC
                LIMIT %s
            """, (ticker.upper(), limit))

            trades = cur.fetchall()
            return [dict(trade) for trade in trades]

        except Exception as e:
            logger.error(f"Error fetching trades for ticker {ticker}: {e}")
            return []
        finally:
            if conn:
                self.release_connection(conn)

    def get_recent_activity(self, days: int = 7, limit: int = 100) -> List[Dict[str, Any]]:
        """
        Get recent trading activity across all profiles

        Args:
            days: Number of days to look back (must be positive integer, max 365)
            limit: Maximum trades to return

        Returns:
            List of trade dictionaries with profile info
        """
        # Validate days parameter to prevent SQL injection
        if not isinstance(days, int) or days < 0 or days > 365:
            logger.warning(f"Invalid days parameter: {days}, defaulting to 7")
            days = 7

        conn = None
        try:
            conn = self.get_connection()
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                # Use CAST for safe interval parameterization
                cur.execute("""
                    SELECT t.*, p.username, p.display_name
                    FROM xtrades_trades t
                    JOIN xtrades_profiles p ON t.profile_id = p.id
                    WHERE t.alert_timestamp >= NOW() - CAST(%s || ' days' AS INTERVAL)
                    ORDER BY t.alert_timestamp DESC
                    LIMIT %s
                """, (str(days), limit))

                trades = cur.fetchall()
                return [dict(trade) for trade in trades]

        except Exception as e:
            logger.error(f"Error fetching recent activity: {e}")
            return []
        finally:
            if conn:
                self.release_connection(conn)

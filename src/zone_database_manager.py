"""
Zone Database Manager
Handles all PostgreSQL CRUD operations for supply/demand zones
"""

import psycopg2
from psycopg2.extras import RealDictCursor, execute_batch
from typing import List, Dict, Optional, Tuple
from datetime import datetime
import logging
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

logger = logging.getLogger(__name__)


class ZoneDatabaseManager:
    """
    Manages supply/demand zone data in PostgreSQL

    Tables:
    - sd_zones: Main zones table
    - sd_zone_tests: Zone test history
    - sd_alerts: Alert history
    - sd_scan_log: Scanner audit log
    """

    def __init__(self, connection_string: Optional[str] = None):
        """
        Initialize database manager

        Args:
            connection_string: PostgreSQL connection string
                             (defaults to environment variable or local postgres)
        """
        # Ensure environment variables are loaded
        load_dotenv()

        if connection_string:
            self.connection_string = connection_string
        else:
            # Try environment variable first
            self.connection_string = os.getenv('POSTGRES_CONNECTION')

            if not self.connection_string:
                # Build from individual environment variables
                db_password = os.getenv('DB_PASSWORD', 'postgres123!')
                db_user = os.getenv('DB_USER', 'postgres')
                db_host = os.getenv('DB_HOST', 'localhost')
                db_port = os.getenv('DB_PORT', '5432')
                db_name = os.getenv('DB_NAME', 'magnus')

                self.connection_string = f'postgresql://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}'

    def get_connection(self) -> None:
        """Get database connection"""
        return psycopg2.connect(self.connection_string)

    # =========================================================================
    # ZONE CRUD OPERATIONS
    # =========================================================================

    def save_zone(self, zone: Dict) -> int:
        """
        Save a new zone to database

        Args:
            zone: Zone dictionary from ZoneDetector/ZoneAnalyzer

        Returns:
            zone_id (primary key)
        """

        query = """
            INSERT INTO sd_zones (
                ticker, zone_type, timeframe,
                zone_top, zone_bottom, zone_midpoint,
                formed_date, formation_candle_index,
                approach_volume, departure_volume, volume_ratio,
                strength_score, time_at_zone, rejection_candles,
                status, test_count, is_active, notes
            ) VALUES (
                %(symbol)s, %(zone_type)s, %(timeframe)s,
                %(zone_top)s, %(zone_bottom)s, %(zone_midpoint)s,
                %(formed_date)s, %(formation_candle_index)s,
                %(approach_volume)s, %(departure_volume)s, %(volume_ratio)s,
                %(strength_score)s, %(time_at_zone)s, %(rejection_candles)s,
                %(status)s, %(test_count)s, %(is_active)s, %(notes)s
            )
            RETURNING id
        """

        # Set defaults
        zone_data = {
            'symbol': zone['symbol'],
            'zone_type': zone['zone_type'],
            'timeframe': zone.get('timeframe', '1d'),
            'zone_top': zone['zone_top'],
            'zone_bottom': zone['zone_bottom'],
            'zone_midpoint': zone['zone_midpoint'],
            'formed_date': zone.get('formed_date', datetime.now()),
            'formation_candle_index': zone.get('formation_candle_index'),
            'approach_volume': zone.get('approach_volume'),
            'departure_volume': zone.get('departure_volume'),
            'volume_ratio': zone.get('volume_ratio'),
            'strength_score': zone.get('strength_score', 50),
            'time_at_zone': zone.get('time_at_zone'),
            'rejection_candles': zone.get('rejection_candles'),
            'status': zone.get('status', 'FRESH'),
            'test_count': zone.get('test_count', 0),
            'is_active': zone.get('is_active', True),
            'notes': zone.get('notes', '')
        }

        with self.get_connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute(query, zone_data)
                zone_id = cursor.fetchone()[0]
                conn.commit()

        logger.info(f"Saved zone {zone_id} for {zone['symbol']}")
        return zone_id

    def save_zones_batch(self, zones: List[Dict]) -> List[int]:
        """
        Save multiple zones in batch

        Args:
            zones: List of zone dictionaries

        Returns:
            List of zone IDs
        """

        if not zones:
            return []

        zone_ids = []

        with self.get_connection() as conn:
            with conn.cursor() as cursor:
                for zone in zones:
                    zone_id = self.save_zone(zone)
                    zone_ids.append(zone_id)

        logger.info(f"Saved {len(zone_ids)} zones in batch")
        return zone_ids

    def get_active_zones(
        self,
        symbol: Optional[str] = None,
        zone_type: Optional[str] = None,
        min_strength: int = 0
    ) -> List[Dict]:
        """
        Get active zones

        Args:
            symbol: Filter by ticker (optional)
            zone_type: Filter by type: 'SUPPLY' or 'DEMAND' (optional)
            min_strength: Minimum strength score (default: 0)

        Returns:
            List of zone dictionaries
        """

        query = """
            SELECT *
            FROM sd_zones
            WHERE is_active = TRUE
              AND status != 'BROKEN'
              AND strength_score >= %s
        """

        params = [min_strength]

        if symbol:
            query += " AND ticker = %s"
            params.append(symbol)

        if zone_type:
            query += " AND zone_type = %s"
            params.append(zone_type)

        query += " ORDER BY strength_score DESC, formed_date DESC"

        with self.get_connection() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cursor:
                cursor.execute(query, params)
                zones = cursor.fetchall()

        return [dict(zone) for zone in zones]

    def get_zone_by_id(self, zone_id: int) -> Optional[Dict]:
        """Get zone by ID"""

        query = "SELECT * FROM sd_zones WHERE id = %s"

        with self.get_connection() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cursor:
                cursor.execute(query, (zone_id,))
                zone = cursor.fetchone()

        return dict(zone) if zone else None

    def update_zone(self, zone_id: int, updates: Dict) -> bool:
        """
        Update zone fields

        Args:
            zone_id: Zone ID
            updates: Dictionary of fields to update

        Returns:
            True if successful
        """

        # Build SET clause dynamically
        set_clauses = []
        params = []

        for key, value in updates.items():
            set_clauses.append(f"{key} = %s")
            params.append(value)

        # Add updated_at
        set_clauses.append("updated_at = NOW()")

        query = f"""
            UPDATE sd_zones
            SET {', '.join(set_clauses)}
            WHERE id = %s
        """

        params.append(zone_id)

        with self.get_connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute(query, params)
                conn.commit()
                success = cursor.rowcount > 0

        if success:
            logger.info(f"Updated zone {zone_id}")

        return success

    def mark_zone_broken(self, zone_id: int) -> bool:
        """Mark zone as broken"""

        updates = {
            'status': 'BROKEN',
            'broken_date': datetime.now(),
            'is_active': False
        }

        return self.update_zone(zone_id, updates)

    def increment_test_count(self, zone_id: int) -> bool:
        """Increment zone test count"""

        query = """
            UPDATE sd_zones
            SET test_count = test_count + 1,
                last_test_date = NOW(),
                updated_at = NOW()
            WHERE id = %s
        """

        with self.get_connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute(query, (zone_id,))
                conn.commit()
                return cursor.rowcount > 0

    def deactivate_old_zones(self, days: int = 90) -> int:
        """
        Deactivate zones older than N days

        Args:
            days: Age threshold in days

        Returns:
            Number of zones deactivated
        """

        query = """
            UPDATE sd_zones
            SET is_active = FALSE,
                updated_at = NOW()
            WHERE is_active = TRUE
              AND formed_date < NOW() - INTERVAL '%s days'
        """

        with self.get_connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute(query, (days,))
                conn.commit()
                count = cursor.rowcount

        logger.info(f"Deactivated {count} zones older than {days} days")
        return count

    # =========================================================================
    # ZONE TEST OPERATIONS
    # =========================================================================

    def save_zone_test(self, test: Dict) -> int:
        """
        Save zone test event

        Args:
            test: Test dictionary with zone_id, test details

        Returns:
            test_id
        """

        query = """
            INSERT INTO sd_zone_tests (
                zone_id, test_date, test_price, test_type,
                penetration_percent, reaction_candles, bounce_percent,
                test_volume, held, broke_through
            ) VALUES (
                %(zone_id)s, %(test_date)s, %(test_price)s, %(test_type)s,
                %(penetration_percent)s, %(reaction_candles)s, %(bounce_percent)s,
                %(test_volume)s, %(held)s, %(broke_through)s
            )
            RETURNING id
        """

        test_data = {
            'zone_id': test['zone_id'],
            'test_date': test.get('test_date', datetime.now()),
            'test_price': test['test_price'],
            'test_type': test.get('test_type', 'TOUCH'),
            'penetration_percent': test.get('penetration_percent'),
            'reaction_candles': test.get('reaction_candles'),
            'bounce_percent': test.get('bounce_percent'),
            'test_volume': test.get('test_volume'),
            'held': test.get('held', True),
            'broke_through': test.get('broke_through', False)
        }

        with self.get_connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute(query, test_data)
                test_id = cursor.fetchone()[0]
                conn.commit()

        # Increment zone test count
        self.increment_test_count(test['zone_id'])

        logger.info(f"Saved test {test_id} for zone {test['zone_id']}")
        return test_id

    def get_zone_tests(self, zone_id: int) -> List[Dict]:
        """Get all tests for a zone"""

        query = """
            SELECT *
            FROM sd_zone_tests
            WHERE zone_id = %s
            ORDER BY test_date DESC
        """

        with self.get_connection() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cursor:
                cursor.execute(query, (zone_id,))
                tests = cursor.fetchall()

        return [dict(test) for test in tests]

    # =========================================================================
    # ALERT OPERATIONS
    # =========================================================================

    def save_alert(self, alert: Dict) -> int:
        """
        Save alert record

        Args:
            alert: Alert dictionary

        Returns:
            alert_id
        """

        query = """
            INSERT INTO sd_alerts (
                zone_id, ticker, alert_type, alert_price, zone_type,
                distance_to_zone, zone_strength, setup_quality,
                telegram_message_id, status, error_message
            ) VALUES (
                %(zone_id)s, %(ticker)s, %(alert_type)s, %(alert_price)s, %(zone_type)s,
                %(distance_to_zone)s, %(zone_strength)s, %(setup_quality)s,
                %(telegram_message_id)s, %(status)s, %(error_message)s
            )
            RETURNING id
        """

        alert_data = {
            'zone_id': alert['zone_id'],
            'ticker': alert['ticker'],
            'alert_type': alert['alert_type'],
            'alert_price': alert['alert_price'],
            'zone_type': alert['zone_type'],
            'distance_to_zone': alert.get('distance_to_zone'),
            'zone_strength': alert.get('zone_strength'),
            'setup_quality': alert.get('setup_quality', 'MEDIUM'),
            'telegram_message_id': alert.get('telegram_message_id'),
            'status': alert.get('status', 'sent'),
            'error_message': alert.get('error_message')
        }

        with self.get_connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute(query, alert_data)
                alert_id = cursor.fetchone()[0]
                conn.commit()

        logger.info(f"Saved alert {alert_id} for {alert['ticker']}")
        return alert_id

    def get_recent_alerts(
        self,
        hours: int = 24,
        symbol: Optional[str] = None
    ) -> List[Dict]:
        """
        Get recent alerts

        Args:
            hours: Look back N hours
            symbol: Filter by ticker (optional)

        Returns:
            List of alert dictionaries
        """

        query = """
            SELECT *
            FROM sd_alerts
            WHERE sent_at >= NOW() - INTERVAL '%s hours'
        """

        params = [hours]

        if symbol:
            query += " AND ticker = %s"
            params.append(symbol)

        query += " ORDER BY sent_at DESC"

        with self.get_connection() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cursor:
                cursor.execute(query, params)
                alerts = cursor.fetchall()

        return [dict(alert) for alert in alerts]

    # =========================================================================
    # SCAN LOG OPERATIONS
    # =========================================================================

    def log_scan(self, scan_data: Dict) -> int:
        """
        Log scanner operation

        Args:
            scan_data: Scan log dictionary

        Returns:
            log_id
        """

        query = """
            INSERT INTO sd_scan_log (
                scan_type, tickers_scanned, zones_found, zones_updated,
                alerts_sent, duration_seconds, status, errors
            ) VALUES (
                %(scan_type)s, %(tickers_scanned)s, %(zones_found)s, %(zones_updated)s,
                %(alerts_sent)s, %(duration_seconds)s, %(status)s, %(errors)s
            )
            RETURNING id
        """

        log_data = {
            'scan_type': scan_data.get('scan_type', 'ZONE_DETECTION'),
            'tickers_scanned': scan_data.get('tickers_scanned', 0),
            'zones_found': scan_data.get('zones_found', 0),
            'zones_updated': scan_data.get('zones_updated', 0),
            'alerts_sent': scan_data.get('alerts_sent', 0),
            'duration_seconds': scan_data.get('duration_seconds'),
            'status': scan_data.get('status', 'success'),
            'errors': scan_data.get('errors')
        }

        with self.get_connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute(query, log_data)
                log_id = cursor.fetchone()[0]
                conn.commit()

        return log_id

    def get_scan_logs(self, limit: int = 50) -> List[Dict]:
        """Get recent scan logs"""

        query = """
            SELECT *
            FROM sd_scan_log
            ORDER BY scan_timestamp DESC
            LIMIT %s
        """

        with self.get_connection() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cursor:
                cursor.execute(query, (limit,))
                logs = cursor.fetchall()

        return [dict(log) for log in logs]

    # =========================================================================
    # QUERY HELPERS
    # =========================================================================

    def get_zones_near_price(
        self,
        symbol: str,
        current_price: float,
        distance_pct: float = 5.0
    ) -> List[Dict]:
        """
        Get zones within X% of current price

        Args:
            symbol: Ticker symbol
            current_price: Current stock price
            distance_pct: Distance threshold (%)

        Returns:
            List of zones near current price
        """

        # Calculate price range
        upper_bound = current_price * (1 + distance_pct / 100)
        lower_bound = current_price * (1 - distance_pct / 100)

        query = """
            SELECT *
            FROM sd_zones
            WHERE ticker = %s
              AND is_active = TRUE
              AND status != 'BROKEN'
              AND (
                (zone_top >= %s AND zone_bottom <= %s) OR
                (zone_top >= %s AND zone_top <= %s) OR
                (zone_bottom >= %s AND zone_bottom <= %s)
              )
            ORDER BY strength_score DESC
        """

        params = [symbol, lower_bound, upper_bound, lower_bound, upper_bound, lower_bound, upper_bound]

        with self.get_connection() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cursor:
                cursor.execute(query, params)
                zones = cursor.fetchall()

        return [dict(zone) for zone in zones]

    def get_zone_statistics(self, symbol: Optional[str] = None) -> Dict:
        """
        Get zone statistics

        Args:
            symbol: Filter by ticker (optional)

        Returns:
            Dictionary with statistics
        """

        query = """
            SELECT
                COUNT(*) as total_zones,
                COUNT(CASE WHEN status = 'FRESH' THEN 1 END) as fresh_zones,
                COUNT(CASE WHEN status = 'TESTED' THEN 1 END) as tested_zones,
                COUNT(CASE WHEN zone_type = 'DEMAND' THEN 1 END) as demand_zones,
                COUNT(CASE WHEN zone_type = 'SUPPLY' THEN 1 END) as supply_zones,
                AVG(strength_score) as avg_strength,
                AVG(test_count) as avg_tests,
                COUNT(CASE WHEN is_active THEN 1 END) as active_zones
            FROM sd_zones
        """

        params = []

        if symbol:
            query += " WHERE ticker = %s"
            params.append(symbol)

        with self.get_connection() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cursor:
                cursor.execute(query, params)
                stats = cursor.fetchone()

        return dict(stats) if stats else {}


if __name__ == "__main__":
    # Test database manager
    logging.basicConfig(level=logging.INFO)

    db = ZoneDatabaseManager()

    print("Testing ZoneDatabaseManager...")

    # Get active zones
    zones = db.get_active_zones(min_strength=50)
    print(f"\nFound {len(zones)} active zones with strength >= 50")

    # Get statistics
    stats = db.get_zone_statistics()
    print(f"\nZone Statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")

    # Get recent scan logs
    logs = db.get_scan_logs(limit=5)
    print(f"\nRecent scan logs: {len(logs)}")

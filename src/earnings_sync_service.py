"""
Earnings Sync Service - Robust earnings data synchronization from Robinhood API

This service syncs earnings data for all stocks in the database, storing both
historical earnings and upcoming earnings events. It includes comprehensive error
handling, retry logic, and progress tracking.

Key Features:
- Fetches up to 8 quarters of historical earnings from Robinhood
- Calculates surprise percentages and beat/miss classifications
- Tracks sync status for monitoring
- Handles rate limiting and API errors gracefully
- Batch processing with progress updates
- Upsert logic to prevent duplicates

Author: Wheel Strategy Trading System
Date: 2025-10-28
"""

import os
import logging
import time
from datetime import datetime, date, timedelta
from typing import Dict, List, Optional, Tuple, Any
from decimal import Decimal, InvalidOperation
import psycopg2
from psycopg2.extras import RealDictCursor, execute_batch
import robin_stocks.robinhood as rh
from dotenv import load_dotenv
import json

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()


class EarningsDataError(Exception):
    """Custom exception for earnings data errors"""
    pass


class RobinhoodAPIError(Exception):
    """Custom exception for Robinhood API errors"""
    pass


class EarningsSyncService:
    """
    Service to sync earnings data from Robinhood API to PostgreSQL database

    Handles:
    - Batch syncing for all stocks
    - Individual symbol sync
    - Historical earnings storage
    - Upcoming earnings calendar
    - Beat/miss pattern tracking
    - Comprehensive error handling and retry logic
    """

    def __init__(self, max_retries: int = 3, retry_delay: int = 5, batch_size: int = 50):
        """
        Initialize the Earnings Sync Service

        Args:
            max_retries: Maximum number of retry attempts for failed API calls
            retry_delay: Delay in seconds between retry attempts
            batch_size: Number of database inserts to batch together
        """
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.batch_size = batch_size
        self.is_logged_in = False

        # Database configuration
        self.db_config = {
            'host': os.getenv('DB_HOST', 'localhost'),
            'port': int(os.getenv('DB_PORT', '5432')),
            'database': os.getenv('DB_NAME', 'magnus'),
            'user': os.getenv('DB_USER', 'postgres'),
            'password': os.getenv('DB_PASSWORD', 'postgres123!')
        }

        # Robinhood credentials
        self.rh_username = os.getenv('ROBINHOOD_USERNAME')
        self.rh_password = os.getenv('ROBINHOOD_PASSWORD')

        if not self.rh_username or not self.rh_password:
            raise ValueError("Robinhood credentials not found in environment variables")

        logger.info("EarningsSyncService initialized")

    def get_db_connection(self) -> None:
        """Get database connection"""
        try:
            conn = psycopg2.connect(**self.db_config)
            return conn
        except Exception as e:
            logger.error(f"Failed to connect to database: {e}")
            raise

    def login_robinhood(self) -> bool:
        """
        Login to Robinhood API

        Returns:
            bool: True if login successful, False otherwise
        """
        if self.is_logged_in:
            return True

        try:
            logger.info("Logging in to Robinhood...")
            login_response = rh.login(self.rh_username, self.rh_password)

            if login_response:
                self.is_logged_in = True
                logger.info("Successfully logged in to Robinhood")
                return True
            else:
                logger.error("Robinhood login failed - no response")
                return False

        except Exception as e:
            logger.error(f"Robinhood login error: {e}")
            return False

    def logout_robinhood(self) -> None:
        """Logout from Robinhood API"""
        if self.is_logged_in:
            try:
                rh.logout()
                self.is_logged_in = False
                logger.info("Logged out from Robinhood")
            except Exception as e:
                logger.warning(f"Error during logout: {e}")

    def fetch_earnings_with_retry(self, symbol: str) -> Optional[List[Dict]]:
        """
        Fetch earnings data from Robinhood with retry logic

        Args:
            symbol: Stock ticker symbol

        Returns:
            List of earnings data dictionaries or None if failed
        """
        for attempt in range(1, self.max_retries + 1):
            try:
                logger.debug(f"Fetching earnings for {symbol} (attempt {attempt}/{self.max_retries})")
                earnings_data = rh.get_earnings(symbol)

                if earnings_data is None:
                    logger.debug(f"No earnings data available for {symbol}")
                    return None

                if isinstance(earnings_data, list):
                    logger.debug(f"Found {len(earnings_data)} earnings records for {symbol}")
                    return earnings_data
                else:
                    logger.warning(f"Unexpected data type for {symbol}: {type(earnings_data)}")
                    return None

            except Exception as e:
                logger.warning(f"Error fetching earnings for {symbol} (attempt {attempt}): {e}")

                if attempt < self.max_retries:
                    time.sleep(self.retry_delay)
                else:
                    logger.error(f"Failed to fetch earnings for {symbol} after {self.max_retries} attempts")
                    return None

        return None

    def parse_earnings_record(self, record: Dict, symbol: str) -> Optional[Dict]:
        """
        Parse a single earnings record from Robinhood API

        Args:
            record: Raw earnings record from API
            symbol: Stock ticker symbol

        Returns:
            Parsed earnings dictionary or None if invalid
        """
        try:
            # Extract report information
            report = record.get('report', {})
            report_date_str = report.get('date')
            if not report_date_str:
                logger.warning(f"No report date for {symbol}, skipping record")
                return None

            # Parse date
            try:
                report_date = datetime.strptime(report_date_str, '%Y-%m-%d').date()
            except ValueError:
                logger.warning(f"Invalid date format for {symbol}: {report_date_str}")
                return None

            # Extract EPS data
            eps_data = record.get('eps', {})
            eps_actual = self._safe_decimal(eps_data.get('actual'))
            eps_estimate = self._safe_decimal(eps_data.get('estimate'))

            # Calculate surprise
            eps_surprise = None
            eps_surprise_percent = None
            beat_miss = 'unknown'

            if eps_actual is not None and eps_estimate is not None:
                eps_surprise = eps_actual - eps_estimate

                if eps_estimate != 0:
                    eps_surprise_percent = float((eps_surprise / abs(eps_estimate)) * 100)

                    # Classify beat/miss
                    if eps_actual > eps_estimate:
                        beat_miss = 'beat'
                    elif eps_actual < eps_estimate:
                        beat_miss = 'miss'
                    else:
                        beat_miss = 'meet'

            # Extract quarter/year
            year = record.get('year')
            quarter = record.get('quarter')

            # Extract call information
            call_data = record.get('call', {})
            call_datetime_str = call_data.get('datetime')
            call_datetime = None

            if call_datetime_str:
                try:
                    call_datetime = datetime.fromisoformat(call_datetime_str.replace('Z', '+00:00'))
                except Exception as e:
                    logger.debug(f"Could not parse call datetime: {e}")

            call_url = call_data.get('broadcast_url')

            # Determine earnings time (before market open / after market close)
            earnings_time = 'unspecified'
            timing = report.get('timing', '').lower()
            if 'bmo' in timing or 'before' in timing:
                earnings_time = 'bmo'
            elif 'amc' in timing or 'after' in timing:
                earnings_time = 'amc'

            # Build parsed record
            parsed = {
                'symbol': symbol,
                'report_date': report_date,
                'fiscal_year': year,
                'fiscal_quarter': quarter,
                'earnings_time': earnings_time,
                'eps_actual': float(eps_actual) if eps_actual is not None else None,
                'eps_estimate': float(eps_estimate) if eps_estimate is not None else None,
                'eps_surprise': float(eps_surprise) if eps_surprise is not None else None,
                'eps_surprise_percent': eps_surprise_percent,
                'beat_miss': beat_miss,
                'call_datetime': call_datetime,
                'call_broadcast_url': call_url,
                'raw_data': json.dumps(record)
            }

            return parsed

        except Exception as e:
            logger.error(f"Error parsing earnings record for {symbol}: {e}")
            return None

    def _safe_decimal(self, value: Any) -> Optional[Decimal]:
        """Safely convert value to Decimal"""
        if value is None or value == '':
            return None
        try:
            return Decimal(str(value))
        except (InvalidOperation, ValueError, TypeError):
            return None

    def store_historical_earnings(self, earnings_records: List[Dict]) -> int:
        """
        Store historical earnings records in database

        Args:
            earnings_records: List of parsed earnings dictionaries

        Returns:
            Number of records inserted/updated
        """
        if not earnings_records:
            return 0

        conn = self.get_db_connection()
        cur = conn.cursor()
        stored_count = 0

        try:
            insert_query = """
                INSERT INTO earnings_history (
                    symbol, report_date, fiscal_year, fiscal_quarter, earnings_time,
                    eps_actual, eps_estimate, eps_surprise, eps_surprise_percent,
                    beat_miss, call_datetime, call_broadcast_url, raw_data
                )
                VALUES (
                    %(symbol)s, %(report_date)s, %(fiscal_year)s, %(fiscal_quarter)s,
                    %(earnings_time)s, %(eps_actual)s, %(eps_estimate)s, %(eps_surprise)s,
                    %(eps_surprise_percent)s, %(beat_miss)s, %(call_datetime)s,
                    %(call_broadcast_url)s, %(raw_data)s
                )
                ON CONFLICT (symbol, report_date, fiscal_quarter, fiscal_year)
                DO UPDATE SET
                    earnings_time = EXCLUDED.earnings_time,
                    eps_actual = COALESCE(EXCLUDED.eps_actual, earnings_history.eps_actual),
                    eps_estimate = COALESCE(EXCLUDED.eps_estimate, earnings_history.eps_estimate),
                    eps_surprise = COALESCE(EXCLUDED.eps_surprise, earnings_history.eps_surprise),
                    eps_surprise_percent = COALESCE(EXCLUDED.eps_surprise_percent, earnings_history.eps_surprise_percent),
                    beat_miss = COALESCE(EXCLUDED.beat_miss, earnings_history.beat_miss),
                    call_datetime = COALESCE(EXCLUDED.call_datetime, earnings_history.call_datetime),
                    call_broadcast_url = COALESCE(EXCLUDED.call_broadcast_url, earnings_history.call_broadcast_url),
                    raw_data = EXCLUDED.raw_data,
                    updated_at = NOW()
            """

            execute_batch(cur, insert_query, earnings_records, page_size=self.batch_size)
            stored_count = len(earnings_records)
            conn.commit()

            logger.info(f"Stored {stored_count} historical earnings records")

        except Exception as e:
            logger.error(f"Error storing historical earnings: {e}")
            conn.rollback()
            stored_count = 0

        finally:
            cur.close()
            conn.close()

        return stored_count

    def store_upcoming_earnings(self, earnings_records: List[Dict]) -> int:
        """
        Store upcoming earnings events in database

        Args:
            earnings_records: List of parsed earnings dictionaries

        Returns:
            Number of records inserted/updated
        """
        if not earnings_records:
            return 0

        # Filter for future earnings only
        today = date.today()
        upcoming_records = [r for r in earnings_records if r['report_date'] >= today]

        if not upcoming_records:
            return 0

        conn = self.get_db_connection()
        cur = conn.cursor()
        stored_count = 0

        try:
            insert_query = """
                INSERT INTO earnings_events (
                    symbol, earnings_date, fiscal_year, fiscal_quarter, earnings_time,
                    eps_estimate, call_datetime, call_broadcast_url, is_confirmed,
                    has_occurred, raw_data
                )
                VALUES (
                    %(symbol)s, %(report_date)s, %(fiscal_year)s, %(fiscal_quarter)s,
                    %(earnings_time)s, %(eps_estimate)s, %(call_datetime)s,
                    %(call_broadcast_url)s, TRUE, FALSE, %(raw_data)s
                )
                ON CONFLICT (symbol, earnings_date, fiscal_quarter, fiscal_year)
                DO UPDATE SET
                    earnings_time = EXCLUDED.earnings_time,
                    eps_estimate = COALESCE(EXCLUDED.eps_estimate, earnings_events.eps_estimate),
                    call_datetime = COALESCE(EXCLUDED.call_datetime, earnings_events.call_datetime),
                    call_broadcast_url = COALESCE(EXCLUDED.call_broadcast_url, earnings_events.call_broadcast_url),
                    is_confirmed = TRUE,
                    raw_data = EXCLUDED.raw_data,
                    updated_at = NOW()
            """

            execute_batch(cur, insert_query, upcoming_records, page_size=self.batch_size)
            stored_count = len(upcoming_records)
            conn.commit()

            logger.info(f"Stored {stored_count} upcoming earnings events")

        except Exception as e:
            logger.error(f"Error storing upcoming earnings: {e}")
            conn.rollback()
            stored_count = 0

        finally:
            cur.close()
            conn.close()

        return stored_count

    def update_sync_status(self, symbol: str, status: str,
                          historical_count: int = 0, upcoming_count: int = 0,
                          error_msg: Optional[str] = None):
        """
        Update sync status for a symbol

        Args:
            symbol: Stock ticker
            status: Sync status (success, failed, partial, no_data)
            historical_count: Number of historical records found
            upcoming_count: Number of upcoming events found
            error_msg: Error message if failed
        """
        conn = self.get_db_connection()
        cur = conn.cursor()

        try:
            cur.execute(
                "SELECT update_sync_status(%s, %s, %s, %s, %s)",
                (symbol, status, historical_count, upcoming_count, error_msg)
            )
            conn.commit()

        except Exception as e:
            logger.error(f"Error updating sync status for {symbol}: {e}")
            conn.rollback()

        finally:
            cur.close()
            conn.close()

    def sync_symbol_earnings(self, symbol: str) -> Dict[str, Any]:
        """
        Sync earnings data for a single symbol

        Args:
            symbol: Stock ticker symbol

        Returns:
            Dictionary with sync results
        """
        result = {
            'symbol': symbol,
            'status': 'failed',
            'historical_count': 0,
            'upcoming_count': 0,
            'error': None
        }

        try:
            # Ensure we're logged in
            if not self.is_logged_in:
                if not self.login_robinhood():
                    result['error'] = 'Failed to login to Robinhood'
                    self.update_sync_status(symbol, 'failed', error_msg=result['error'])
                    return result

            # Fetch earnings data
            earnings_data = self.fetch_earnings_with_retry(symbol)

            if earnings_data is None:
                result['status'] = 'no_data'
                result['error'] = 'No earnings data available'
                self.update_sync_status(symbol, 'no_data')
                return result

            # Parse all records
            parsed_records = []
            for record in earnings_data:
                parsed = self.parse_earnings_record(record, symbol)
                if parsed:
                    parsed_records.append(parsed)

            if not parsed_records:
                result['status'] = 'no_data'
                self.update_sync_status(symbol, 'no_data')
                return result

            # Store historical earnings
            historical_count = self.store_historical_earnings(parsed_records)

            # Store upcoming earnings
            upcoming_count = self.store_upcoming_earnings(parsed_records)

            # Update result
            result['status'] = 'success'
            result['historical_count'] = historical_count
            result['upcoming_count'] = upcoming_count

            # Update sync status
            self.update_sync_status(
                symbol, 'success',
                historical_count=historical_count,
                upcoming_count=upcoming_count
            )

            logger.info(f"Successfully synced {symbol}: {historical_count} historical, {upcoming_count} upcoming")

        except Exception as e:
            result['error'] = str(e)
            result['status'] = 'failed'
            logger.error(f"Error syncing {symbol}: {e}")
            self.update_sync_status(symbol, 'failed', error_msg=str(e))

        return result

    def get_all_stock_symbols(self) -> List[str]:
        """
        Get all stock symbols from database

        Returns:
            List of stock symbols
        """
        conn = self.get_db_connection()
        cur = conn.cursor()
        symbols = []

        try:
            # Try multiple potential table names
            table_queries = [
                "SELECT DISTINCT symbol FROM stocks WHERE is_active = TRUE ORDER BY symbol",
                "SELECT DISTINCT symbol FROM tv_watchlist_symbols ORDER BY symbol",
                "SELECT DISTINCT symbol FROM watchlist_items wi JOIN stocks s ON wi.stock_id = s.id ORDER BY symbol"
            ]

            for query in table_queries:
                try:
                    cur.execute(query)
                    symbols = [row[0] for row in cur.fetchall()]
                    if symbols:
                        logger.info(f"Found {len(symbols)} symbols from database")
                        break
                except Exception:
                    continue

            if not symbols:
                logger.warning("No symbols found in database")

        except Exception as e:
            logger.error(f"Error fetching symbols: {e}")

        finally:
            cur.close()
            conn.close()

        return symbols

    def sync_all_stocks_earnings(self, limit: Optional[int] = None,
                                rate_limit_delay: float = 1.0) -> Dict[str, Any]:
        """
        Sync earnings for all stocks in database

        Args:
            limit: Optional limit on number of stocks to sync
            rate_limit_delay: Delay between API calls to avoid rate limiting

        Returns:
            Dictionary with overall sync results
        """
        logger.info("Starting full earnings sync for all stocks")

        # Login to Robinhood
        if not self.login_robinhood():
            return {
                'status': 'failed',
                'error': 'Failed to login to Robinhood',
                'total_stocks': 0,
                'successful': 0,
                'failed': 0,
                'no_data': 0
            }

        # Get all symbols
        symbols = self.get_all_stock_symbols()

        if not symbols:
            return {
                'status': 'failed',
                'error': 'No symbols found in database',
                'total_stocks': 0,
                'successful': 0,
                'failed': 0,
                'no_data': 0
            }

        # Apply limit if specified
        if limit:
            symbols = symbols[:limit]

        total_stocks = len(symbols)
        successful = 0
        failed = 0
        no_data = 0
        total_historical = 0
        total_upcoming = 0

        logger.info(f"Syncing earnings for {total_stocks} stocks...")

        # Process each symbol
        for idx, symbol in enumerate(symbols, 1):
            try:
                # Progress update
                if idx % 10 == 0:
                    logger.info(f"Progress: {idx}/{total_stocks} ({idx/total_stocks*100:.1f}%)")

                # Sync symbol
                result = self.sync_symbol_earnings(symbol)

                # Update counters
                if result['status'] == 'success':
                    successful += 1
                    total_historical += result['historical_count']
                    total_upcoming += result['upcoming_count']
                elif result['status'] == 'no_data':
                    no_data += 1
                else:
                    failed += 1

                # Rate limiting delay
                if idx < total_stocks:  # Don't delay after last symbol
                    time.sleep(rate_limit_delay)

            except Exception as e:
                logger.error(f"Unexpected error processing {symbol}: {e}")
                failed += 1

        # Logout
        self.logout_robinhood()

        # Prepare summary
        summary = {
            'status': 'completed',
            'total_stocks': total_stocks,
            'successful': successful,
            'failed': failed,
            'no_data': no_data,
            'total_historical_records': total_historical,
            'total_upcoming_events': total_upcoming,
            'success_rate': f"{successful/total_stocks*100:.1f}%" if total_stocks > 0 else "0%"
        }

        logger.info(f"Earnings sync completed: {json.dumps(summary, indent=2)}")
        return summary

    def get_upcoming_earnings(self, days_ahead: int = 30) -> List[Dict]:
        """
        Get upcoming earnings events

        Args:
            days_ahead: Number of days to look ahead

        Returns:
            List of upcoming earnings events
        """
        conn = self.get_db_connection()
        cur = conn.cursor(cursor_factory=RealDictCursor)

        try:
            query = """
                SELECT * FROM v_upcoming_earnings
                WHERE earnings_date <= CURRENT_DATE + INTERVAL '%s days'
                ORDER BY earnings_date, symbol
            """
            cur.execute(query, (days_ahead,))
            results = cur.fetchall()
            return [dict(row) for row in results]

        except Exception as e:
            logger.error(f"Error fetching upcoming earnings: {e}")
            return []

        finally:
            cur.close()
            conn.close()

    def get_historical_earnings(self, symbol: str, limit: int = 8) -> List[Dict]:
        """
        Get historical earnings for a symbol

        Args:
            symbol: Stock ticker
            limit: Number of quarters to retrieve

        Returns:
            List of historical earnings records
        """
        conn = self.get_db_connection()
        cur = conn.cursor(cursor_factory=RealDictCursor)

        try:
            query = """
                SELECT * FROM earnings_history
                WHERE symbol = %s
                ORDER BY report_date DESC
                LIMIT %s
            """
            cur.execute(query, (symbol, limit))
            results = cur.fetchall()
            return [dict(row) for row in results]

        except Exception as e:
            logger.error(f"Error fetching historical earnings for {symbol}: {e}")
            return []

        finally:
            cur.close()
            conn.close()

    def calculate_beat_rate(self, symbol: str, lookback_quarters: int = 8) -> float:
        """
        Calculate earnings beat rate for a symbol

        Args:
            symbol: Stock ticker
            lookback_quarters: Number of quarters to analyze

        Returns:
            Beat rate percentage (0-100)
        """
        conn = self.get_db_connection()
        cur = conn.cursor()

        try:
            cur.execute(
                "SELECT calculate_beat_rate(%s, %s)",
                (symbol, lookback_quarters)
            )
            result = cur.fetchone()
            return float(result[0]) if result and result[0] is not None else 0.0

        except Exception as e:
            logger.error(f"Error calculating beat rate for {symbol}: {e}")
            return 0.0

        finally:
            cur.close()
            conn.close()


# Example usage
if __name__ == "__main__":
    # Initialize service
    service = EarningsSyncService(max_retries=3, retry_delay=5, batch_size=50)

    # Example 1: Sync a single symbol
    print("\n" + "="*80)
    print("Example 1: Sync Single Symbol")
    print("="*80)
    result = service.sync_symbol_earnings('AAPL')
    print(json.dumps(result, indent=2))

    # Example 2: Get historical earnings
    print("\n" + "="*80)
    print("Example 2: Historical Earnings for AAPL")
    print("="*80)
    historical = service.get_historical_earnings('AAPL', limit=4)
    for record in historical:
        print(f"{record['report_date']}: EPS {record['eps_actual']} vs {record['eps_estimate']} - {record['beat_miss']}")

    # Example 3: Calculate beat rate
    print("\n" + "="*80)
    print("Example 3: Calculate Beat Rate")
    print("="*80)
    beat_rate = service.calculate_beat_rate('AAPL', lookback_quarters=8)
    print(f"AAPL Beat Rate (last 8 quarters): {beat_rate}%")

    # Example 4: Get upcoming earnings
    print("\n" + "="*80)
    print("Example 4: Upcoming Earnings (Next 7 Days)")
    print("="*80)
    upcoming = service.get_upcoming_earnings(days_ahead=7)
    for event in upcoming:
        print(f"{event['earnings_date']}: {event['symbol']} - Beat Rate: {event.get('historical_beat_rate_pct', 'N/A')}%")

    # Example 5: Sync all stocks (with limit for testing)
    # Uncomment to run full sync
    # print("\n" + "="*80)
    # print("Example 5: Sync All Stocks")
    # print("="*80)
    # summary = service.sync_all_stocks_earnings(limit=10)  # Test with first 10 stocks
    # print(json.dumps(summary, indent=2))

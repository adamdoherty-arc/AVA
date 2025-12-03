"""
Delisted Symbols Manager

Manages tracking of delisted or problematic stock symbols in the database.
Provides functions to mark symbols as delisted, check status, and clean up data.
"""

import psycopg2
from psycopg2.extras import RealDictCursor
import os
from dotenv import load_dotenv
from datetime import datetime
from typing import List, Dict, Optional
import logging

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DelistedSymbolsManager:
    """Manages delisted symbols in the database"""

    def __init__(self) -> None:
        self.db_config = {
            'host': os.getenv('DB_HOST', 'localhost'),
            'port': os.getenv('DB_PORT', '5432'),
            'database': os.getenv('DB_NAME', 'magnus'),
            'user': os.getenv('DB_USER', 'postgres'),
            'password': os.getenv('DB_PASSWORD', 'postgres123!')
        }
        self.conn = None

    def connect(self) -> None:
        """Connect to PostgreSQL database"""
        try:
            self.conn = psycopg2.connect(**self.db_config)
            logger.info("Connected to database")
            return True
        except Exception as e:
            logger.error(f"Database connection error: {e}")
            return False

    def disconnect(self) -> None:
        """Disconnect from database"""
        if self.conn:
            self.conn.close()
            logger.info("Disconnected from database")

    def mark_symbol_as_delisted(
        self,
        symbol: str,
        reason: str = "Data unavailable from yfinance"
    ) -> bool:
        """
        Mark a symbol as delisted in the database.

        Args:
            symbol: Stock ticker symbol
            reason: Reason why symbol is being marked as delisted

        Returns:
            True if successful, False otherwise
        """
        if not self.conn:
            if not self.connect():
                return False

        try:
            cursor = self.conn.cursor()

            cursor.execute("""
                UPDATE stocks
                SET is_delisted = TRUE,
                    delisted_date = NOW(),
                    delisted_reason = %s
                WHERE symbol = %s
                  AND is_delisted = FALSE
            """, (reason, symbol.upper()))

            rows_affected = cursor.rowcount
            self.conn.commit()
            cursor.close()

            if rows_affected > 0:
                logger.info(f"Marked {symbol} as delisted: {reason}")
                return True
            else:
                logger.warning(f"Symbol {symbol} not found in database or already delisted")
                return False

        except Exception as e:
            logger.error(f"Error marking {symbol} as delisted: {e}")
            self.conn.rollback()
            return False

    def unmark_symbol_as_delisted(self, symbol: str) -> bool:
        """
        Remove delisted flag from a symbol (if it's back and trading).

        Args:
            symbol: Stock ticker symbol

        Returns:
            True if successful, False otherwise
        """
        if not self.conn:
            if not self.connect():
                return False

        try:
            cursor = self.conn.cursor()

            cursor.execute("""
                UPDATE stocks
                SET is_delisted = FALSE,
                    delisted_date = NULL,
                    delisted_reason = NULL
                WHERE symbol = %s
                  AND is_delisted = TRUE
            """, (symbol.upper(),))

            rows_affected = cursor.rowcount
            self.conn.commit()
            cursor.close()

            if rows_affected > 0:
                logger.info(f"Removed delisted flag from {symbol}")
                return True
            else:
                logger.warning(f"Symbol {symbol} not found or not marked as delisted")
                return False

        except Exception as e:
            logger.error(f"Error removing delisted flag from {symbol}: {e}")
            self.conn.rollback()
            return False

    def is_symbol_delisted(self, symbol: str) -> bool:
        """
        Check if a symbol is marked as delisted in the database.

        Args:
            symbol: Stock ticker symbol

        Returns:
            True if delisted, False otherwise
        """
        if not self.conn:
            if not self.connect():
                return False

        try:
            cursor = self.conn.cursor()

            cursor.execute("""
                SELECT is_delisted
                FROM stocks
                WHERE symbol = %s
            """, (symbol.upper(),))

            result = cursor.fetchone()
            cursor.close()

            if result:
                return result[0] is True
            return False

        except Exception as e:
            logger.error(f"Error checking delisted status for {symbol}: {e}")
            return False

    def get_all_delisted_symbols(self) -> List[Dict]:
        """
        Get all symbols marked as delisted.

        Returns:
            List of dictionaries with symbol information
        """
        if not self.conn:
            if not self.connect():
                return []

        try:
            cursor = self.conn.cursor(cursor_factory=RealDictCursor)

            cursor.execute("""
                SELECT symbol, name, delisted_date, delisted_reason
                FROM stocks
                WHERE is_delisted = TRUE
                ORDER BY delisted_date DESC
            """)

            results = cursor.fetchall()
            cursor.close()

            return [dict(row) for row in results]

        except Exception as e:
            logger.error(f"Error fetching delisted symbols: {e}")
            return []

    def bulk_mark_as_delisted(self, symbols: List[str], reason: str) -> int:
        """
        Mark multiple symbols as delisted at once.

        Args:
            symbols: List of stock ticker symbols
            reason: Reason for marking as delisted

        Returns:
            Number of symbols successfully marked
        """
        if not self.conn:
            if not self.connect():
                return 0

        marked_count = 0

        for symbol in symbols:
            if self.mark_symbol_as_delisted(symbol, reason):
                marked_count += 1

        logger.info(f"Marked {marked_count} out of {len(symbols)} symbols as delisted")
        return marked_count

    def cleanup_delisted_symbol_data(self, symbol: str) -> bool:
        """
        Clean up old option data and premiums for a delisted symbol.

        Args:
            symbol: Stock ticker symbol

        Returns:
            True if successful, False otherwise
        """
        if not self.conn:
            if not self.connect():
                return False

        try:
            cursor = self.conn.cursor()

            # Delete old option data
            cursor.execute("""
                DELETE FROM options
                WHERE symbol = %s
            """, (symbol.upper(),))
            options_deleted = cursor.rowcount

            # Delete old premium data
            cursor.execute("""
                DELETE FROM stock_premiums
                WHERE symbol = %s
            """, (symbol.upper(),))
            premiums_deleted = cursor.rowcount

            self.conn.commit()
            cursor.close()

            logger.info(
                f"Cleaned up data for {symbol}: "
                f"{options_deleted} option records, {premiums_deleted} premium records"
            )
            return True

        except Exception as e:
            logger.error(f"Error cleaning up data for {symbol}: {e}")
            self.conn.rollback()
            return False

    def get_delisted_summary(self) -> Dict:
        """
        Get a summary of delisted symbols.

        Returns:
            Dictionary with summary statistics
        """
        if not self.conn:
            if not self.connect():
                return {}

        try:
            cursor = self.conn.cursor(cursor_factory=RealDictCursor)

            cursor.execute("""
                SELECT
                    COUNT(*) as total_delisted,
                    COUNT(CASE WHEN delisted_date >= NOW() - INTERVAL '7 days' THEN 1 END) as delisted_last_week,
                    COUNT(CASE WHEN delisted_date >= NOW() - INTERVAL '30 days' THEN 1 END) as delisted_last_month
                FROM stocks
                WHERE is_delisted = TRUE
            """)

            result = cursor.fetchone()
            cursor.close()

            if result:
                return dict(result)
            return {}

        except Exception as e:
            logger.error(f"Error getting delisted summary: {e}")
            return {}

    def apply_schema_migration(self) -> bool:
        """
        Apply the database schema migration to add delisted tracking columns.

        Returns:
            True if successful, False otherwise
        """
        if not self.conn:
            if not self.connect():
                return False

        try:
            cursor = self.conn.cursor()

            # Add columns if they don't exist
            cursor.execute("""
                ALTER TABLE stocks
                ADD COLUMN IF NOT EXISTS is_delisted BOOLEAN DEFAULT FALSE
            """)

            cursor.execute("""
                ALTER TABLE stocks
                ADD COLUMN IF NOT EXISTS delisted_date TIMESTAMP
            """)

            cursor.execute("""
                ALTER TABLE stocks
                ADD COLUMN IF NOT EXISTS delisted_reason VARCHAR(255)
            """)

            # Create index
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_stocks_is_delisted ON stocks(is_delisted)
            """)

            self.conn.commit()
            cursor.close()

            logger.info("Successfully applied delisted tracking schema migration")
            return True

        except Exception as e:
            logger.error(f"Error applying schema migration: {e}")
            self.conn.rollback()
            return False


def main():
    """Demo usage of DelistedSymbolsManager"""
    manager = DelistedSymbolsManager()

    if not manager.connect():
        print("Failed to connect to database")
        return

    # Apply schema migration
    print("Applying schema migration...")
    manager.apply_schema_migration()

    # Mark known delisted symbols
    print("\nMarking known delisted symbols...")
    known_delisted = ['BMNR', 'PLUG', 'BBAI']
    manager.bulk_mark_as_delisted(
        known_delisted,
        "Known delisted - yfinance data unavailable"
    )

    # Get summary
    print("\nDelisted symbols summary:")
    summary = manager.get_delisted_summary()
    print(f"  Total delisted: {summary.get('total_delisted', 0)}")
    print(f"  Delisted last week: {summary.get('delisted_last_week', 0)}")
    print(f"  Delisted last month: {summary.get('delisted_last_month', 0)}")

    # List all delisted symbols
    print("\nAll delisted symbols:")
    delisted = manager.get_all_delisted_symbols()
    for symbol_info in delisted:
        print(f"  {symbol_info['symbol']}: {symbol_info['delisted_reason']}")

    manager.disconnect()


if __name__ == '__main__':
    main()

"""Portfolio Balance Tracker - Daily P/L and Balance History"""

import psycopg2
from psycopg2.extras import RealDictCursor
import os
from datetime import datetime, timedelta, date
from typing import Dict, List, Optional
from dotenv import load_dotenv
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()


class PortfolioBalanceTracker:
    """Tracks daily portfolio balances and P/L in database"""

    def __init__(self) -> None:
        self.db_config = {
            'host': os.getenv('DB_HOST', 'localhost'),
            'port': os.getenv('DB_PORT', '5432'),
            'user': os.getenv('DB_USER', 'postgres'),
            'password': os.getenv('DB_PASSWORD', 'postgres123!'),
            'database': os.getenv('DB_NAME', 'magnus')
        }
        self.initialize_tables()

    def get_connection(self) -> None:
        """Get database connection"""
        return psycopg2.connect(**self.db_config)

    def initialize_tables(self) -> None:
        """Create daily balance tables if they don't exist"""
        conn = self.get_connection()
        cur = conn.cursor()

        try:
            # Daily portfolio balances table
            cur.execute("""
                CREATE TABLE IF NOT EXISTS daily_portfolio_balances (
                    id SERIAL PRIMARY KEY,
                    date DATE UNIQUE NOT NULL,
                    starting_balance DECIMAL(12,2),
                    ending_balance DECIMAL(12,2) NOT NULL,
                    daily_pl DECIMAL(12,2),
                    daily_pl_percent DECIMAL(8,4),
                    buying_power DECIMAL(12,2),
                    options_value DECIMAL(12,2),
                    stock_value DECIMAL(12,2),
                    cash_value DECIMAL(12,2),
                    total_positions INTEGER,
                    notes TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)

            # Create index for faster date queries
            cur.execute("""
                CREATE INDEX IF NOT EXISTS idx_daily_balances_date
                ON daily_portfolio_balances(date DESC);
            """)

            conn.commit()
            logger.info("Portfolio balance tables initialized successfully")

        except Exception as e:
            conn.rollback()
            logger.error(f"Error initializing tables: {e}")
            raise
        finally:
            cur.close()
            conn.close()

    def record_daily_balance(
        self,
        balance_date: date,
        ending_balance: float,
        starting_balance: Optional[float] = None,
        buying_power: Optional[float] = None,
        options_value: Optional[float] = None,
        stock_value: Optional[float] = None,
        cash_value: Optional[float] = None,
        total_positions: Optional[int] = None,
        notes: Optional[str] = None
    ) -> bool:
        """
        Record or update daily balance snapshot

        Args:
            balance_date: Date for this balance
            ending_balance: Ending total portfolio value
            starting_balance: Starting balance (optional, will use previous day's ending)
            buying_power: Available buying power
            options_value: Total value of options positions
            stock_value: Total value of stock positions
            cash_value: Cash balance
            total_positions: Number of active positions
            notes: Optional notes for the day

        Returns:
            True if successful
        """
        conn = self.get_connection()
        cur = conn.cursor()

        try:
            # If starting balance not provided, get previous day's ending balance
            if starting_balance is None:
                cur.execute("""
                    SELECT ending_balance
                    FROM daily_portfolio_balances
                    WHERE date < %s
                    ORDER BY date DESC
                    LIMIT 1
                """, (balance_date,))

                result = cur.fetchone()
                starting_balance = float(result[0]) if result else ending_balance

            # Calculate daily P/L
            daily_pl = ending_balance - starting_balance
            daily_pl_percent = (daily_pl / starting_balance * 100) if starting_balance > 0 else 0

            # Insert or update
            cur.execute("""
                INSERT INTO daily_portfolio_balances (
                    date, starting_balance, ending_balance, daily_pl, daily_pl_percent,
                    buying_power, options_value, stock_value, cash_value,
                    total_positions, notes, updated_at
                )
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, CURRENT_TIMESTAMP)
                ON CONFLICT (date)
                DO UPDATE SET
                    starting_balance = EXCLUDED.starting_balance,
                    ending_balance = EXCLUDED.ending_balance,
                    daily_pl = EXCLUDED.daily_pl,
                    daily_pl_percent = EXCLUDED.daily_pl_percent,
                    buying_power = EXCLUDED.buying_power,
                    options_value = EXCLUDED.options_value,
                    stock_value = EXCLUDED.stock_value,
                    cash_value = EXCLUDED.cash_value,
                    total_positions = EXCLUDED.total_positions,
                    notes = EXCLUDED.notes,
                    updated_at = CURRENT_TIMESTAMP
            """, (
                balance_date, starting_balance, ending_balance, daily_pl, daily_pl_percent,
                buying_power, options_value, stock_value, cash_value,
                total_positions, notes
            ))

            conn.commit()
            logger.info(f"Recorded balance for {balance_date}: ${ending_balance:,.2f} (P/L: ${daily_pl:,.2f})")
            return True

        except Exception as e:
            conn.rollback()
            logger.error(f"Error recording daily balance: {e}")
            return False
        finally:
            cur.close()
            conn.close()

    def get_balance_history(self, days_back: int = 365) -> List[Dict]:
        """
        Get balance history for the last N days

        Args:
            days_back: Number of days to retrieve

        Returns:
            List of balance records
        """
        conn = self.get_connection()
        cur = conn.cursor(cursor_factory=RealDictCursor)

        try:
            cutoff_date = date.today() - timedelta(days=days_back)

            cur.execute("""
                SELECT
                    date,
                    starting_balance,
                    ending_balance,
                    daily_pl,
                    daily_pl_percent,
                    buying_power,
                    options_value,
                    stock_value,
                    cash_value,
                    total_positions,
                    notes
                FROM daily_portfolio_balances
                WHERE date >= %s
                ORDER BY date DESC
            """, (cutoff_date,))

            results = cur.fetchall()
            return [dict(row) for row in results]

        except Exception as e:
            logger.error(f"Error retrieving balance history: {e}")
            return []
        finally:
            cur.close()
            conn.close()

    def get_summary_stats(self, days_back: int = 30) -> Dict:
        """
        Get summary statistics for the specified period

        Args:
            days_back: Number of days to analyze

        Returns:
            Dictionary with summary stats
        """
        conn = self.get_connection()
        cur = conn.cursor(cursor_factory=RealDictCursor)

        try:
            cutoff_date = date.today() - timedelta(days=days_back)

            cur.execute("""
                SELECT
                    COUNT(*) as trading_days,
                    MIN(ending_balance) as min_balance,
                    MAX(ending_balance) as max_balance,
                    AVG(ending_balance) as avg_balance,
                    SUM(daily_pl) as total_pl,
                    AVG(daily_pl) as avg_daily_pl,
                    AVG(daily_pl_percent) as avg_daily_pl_percent,
                    MAX(daily_pl) as best_day,
                    MIN(daily_pl) as worst_day,
                    COUNT(CASE WHEN daily_pl > 0 THEN 1 END) as winning_days,
                    COUNT(CASE WHEN daily_pl < 0 THEN 1 END) as losing_days
                FROM daily_portfolio_balances
                WHERE date >= %s
            """, (cutoff_date,))

            result = cur.fetchone()

            if result:
                stats = dict(result)
                # Calculate win rate
                total_days = stats['winning_days'] + stats['losing_days']
                stats['win_rate'] = (stats['winning_days'] / total_days * 100) if total_days > 0 else 0

                # Get starting and ending balance for period
                cur.execute("""
                    SELECT ending_balance
                    FROM daily_portfolio_balances
                    WHERE date >= %s
                    ORDER BY date ASC
                    LIMIT 1
                """, (cutoff_date,))
                start = cur.fetchone()

                cur.execute("""
                    SELECT ending_balance
                    FROM daily_portfolio_balances
                    WHERE date >= %s
                    ORDER BY date DESC
                    LIMIT 1
                """, (cutoff_date,))
                end = cur.fetchone()

                if start and end:
                    stats['period_start_balance'] = float(start['ending_balance'])
                    stats['period_end_balance'] = float(end['ending_balance'])
                    stats['period_return'] = stats['period_end_balance'] - stats['period_start_balance']
                    stats['period_return_percent'] = (
                        stats['period_return'] / stats['period_start_balance'] * 100
                    ) if stats['period_start_balance'] > 0 else 0

                return stats

            return {}

        except Exception as e:
            logger.error(f"Error calculating summary stats: {e}")
            return {}
        finally:
            cur.close()
            conn.close()

    def get_latest_balance(self) -> Optional[Dict]:
        """Get the most recent balance record"""
        conn = self.get_connection()
        cur = conn.cursor(cursor_factory=RealDictCursor)

        try:
            cur.execute("""
                SELECT *
                FROM daily_portfolio_balances
                ORDER BY date DESC
                LIMIT 1
            """)

            result = cur.fetchone()
            return dict(result) if result else None

        except Exception as e:
            logger.error(f"Error retrieving latest balance: {e}")
            return None
        finally:
            cur.close()
            conn.close()

    def delete_balance(self, balance_date: date) -> bool:
        """Delete a balance record for a specific date"""
        conn = self.get_connection()
        cur = conn.cursor()

        try:
            cur.execute("""
                DELETE FROM daily_portfolio_balances
                WHERE date = %s
            """, (balance_date,))

            conn.commit()
            logger.info(f"Deleted balance for {balance_date}")
            return True

        except Exception as e:
            conn.rollback()
            logger.error(f"Error deleting balance: {e}")
            return False
        finally:
            cur.close()
            conn.close()

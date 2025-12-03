"""Auto Balance Recorder - Automatically tracks portfolio balance changes"""
import logging
from datetime import datetime
from typing import Optional, Dict
import psycopg2
from dotenv import load_dotenv
import os

load_dotenv(override=True)
logger = logging.getLogger(__name__)


class AutoBalanceRecorder:
    """Automatically records portfolio balance changes to database"""

    def __init__(self) -> None:
        self.db_config = {
            'host': os.getenv('DB_HOST', 'localhost'),
            'port': os.getenv('DB_PORT', '5432'),
            'user': os.getenv('DB_USER', 'postgres'),
            'password': os.getenv('DB_PASSWORD'),
            'database': os.getenv('DB_NAME', 'magnus')
        }

    def record_balance(
        self,
        total_balance: float,
        buying_power: float,
        total_equity: Optional[float] = None,
        options_buying_power: Optional[float] = None
    ) -> bool:
        """
        Record portfolio balance snapshot

        Args:
            total_balance: Total account balance
            buying_power: Available buying power
            total_equity: Total equity (optional)
            options_buying_power: Options buying power (optional)

        Returns:
            True if recorded successfully
        """
        try:
            conn = psycopg2.connect(**self.db_config)
            cur = conn.cursor()

            # Ensure table exists
            cur.execute("""
                CREATE TABLE IF NOT EXISTS portfolio_balance_history (
                    id SERIAL PRIMARY KEY,
                    recorded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    total_balance DECIMAL(15, 2),
                    buying_power DECIMAL(15, 2),
                    total_equity DECIMAL(15, 2),
                    options_buying_power DECIMAL(15, 2)
                )
            """)

            # Insert balance record
            cur.execute("""
                INSERT INTO portfolio_balance_history
                (total_balance, buying_power, total_equity, options_buying_power)
                VALUES (%s, %s, %s, %s)
            """, (total_balance, buying_power, total_equity, options_buying_power))

            conn.commit()
            cur.close()
            conn.close()

            logger.info(f"Recorded balance: ${total_balance:,.2f}")
            return True

        except Exception as e:
            logger.error(f"Error recording balance: {e}")
            return False

    def get_latest_balance(self) -> Optional[Dict]:
        """Get the most recent balance record"""
        try:
            conn = psycopg2.connect(**self.db_config)
            cur = conn.cursor()

            cur.execute("""
                SELECT total_balance, buying_power, total_equity,
                       options_buying_power, recorded_at
                FROM portfolio_balance_history
                ORDER BY recorded_at DESC
                LIMIT 1
            """)

            result = cur.fetchone()
            cur.close()
            conn.close()

            if result:
                return {
                    'total_balance': float(result[0]),
                    'buying_power': float(result[1]),
                    'total_equity': float(result[2]) if result[2] else None,
                    'options_buying_power': float(result[3]) if result[3] else None,
                    'recorded_at': result[4]
                }

            return None

        except Exception as e:
            logger.error(f"Error getting latest balance: {e}")
            return None

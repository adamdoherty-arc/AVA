"""
Dashboard Connector - Provides access to portfolio data

Handles:
- Portfolio balance (from Robinhood or database)
- Trade history
- P&L metrics
- Balance forecasts

Author: Backend Architect Agent
Date: January 10, 2025
"""

from typing import Dict, Any, List
from datetime import datetime
import os
from loguru import logger

from src.mfa.data_integration_service import DataConnector, DataSourceType, CachePolicy


class DashboardConnector(DataConnector):
    """
    Connector for Dashboard feature.

    Provides access to:
    - Portfolio balance (live from Robinhood or cached in DB)
    - Trade history with filters
    - P&L metrics (realized, unrealized, total)
    - Balance forecasts based on position expirations

    Cache Strategy:
    - Balance: 60 second TTL (changes frequently but not real-time)
    - Trade history: 5 minute TTL (historical data)
    - Summary: 60 second TTL
    """

    def __init__(self) -> None:
        # Dashboard data changes frequently but not in real-time
        # 60-second cache is appropriate
        cache_policy = CachePolicy(enabled=True, ttl_seconds=60, max_size=100)
        super().__init__(cache_policy)

        # Initialize dependencies (lazy loading to avoid circular imports)
        self._rh_client = None
        self._db_config = None

    def _get_robinhood_client(self) -> None:
        """Lazy load Robinhood client"""
        if self._rh_client is None:
            try:
                from src.services.robinhood_client import RobinhoodClient
                self._rh_client = RobinhoodClient()
            except Exception as e:
                logger.warning(f"Could not load Robinhood client: {e}")
        return self._rh_client

    def _get_db_config(self) -> Dict[str, str]:
        """Get database configuration from environment"""
        if self._db_config is None:
            self._db_config = {
                'host': os.getenv('DB_HOST', 'localhost'),
                'database': os.getenv('DB_NAME', 'trading'),
                'user': os.getenv('DB_USER', 'postgres'),
                'password': os.getenv('DB_PASSWORD', '')
            }
        return self._db_config

    def get_name(self) -> str:
        return "dashboard"

    def get_data_source_type(self) -> DataSourceType:
        return DataSourceType.HYBRID  # Database + Robinhood API

    def fetch_data(self, query: Dict[str, Any]) -> Dict[str, Any]:
        """
        Fetch dashboard data based on query type.

        Supported query types:
        - 'get_balance': Get current portfolio balance
        - 'get_summary': Get complete portfolio summary
        - 'get_trade_history': Get trade history with filters

        Args:
            query: Query dictionary with 'type' and 'params'

        Returns:
            Query results

        Raises:
            ValueError: If query type is unknown
        """
        query_type = query.get('type')
        params = query.get('params', {})

        if query_type == 'get_balance':
            return self._get_balance()
        elif query_type == 'get_summary':
            return self._get_summary()
        elif query_type == 'get_trade_history':
            return self._get_trade_history(params)
        else:
            raise ValueError(f"Unknown query type: {query_type}")

    def _get_balance(self) -> Dict[str, Any]:
        """
        Get current portfolio balance.

        Strategy:
        1. Try Robinhood API first (live balance)
        2. Fallback to database (last known balance)

        Returns:
            Dictionary with:
                - balance: Current balance (float)
                - source: 'robinhood' or 'database'
                - timestamp: When balance was retrieved
        """
        # Try Robinhood first (live data)
        rh = self._get_robinhood_client()
        if rh:
            try:
                # TODO: Implement actual Robinhood balance fetch
                # balance = rh.get_portfolio_balance()
                # return {
                #     'balance': balance,
                #     'source': 'robinhood',
                #     'timestamp': datetime.now().isoformat()
                # }
                logger.debug("Robinhood balance fetch not yet implemented")
            except Exception as e:
                logger.warning(f"Robinhood balance fetch failed: {e}")

        # Fallback to database (last known balance)
        try:
            import psycopg2
            conn = psycopg2.connect(**self._get_db_config())
            cur = conn.cursor()

            cur.execute("""
                SELECT balance, timestamp
                FROM portfolio_balances
                ORDER BY timestamp DESC
                LIMIT 1
            """)

            row = cur.fetchone()
            conn.close()

            if row:
                return {
                    'balance': float(row[0]),
                    'source': 'database',
                    'timestamp': row[1].isoformat() if row[1] else None
                }
        except Exception as e:
            logger.error(f"Database balance fetch failed: {e}")

        # Ultimate fallback
        return {
            'balance': 0.0,
            'source': 'unknown',
            'timestamp': None
        }

    def _get_summary(self) -> Dict[str, Any]:
        """
        Get complete portfolio summary.

        Returns:
            Dictionary with:
                - balance: Current balance
                - position_count: Number of active positions
                - realized_pnl: Realized P&L
                - unrealized_pnl: Unrealized P&L
                - total_pnl: Total P&L
                - theta_decay: Daily theta decay
                - timestamp: Data timestamp
        """
        try:
            import psycopg2
            conn = psycopg2.connect(**self._get_db_config())
            cur = conn.cursor()

            # Get current balance
            balance_data = self._get_balance()

            # Get position count
            cur.execute("SELECT COUNT(*) FROM positions WHERE status = 'active'")
            position_count = cur.fetchone()[0]

            # Get total P&L
            cur.execute("""
                SELECT
                    COALESCE(SUM(realized_pnl), 0) as realized,
                    COALESCE(SUM(unrealized_pnl), 0) as unrealized
                FROM positions
            """)
            pnl_row = cur.fetchone()

            # Get theta decay
            cur.execute("""
                SELECT COALESCE(SUM(theta), 0) as total_theta
                FROM positions
                WHERE status = 'active'
            """)
            theta_row = cur.fetchone()

            conn.close()

            return {
                'balance': balance_data['balance'],
                'position_count': position_count,
                'realized_pnl': float(pnl_row[0]) if pnl_row[0] else 0.0,
                'unrealized_pnl': float(pnl_row[1]) if pnl_row[1] else 0.0,
                'total_pnl': float(pnl_row[0] or 0.0) + float(pnl_row[1] or 0.0),
                'theta_decay': float(theta_row[0]) if theta_row[0] else 0.0,
                'timestamp': datetime.now().isoformat()
            }

        except Exception as e:
            logger.error(f"Error fetching portfolio summary: {e}")
            return {
                'balance': 0.0,
                'position_count': 0,
                'realized_pnl': 0.0,
                'unrealized_pnl': 0.0,
                'total_pnl': 0.0,
                'theta_decay': 0.0,
                'timestamp': datetime.now().isoformat(),
                'error': str(e)
            }

    def _get_trade_history(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Get trade history with optional filters.

        Args:
            params: Filter parameters
                - start_date: Filter trades after this date (ISO format)
                - end_date: Filter trades before this date (ISO format)
                - symbol: Filter by symbol
                - limit: Max number of trades

        Returns:
            Dictionary with:
                - trades: List of trade dictionaries
                - count: Number of trades returned
        """
        try:
            import psycopg2
            conn = psycopg2.connect(**self._get_db_config())
            cur = conn.cursor()

            # Build query with filters
            query = """
                SELECT
                    id, symbol, strategy, open_date, close_date,
                    premium_collected, cost_to_close, realized_pnl,
                    status
                FROM trade_history
                WHERE 1=1
            """

            query_params = []

            if params.get('start_date'):
                query += " AND open_date >= %s"
                query_params.append(params['start_date'])

            if params.get('end_date'):
                query += " AND open_date <= %s"
                query_params.append(params['end_date'])

            if params.get('symbol'):
                query += " AND symbol = %s"
                query_params.append(params['symbol'])

            query += " ORDER BY open_date DESC"

            if params.get('limit'):
                query += " LIMIT %s"
                query_params.append(params['limit'])

            cur.execute(query, query_params)
            rows = cur.fetchall()
            conn.close()

            trades = []
            for row in rows:
                trades.append({
                    'id': row[0],
                    'symbol': row[1],
                    'strategy': row[2],
                    'open_date': row[3].isoformat() if row[3] else None,
                    'close_date': row[4].isoformat() if row[4] else None,
                    'premium_collected': float(row[5]) if row[5] else 0.0,
                    'cost_to_close': float(row[6]) if row[6] else 0.0,
                    'realized_pnl': float(row[7]) if row[7] else 0.0,
                    'status': row[8]
                })

            return {
                'trades': trades,
                'count': len(trades)
            }

        except Exception as e:
            logger.error(f"Error fetching trade history: {e}")
            return {
                'trades': [],
                'count': 0,
                'error': str(e)
            }

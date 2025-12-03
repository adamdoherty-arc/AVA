"""
AVA Magnus Integration
======================

Integration with Magnus trading platform components:
- Robinhood portfolio and positions
- TradingView watchlists and alerts
- Xtrades followed traders and alerts

Usage:
    from src.ava.magnus_integration import MagnusIntegration

    magnus = MagnusIntegration()

    # Get portfolio data
    portfolio = await magnus.get_portfolio_summary()

    # Get positions
    positions = await magnus.get_options_positions()

    # Get CSP opportunities
    opportunities = await magnus.get_csp_opportunities()
"""

import os
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
from psycopg2.extras import RealDictCursor

from src.ava.db_manager import get_db_manager, DatabaseError
from src.services.robinhood_client import RobinhoodClient

logger = logging.getLogger(__name__)


class MagnusIntegration:
    """
    Integration with Magnus trading platform.

    Provides unified access to Robinhood, TradingView, and Xtrades data.
    """

    def __init__(self) -> None:
        """Initialize Magnus integration"""
        self.db = get_db_manager()
        self.rh_client = RobinhoodClient()
        logger.info("Magnus integration initialized")

    # ==================== Portfolio & Positions ====================

    async def get_portfolio_summary(self) -> Dict[str, Any]:
        """
        Get portfolio summary with latest balance and performance.

        Returns:
            Dictionary with portfolio data
        """
        try:
            with self.db.get_cursor(RealDictCursor) as cursor:
                # Get latest balance from daily_portfolio_balances
                cursor.execute("""
                    SELECT ending_balance as balance, date as timestamp, notes
                    FROM daily_portfolio_balances
                    ORDER BY date DESC
                    LIMIT 1
                """)
                latest = cursor.fetchone()

                if not latest:
                    return {
                        'balance': None,
                        'timestamp': None,
                        'daily_change': 0,
                        'daily_change_pct': 0,
                        'message': "No portfolio data available"
                    }

                # Get balance from yesterday
                cursor.execute("""
                    SELECT ending_balance as balance
                    FROM daily_portfolio_balances
                    WHERE date < %s
                    ORDER BY date DESC
                    LIMIT 1
                """, (latest['timestamp'],))

                previous = cursor.fetchone()

                daily_change = 0
                daily_change_pct = 0

                if previous:
                    daily_change = latest['balance'] - previous['balance']
                    if previous['balance'] > 0:
                        daily_change_pct = (daily_change / previous['balance']) * 100

                return {
                    'balance': float(latest['balance']),
                    'timestamp': latest['timestamp'],
                    'notes': latest['notes'],
                    'daily_change': float(daily_change),
                    'daily_change_pct': float(daily_change_pct),
                }

        except DatabaseError as e:
            logger.error(f"Database error getting portfolio summary: {e}")
            raise
        except Exception as e:
            logger.error(f"Error getting portfolio summary: {e}")
            raise DatabaseError("Failed to retrieve portfolio summary")

    async def get_options_positions(self, active_only: bool = True) -> List[Dict[str, Any]]:
        """
        Get current options positions from Robinhood.

        Args:
            active_only: If True, only return open positions

        Returns:
            List of position dictionaries
        """
        try:
            # Get positions from Robinhood API (live data)
            positions = self.rh_client.get_options_positions()

            # Filter by status if needed
            if active_only:
                positions = [p for p in positions if p.get('quantity', 0) != 0]

            return positions

        except Exception as e:
            logger.error(f"Error getting options positions from Robinhood: {e}")
            raise DatabaseError("Failed to retrieve options positions")

    async def get_csp_opportunities(
        self,
        min_premium: Optional[float] = None,
        max_dte: Optional[int] = 45,
        limit: int = 20
    ) -> List[Dict[str, Any]]:
        """
        Get cash-secured put opportunities.

        Args:
            min_premium: Minimum premium (optional)
            max_dte: Maximum days to expiration
            limit: Maximum number of results

        Returns:
            List of CSP opportunity dictionaries
        """
        try:
            with self.db.get_cursor(RealDictCursor) as cursor:
                # Use covered_call_opportunities table
                query = """
                    SELECT
                        symbol as ticker,
                        strike_price,
                        expiration_date,
                        premium,
                        implied_volatility,
                        delta,
                        open_interest,
                        volume,
                        annualized_return as annual_return,
                        confidence_score as score,
                        last_updated as updated_at
                    FROM covered_call_opportunities
                    WHERE 1=1
                """
                params = []

                if min_premium:
                    query += " AND premium >= %s"
                    params.append(min_premium)

                if max_dte:
                    query += " AND days_to_expiry <= %s"
                    params.append(max_dte)

                query += """
                    ORDER BY confidence_score DESC, annualized_return DESC
                    LIMIT %s
                """
                params.append(limit)

                cursor.execute(query, tuple(params))
                return cursor.fetchall()

        except DatabaseError:
            raise
        except Exception as e:
            logger.error(f"Error getting CSP opportunities: {e}")
            raise DatabaseError("Failed to retrieve CSP opportunities")

    # ==================== TradingView Integration ====================

    async def get_tradingview_watchlists(self) -> List[Dict[str, Any]]:
        """
        Get TradingView watchlists.

        Returns:
            List of watchlist dictionaries
        """
        try:
            with self.db.get_cursor(RealDictCursor) as cursor:
                cursor.execute("""
                    SELECT
                        id,
                        name,
                        symbols,
                        created_at,
                        updated_at
                    FROM tradingview_watchlists
                    ORDER BY name
                """)
                return cursor.fetchall()

        except DatabaseError:
            raise
        except Exception as e:
            logger.error(f"Error getting TradingView watchlists: {e}")
            raise DatabaseError("Failed to retrieve TradingView watchlists")

    async def get_tradingview_alerts(self, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Get recent TradingView alerts.

        Args:
            limit: Maximum number of alerts to return

        Returns:
            List of alert dictionaries
        """
        try:
            with self.db.get_cursor(RealDictCursor) as cursor:
                cursor.execute("""
                    SELECT
                        id,
                        ticker,
                        alert_type,
                        message,
                        price,
                        triggered_at,
                        created_at
                    FROM tradingview_alerts
                    ORDER BY triggered_at DESC
                    LIMIT $1
                """, (limit,))
                return cursor.fetchall()

        except DatabaseError:
            raise
        except Exception as e:
            logger.error(f"Error getting TradingView alerts: {e}")
            raise DatabaseError("Failed to retrieve TradingView alerts")

    # ==================== Xtrades Integration ====================

    async def get_following_traders(self) -> List[Dict[str, Any]]:
        """
        Get list of traders being followed on Xtrades.

        Returns:
            List of trader dictionaries
        """
        try:
            with self.db.get_cursor(RealDictCursor) as cursor:
                cursor.execute("""
                    SELECT
                        id,
                        username,
                        display_name,
                        active as alerts_enabled,
                        last_sync as last_alert_at,
                        added_date as created_at,
                        last_sync as updated_at
                    FROM xtrades_profiles
                    WHERE active = true
                    ORDER BY username
                """)
                return cursor.fetchall()

        except DatabaseError:
            raise
        except Exception as e:
            logger.error(f"Error getting following traders: {e}")
            raise DatabaseError("Failed to retrieve followed traders")

    async def get_xtrades_alerts(
        self,
        trader_username: Optional[str] = None,
        limit: int = 20
    ) -> List[Dict[str, Any]]:
        """
        Get recent Xtrades alerts.

        Args:
            trader_username: Filter by specific trader (optional)
            limit: Maximum number of alerts to return

        Returns:
            List of alert dictionaries
        """
        try:
            with self.db.get_cursor(RealDictCursor) as cursor:
                # Join with trades to get alert details
                query = """
                    SELECT
                        a.id,
                        t.ticker,
                        a.strategy_name as alert_type,
                        t.action,
                        t.strike_price,
                        t.expiration_date,
                        t.contracts as quantity,
                        t.premium,
                        t.alert_time,
                        a.evaluated_at as created_at,
                        a.recommendation,
                        a.consensus_score
                    FROM xtrades_alerts a
                    JOIN xtrades_trades t ON a.trade_id = t.id
                    WHERE 1=1
                """
                params = []

                if trader_username:
                    query += """
                        AND t.profile_id IN (
                            SELECT id FROM xtrades_profiles WHERE username = %s
                        )
                    """
                    params.append(trader_username)

                query += """
                    ORDER BY a.evaluated_at DESC
                    LIMIT %s
                """
                params.append(limit)

                cursor.execute(query, tuple(params) if params else (limit,))
                return cursor.fetchall()

        except DatabaseError:
            raise
        except Exception as e:
            logger.error(f"Error getting Xtrades alerts: {e}")
            raise DatabaseError("Failed to retrieve Xtrades alerts")

    async def toggle_trader_alerts(self, trader_id: int, enabled: bool) -> bool:
        """
        Enable or disable alerts for a specific trader.

        Args:
            trader_id: Trader ID in database
            enabled: True to enable, False to disable

        Returns:
            True if successful
        """
        try:
            with self.db.get_cursor() as cursor:
                cursor.execute("""
                    UPDATE xtrades_profiles
                    SET active = %s
                    WHERE id = %s
                """, (enabled, trader_id))

                return cursor.rowcount > 0

        except DatabaseError:
            raise
        except Exception as e:
            logger.error(f"Error toggling trader alerts: {e}")
            raise DatabaseError("Failed to update trader alerts")

    # ==================== Balance History ====================

    async def get_balance_history(self, days: int = 30) -> List[Dict[str, Any]]:
        """
        Get portfolio balance history.

        Args:
            days: Number of days of history

        Returns:
            List of balance records
        """
        try:
            with self.db.get_cursor(RealDictCursor) as cursor:
                cursor.execute("""
                    SELECT
                        ending_balance as balance,
                        date as timestamp,
                        notes
                    FROM daily_portfolio_balances
                    WHERE date >= %s
                    ORDER BY date ASC
                """, (datetime.now() - timedelta(days=days),))

                return cursor.fetchall()

        except DatabaseError:
            raise
        except Exception as e:
            logger.error(f"Error getting balance history: {e}")
            raise DatabaseError("Failed to retrieve balance history")

    # ==================== Tasks & CI ====================

    async def get_active_tasks(self) -> List[Dict[str, Any]]:
        """
        Get active CI/enhancement tasks.

        Returns:
            List of task dictionaries
        """
        try:
            with self.db.get_cursor(RealDictCursor) as cursor:
                cursor.execute("""
                    SELECT
                        id,
                        title,
                        description,
                        status,
                        priority,
                        created_at,
                        updated_at,
                        completed_at
                    FROM ci_enhancements
                    WHERE status IN ('proposed', 'in_progress')
                    ORDER BY priority DESC, created_at DESC
                """)
                return cursor.fetchall()

        except DatabaseError:
            raise
        except Exception as e:
            logger.error(f"Error getting active tasks: {e}")
            raise DatabaseError("Failed to retrieve active tasks")

    async def get_completed_tasks(self, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Get recently completed tasks.

        Args:
            limit: Maximum number of tasks to return

        Returns:
            List of task dictionaries
        """
        try:
            with self.db.get_cursor(RealDictCursor) as cursor:
                cursor.execute("""
                    SELECT
                        id,
                        title,
                        description,
                        status,
                        completed_at
                    FROM ci_enhancements
                    WHERE status = 'completed'
                    ORDER BY completed_at DESC
                    LIMIT $1
                """, (limit,))
                return cursor.fetchall()

        except DatabaseError:
            raise
        except Exception as e:
            logger.error(f"Error getting completed tasks: {e}")
            raise DatabaseError("Failed to retrieve completed tasks")

    async def get_task_stats(self) -> Dict[str, Any]:
        """
        Get task statistics.

        Returns:
            Dictionary with task counts and statistics
        """
        try:
            with self.db.get_cursor(RealDictCursor) as cursor:
                cursor.execute("""
                    SELECT
                        COUNT(*) FILTER (WHERE status = 'proposed') as proposed,
                        COUNT(*) FILTER (WHERE status = 'in_progress') as in_progress,
                        COUNT(*) FILTER (WHERE status = 'completed') as completed,
                        COUNT(*) FILTER (WHERE status = 'completed' AND completed_at >= CURRENT_DATE) as completed_today,
                        COUNT(*) FILTER (WHERE status = 'completed' AND completed_at >= CURRENT_DATE - INTERVAL '7 days') as completed_this_week
                    FROM ci_enhancements
                """)
                return cursor.fetchone()

        except DatabaseError:
            raise
        except Exception as e:
            logger.error(f"Error getting task stats: {e}")
            raise DatabaseError("Failed to retrieve task statistics")


# Example usage
if __name__ == "__main__":
    import asyncio
    import sys

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    async def test_integration():
        """Test Magnus integration"""
        print("Testing Magnus Integration...\n")

        magnus = MagnusIntegration()

        try:
            # Test portfolio summary
            print("Test 1: Portfolio summary")
            portfolio = await magnus.get_portfolio_summary()
            print(f"Balance: ${portfolio.get('balance', 0):,.2f}")
            print(f"Daily change: ${portfolio.get('daily_change', 0):,.2f} ({portfolio.get('daily_change_pct', 0):.2f}%)")

            # Test positions
            print("\nTest 2: Options positions")
            positions = await magnus.get_options_positions()
            print(f"Found {len(positions)} active positions")

            # Test CSP opportunities
            print("\nTest 3: CSP opportunities")
            opportunities = await magnus.get_csp_opportunities(limit=5)
            print(f"Found {len(opportunities)} CSP opportunities")

            # Test task stats
            print("\nTest 4: Task statistics")
            stats = await magnus.get_task_stats()
            print(f"Proposed: {stats.get('proposed', 0)}")
            print(f"In Progress: {stats.get('in_progress', 0)}")
            print(f"Completed: {stats.get('completed', 0)}")

            print("\n✅ All integration tests passed!")

        except Exception as e:
            print(f"\n❌ Test failed: {e}")
            import traceback
            traceback.print_exc()
            sys.exit(1)

    # Run tests
    asyncio.run(test_integration())

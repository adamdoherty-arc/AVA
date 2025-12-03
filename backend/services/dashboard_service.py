"""Dashboard Service - Aggregates portfolio and market data"""

from typing import Dict, List, Optional
import logging
from datetime import datetime, timedelta
from psycopg2.extras import RealDictCursor
from backend.database.connection import db_pool
from backend.services.portfolio_service import get_portfolio_service

logger = logging.getLogger(__name__)


class DashboardService:
    """Service for dashboard data aggregation"""

    def __init__(self) -> None:
        self._portfolio = get_portfolio_service()

    def get_portfolio_summary(self) -> Dict:
        """Get portfolio summary metrics from Robinhood"""
        try:
            positions_data = self._portfolio.get_positions()

            if positions_data:
                summary = positions_data.get('summary', {})
                stocks = positions_data.get('stocks', [])
                options = positions_data.get('options', [])

                total_equity = summary.get('total_equity', 0)
                buying_power = summary.get('buying_power', 0)

                # Calculate allocations
                stocks_value = sum(s.get('current_value', 0) for s in stocks)
                options_value = sum(o.get('current_value', 0) for o in options)

                if total_equity > 0:
                    allocations = {
                        'stocks': round((stocks_value / total_equity) * 100, 1),
                        'options': round((options_value / total_equity) * 100, 1),
                        'cash': round((buying_power / total_equity) * 100, 1)
                    }
                else:
                    allocations = {'stocks': 0, 'options': 0, 'cash': 100}

                # Calculate day change
                day_change = sum(s.get('pl', 0) for s in stocks) + sum(o.get('pl', 0) for o in options)

                return {
                    'total_value': total_equity,
                    'buying_power': buying_power,
                    'day_change': day_change,
                    'day_change_pct': round((day_change / total_equity * 100) if total_equity > 0 else 0, 2),
                    'allocations': allocations,
                    'positions_count': len(stocks) + len(options),
                    'last_updated': datetime.now().isoformat()
                }
        except Exception as e:
            logger.error(f"Error getting portfolio summary: {e}")

        # Return placeholder if error or not connected
        return {
            'total_value': 0.0,
            'buying_power': 0.0,
            'day_change': 0.0,
            'day_change_pct': 0.0,
            'allocations': {'stocks': 0, 'options': 0, 'cash': 100},
            'positions_count': 0,
            'last_updated': datetime.now().isoformat()
        }

    def get_recent_activity(self, limit: int = 10) -> List[Dict]:
        """Get recent trading activity from database"""
        try:
            with db_pool.get_connection() as conn:
                with conn.cursor(cursor_factory=RealDictCursor) as cur:
                    cur.execute("""
                        SELECT
                            symbol, action, quantity, price,
                            total_value, executed_at, order_type
                        FROM trade_history
                        ORDER BY executed_at DESC
                        LIMIT %s
                    """, (limit,))
                    return [dict(row) for row in cur.fetchall()]
        except Exception as e:
            logger.warning(f"Error getting recent activity: {e}")
            return []

    def get_performance_history(self, period: str = "1M") -> List[Dict]:
        """Get historical performance data for charts"""
        period_days = {'1W': 7, '1M': 30, '3M': 90, '6M': 180, '1Y': 365, 'ALL': 3650}
        days = period_days.get(period, 30)

        try:
            with db_pool.get_connection() as conn:
                with conn.cursor(cursor_factory=RealDictCursor) as cur:
                    cur.execute("""
                        SELECT
                            date, portfolio_value, day_change, day_change_pct
                        FROM portfolio_history
                        WHERE date >= CURRENT_DATE - INTERVAL '%s days'
                        ORDER BY date ASC
                    """, (days,))
                    return [dict(row) for row in cur.fetchall()]
        except Exception as e:
            logger.warning(f"Error getting performance history: {e}")
            return []

    def get_alerts(self, limit: int = 20) -> List[Dict]:
        """Get active alerts"""
        try:
            with db_pool.get_connection() as conn:
                with conn.cursor(cursor_factory=RealDictCursor) as cur:
                    cur.execute("""
                        SELECT
                            id, alert_type, symbol, message,
                            priority, created_at, is_read
                        FROM alerts
                        WHERE is_dismissed = FALSE
                        ORDER BY created_at DESC
                        LIMIT %s
                    """, (limit,))
                    return [dict(row) for row in cur.fetchall()]
        except Exception as e:
            logger.warning(f"Error getting alerts: {e}")
            return []

    def get_dashboard_stats(self) -> Dict:
        """Get comprehensive dashboard statistics"""
        portfolio = self.get_portfolio_summary()
        activity = self.get_recent_activity(5)
        alerts = self.get_alerts(10)

        return {
            'portfolio': portfolio,
            'recent_activity': activity,
            'alerts': alerts,
            'quick_stats': {
                'active_csps': 0,
                'active_ccs': 0,
                'pending_assignments': 0,
                'expiring_this_week': 0
            }
        }


# Singleton instance
_dashboard_service: Optional[DashboardService] = None

def get_dashboard_service() -> DashboardService:
    global _dashboard_service
    if _dashboard_service is None:
        _dashboard_service = DashboardService()
    return _dashboard_service

dashboard_service = DashboardService()

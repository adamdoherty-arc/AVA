"""
Dashboard Service - Aggregates portfolio and market data

OPTIMIZATIONS APPLIED:
1. Redis/In-Memory caching with stampede protection
2. Async database queries where possible
3. Parallel data fetching with asyncio.gather()
4. Configurable cache TTLs
"""

from typing import Dict, List, Optional
import logging
import asyncio
from datetime import datetime
from psycopg2.extras import RealDictCursor
from concurrent.futures import ThreadPoolExecutor
from backend.database.connection import db_pool
from backend.services.portfolio_service import get_portfolio_service
from backend.infrastructure.cache import get_cache

logger = logging.getLogger(__name__)

# Thread pool for sync database calls
_db_executor = ThreadPoolExecutor(max_workers=5, thread_name_prefix="db_")


class DashboardService:
    """
    Service for dashboard data aggregation with caching.

    Performance Improvements:
    - Portfolio summary cached for 60 seconds
    - Recent activity cached for 30 seconds
    - Performance history cached for 5 minutes
    - Alerts cached for 15 seconds
    - Parallel fetching for dashboard stats
    """

    # Cache TTLs (in seconds)
    CACHE_TTL_PORTFOLIO = 60
    CACHE_TTL_ACTIVITY = 30
    CACHE_TTL_HISTORY = 300  # 5 minutes
    CACHE_TTL_ALERTS = 15

    def __init__(self):
        self._portfolio = get_portfolio_service()
        self._cache = get_cache()

    async def get_portfolio_summary(self) -> Dict:
        """Get portfolio summary metrics from Robinhood (cached)."""
        cache_key = "dashboard:portfolio_summary"

        # Use stampede-protected cache
        return await self._cache.get_or_fetch(
            cache_key,
            self._fetch_portfolio_summary,
            ttl=self.CACHE_TTL_PORTFOLIO
        )

    async def _fetch_portfolio_summary(self) -> Dict:
        """Internal method to fetch portfolio summary."""
        try:
            positions_data = await self._portfolio.get_positions()

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

    async def get_recent_activity(self, limit: int = 10) -> List[Dict]:
        """Get recent trading activity from database (cached)."""
        cache_key = f"dashboard:activity:{limit}"

        return await self._cache.get_or_fetch(
            cache_key,
            lambda: self._fetch_recent_activity(limit),
            ttl=self.CACHE_TTL_ACTIVITY
        )

    async def _fetch_recent_activity(self, limit: int) -> List[Dict]:
        """Internal method to fetch recent activity (async wrapper)."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(_db_executor, self._fetch_recent_activity_sync, limit)

    def _fetch_recent_activity_sync(self, limit: int) -> List[Dict]:
        """Sync database query for recent activity."""
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

    async def get_performance_history(self, period: str = "1M") -> List[Dict]:
        """Get historical performance data for charts (cached)."""
        cache_key = f"dashboard:history:{period}"

        return await self._cache.get_or_fetch(
            cache_key,
            lambda: self._fetch_performance_history(period),
            ttl=self.CACHE_TTL_HISTORY
        )

    async def _fetch_performance_history(self, period: str) -> List[Dict]:
        """Internal method to fetch performance history (async wrapper)."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(_db_executor, self._fetch_performance_history_sync, period)

    def _fetch_performance_history_sync(self, period: str) -> List[Dict]:
        """Sync database query for performance history."""
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

    async def get_alerts(self, limit: int = 20) -> List[Dict]:
        """Get active alerts (cached)."""
        cache_key = f"dashboard:alerts:{limit}"

        return await self._cache.get_or_fetch(
            cache_key,
            lambda: self._fetch_alerts(limit),
            ttl=self.CACHE_TTL_ALERTS
        )

    async def _fetch_alerts(self, limit: int) -> List[Dict]:
        """Internal method to fetch alerts (async wrapper)."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(_db_executor, self._fetch_alerts_sync, limit)

    def _fetch_alerts_sync(self, limit: int) -> List[Dict]:
        """Sync database query for alerts."""
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

    async def get_dashboard_stats(self) -> Dict:
        """
        Get comprehensive dashboard statistics.

        Optimization: Fetches all data in PARALLEL using asyncio.gather().
        Performance: 4 sequential calls -> 4 parallel calls (3-4x faster)
        """
        # Fetch all dashboard data in parallel
        portfolio, activity, alerts = await asyncio.gather(
            self.get_portfolio_summary(),
            self.get_recent_activity(5),
            self.get_alerts(10)
        )

        # Calculate quick stats from portfolio data
        quick_stats = self._calculate_quick_stats(portfolio)

        return {
            'portfolio': portfolio,
            'recent_activity': activity,
            'alerts': alerts,
            'quick_stats': quick_stats
        }

    def _calculate_quick_stats(self, portfolio: Dict) -> Dict:
        """Calculate quick stats (could be enhanced with position analysis)."""
        # TODO: Enhance with actual position analysis
        return {
            'active_csps': 0,
            'active_ccs': 0,
            'pending_assignments': 0,
            'expiring_this_week': 0
        }

    async def invalidate_cache(self, pattern: str = "dashboard:*"):
        """Invalidate dashboard caches."""
        await self._cache.invalidate_pattern(pattern)
        logger.info(f"Dashboard cache invalidated: {pattern}")


# Singleton instance
_dashboard_service: Optional[DashboardService] = None

def get_dashboard_service() -> DashboardService:
    global _dashboard_service
    if _dashboard_service is None:
        _dashboard_service = DashboardService()
    return _dashboard_service

dashboard_service = DashboardService()

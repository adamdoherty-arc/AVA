"""
Magnus Financial Assistant - Unified Data Integration Service

This service provides THE SINGLE INTERFACE for MFA to access ALL Magnus features and data.

Architecture:
- Connector pattern: One connector per Magnus feature
- Caching: Built-in intelligent caching with TTL
- Thread-safe: All operations safe for concurrent access
- Observable: All operations logged for monitoring

Author: Backend Architect Agent
Date: January 10, 2025
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum
import threading
import hashlib
import json
from loguru import logger


# ============================================================================
# ENUMS AND DATA CLASSES
# ============================================================================

class DataSourceType(Enum):
    """Types of data sources"""
    DATABASE = "database"          # PostgreSQL direct
    EXTERNAL_API = "external_api"  # Robinhood, Kalshi, etc.
    CACHE = "cache"                # Redis cache
    COMPUTED = "computed"          # Calculated on demand
    HYBRID = "hybrid"              # Multiple sources


@dataclass
class CachePolicy:
    """Cache policy configuration"""
    enabled: bool = True
    ttl_seconds: int = 300  # 5 minutes default
    max_size: int = 1000    # Max items in cache

    @staticmethod
    def no_cache() -> 'CachePolicy':
        """Create a no-cache policy"""
        return CachePolicy(enabled=False, ttl_seconds=0, max_size=0)

    @staticmethod
    def short_ttl() -> 'CachePolicy':
        """30 second cache (for frequently changing data)"""
        return CachePolicy(enabled=True, ttl_seconds=30, max_size=100)

    @staticmethod
    def medium_ttl() -> 'CachePolicy':
        """5 minute cache (default)"""
        return CachePolicy(enabled=True, ttl_seconds=300, max_size=1000)

    @staticmethod
    def long_ttl() -> 'CachePolicy':
        """1 hour cache (for slow-changing data)"""
        return CachePolicy(enabled=True, ttl_seconds=3600, max_size=500)

    @staticmethod
    def daily() -> 'CachePolicy':
        """24 hour cache (for static data)"""
        return CachePolicy(enabled=True, ttl_seconds=86400, max_size=100)


# ============================================================================
# BASE CONNECTOR CLASS
# ============================================================================

class DataConnector(ABC):
    """
    Abstract base class for all data connectors.

    Each Magnus feature has a corresponding connector that implements
    this interface for unified access.
    """

    def __init__(self, cache_policy: Optional[CachePolicy] = None):
        self.cache_policy = cache_policy or CachePolicy.medium_ttl()
        self._cache: Dict[str, Any] = {}
        self._cache_timestamps: Dict[str, datetime] = {}
        self._cache_hits = 0
        self._cache_misses = 0
        self._lock = threading.Lock()

        logger.info(f"Initialized {self.get_name()} connector with cache TTL={self.cache_policy.ttl_seconds}s")

    @abstractmethod
    def get_name(self) -> str:
        """Return connector name (e.g., 'dashboard', 'positions')"""
        pass

    @abstractmethod
    def get_data_source_type(self) -> DataSourceType:
        """Return the primary data source type"""
        pass

    @abstractmethod
    def fetch_data(self, query: Dict[str, Any]) -> Dict[str, Any]:
        """
        Fetch data based on query parameters.

        Args:
            query: Dictionary with query parameters
                   Format: {"type": "query_type", "params": {...}}

        Returns:
            Dictionary with results
        """
        pass

    def get_cached_or_fetch(self, cache_key: str, query: Dict[str, Any]) -> Dict[str, Any]:
        """
        Get data from cache if valid, otherwise fetch fresh.

        Args:
            cache_key: Unique key for this query
            query: Query parameters

        Returns:
            Cached or fresh data
        """
        if not self.cache_policy.enabled:
            self._cache_misses += 1
            return self.fetch_data(query)

        with self._lock:
            # Check cache validity
            if cache_key in self._cache:
                timestamp = self._cache_timestamps.get(cache_key)
                if timestamp:
                    age = (datetime.now() - timestamp).total_seconds()
                    if age < self.cache_policy.ttl_seconds:
                        self._cache_hits += 1
                        logger.debug(f"{self.get_name()}: Cache HIT for key={cache_key}")
                        return self._cache[cache_key]

            # Cache miss - fetch fresh data
            self._cache_misses += 1
            logger.debug(f"{self.get_name()}: Cache MISS for key={cache_key}")

            try:
                data = self.fetch_data(query)

                # Update cache
                self._cache[cache_key] = data
                self._cache_timestamps[cache_key] = datetime.now()

                # Enforce max cache size (LRU-like)
                if len(self._cache) > self.cache_policy.max_size:
                    oldest_key = min(self._cache_timestamps.items(), key=lambda x: x[1])[0]
                    del self._cache[oldest_key]
                    del self._cache_timestamps[oldest_key]
                    logger.debug(f"{self.get_name()}: Evicted oldest cache entry")

                return data

            except Exception as e:
                logger.error(f"{self.get_name()}: Error fetching data: {e}")
                raise

    def invalidate_cache(self, cache_key: Optional[str] = None):
        """
        Invalidate cache (specific key or all).

        Args:
            cache_key: Specific key to invalidate, or None for all
        """
        with self._lock:
            if cache_key:
                self._cache.pop(cache_key, None)
                self._cache_timestamps.pop(cache_key, None)
                logger.debug(f"{self.get_name()}: Invalidated cache key={cache_key}")
            else:
                self._cache.clear()
                self._cache_timestamps.clear()
                logger.info(f"{self.get_name()}: Cleared all cache")

    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache performance statistics"""
        total_requests = self._cache_hits + self._cache_misses
        hit_rate = (self._cache_hits / total_requests * 100) if total_requests > 0 else 0

        return {
            'connector': self.get_name(),
            'cache_size': len(self._cache),
            'cache_hits': self._cache_hits,
            'cache_misses': self._cache_misses,
            'hit_rate_pct': round(hit_rate, 2),
            'ttl_seconds': self.cache_policy.ttl_seconds
        }


# ============================================================================
# MAIN DATA INTEGRATION SERVICE
# ============================================================================

class DataIntegrationService:
    """
    Central service for unified data access across ALL Magnus features.

    This is the SINGLE interface that MFA uses to access any data
    from any Magnus feature.

    Features:
    - 21+ connectors to all Magnus features
    - Intelligent caching per data type
    - Thread-safe concurrent access
    - Cross-feature intelligence
    - Performance monitoring

    Usage:
        service = get_data_service()  # Singleton

        # Simple queries
        balance = service.get_portfolio_balance()
        positions = service.get_active_positions()

        # Complex cross-feature queries
        risky = service.check_position_earnings_risk()
        watchlist_opps = service.find_watchlist_opportunities("main")
    """

    def __init__(self) -> None:
        self._connectors: Dict[str, DataConnector] = {}
        self._initialized = False
        self._lock = threading.Lock()

        logger.info("Initializing DataIntegrationService...")

    def initialize(self) -> None:
        """
        Initialize all connectors (lazy initialization).

        This is called on first use to avoid circular imports
        and to allow connectors to be registered dynamically.
        """
        if self._initialized:
            return

        with self._lock:
            if self._initialized:  # Double-check after lock
                return

            logger.info("Loading all feature connectors...")
            self._register_all_connectors()
            self._initialized = True
            logger.info(f"DataIntegrationService initialized with {len(self._connectors)} connectors")

    def _register_all_connectors(self) -> None:
        """
        Register all feature connectors.

        NOTE: Actual connector implementations would be imported here.
        For now, this is a stub showing the architecture.
        """
        # Import and register each connector
        # Example (when implemented):
        # from src.mfa.connectors.dashboard_connector import DashboardConnector
        # self.register_connector(DashboardConnector())

        # Stub: Register placeholder connectors
        logger.warning("Using placeholder connectors - implement actual connectors in src/mfa/connectors/")

        # TODO: Implement these connectors
        # self.register_connector(DashboardConnector())
        # self.register_connector(PositionsConnector())
        # self.register_connector(OpportunitiesConnector())
        # self.register_connector(TradingViewConnector())
        # self.register_connector(KalshiConnector())
        # self.register_connector(XtradesConnector())
        # self.register_connector(AIResearchConnector())
        # self.register_connector(EarningsConnector())
        # self.register_connector(CalendarSpreadsConnector())
        # self.register_connector(DatabaseScanConnector())
        # ... etc for all 21 features

    def register_connector(self, connector: DataConnector):
        """
        Register a data connector.

        Args:
            connector: DataConnector instance
        """
        name = connector.get_name()
        self._connectors[name] = connector
        logger.info(f"Registered connector: {name}")

    def get_connector(self, name: str) -> Optional[DataConnector]:
        """
        Get connector by name.

        Args:
            name: Connector name (e.g., 'dashboard', 'positions')

        Returns:
            DataConnector instance or None if not found
        """
        self.initialize()  # Ensure initialized
        return self._connectors.get(name)

    def list_connectors(self) -> List[str]:
        """Get list of all registered connector names"""
        self.initialize()
        return list(self._connectors.keys())

    # ========================================================================
    # UNIVERSAL QUERY INTERFACE
    # ========================================================================

    def query(self, feature: str, query_type: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Universal query interface for any Magnus feature.

        Args:
            feature: Feature name (e.g., 'dashboard', 'positions')
            query_type: Type of query (e.g., 'get_balance', 'fetch_positions')
            params: Query parameters (optional)

        Returns:
            Query results as dictionary

        Raises:
            ValueError: If feature not found
            Exception: If query fails

        Examples:
            service.query('dashboard', 'get_balance')
            service.query('positions', 'fetch_active', {'include_closed': False})
            service.query('opportunities', 'scan_csp', {'min_score': 80})
        """
        self.initialize()

        connector = self.get_connector(feature)
        if not connector:
            raise ValueError(f"Unknown feature: {feature}. Available: {self.list_connectors()}")

        # Generate cache key from query
        cache_key = self._generate_cache_key(feature, query_type, params)

        # Build query dict
        query_dict = {
            "type": query_type,
            "params": params or {}
        }

        # Execute query with caching
        return connector.get_cached_or_fetch(cache_key, query_dict)

    def _generate_cache_key(self, feature: str, query_type: str, params: Optional[Dict[str, Any]]) -> str:
        """
        Generate unique cache key from query components.

        Args:
            feature: Feature name
            query_type: Query type
            params: Query parameters

        Returns:
            Unique cache key string
        """
        # Serialize params to JSON for consistent hashing
        params_str = json.dumps(params or {}, sort_keys=True)
        params_hash = hashlib.md5(params_str.encode()).hexdigest()[:8]

        return f"{feature}:{query_type}:{params_hash}"

    # ========================================================================
    # CONVENIENCE METHODS - DASHBOARD
    # ========================================================================

    def get_portfolio_balance(self) -> float:
        """
        Get current portfolio balance.

        Returns:
            Current balance as float

        Example:
            balance = service.get_portfolio_balance()
            # Returns: 52341.67
        """
        result = self.query('dashboard', 'get_balance')
        return result.get('balance', 0.0)

    def get_portfolio_summary(self) -> Dict[str, Any]:
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

        Example:
            summary = service.get_portfolio_summary()
            # Returns: {
            #   'balance': 52341.67,
            #   'position_count': 5,
            #   'total_pnl': 342.50,
            #   ...
            # }
        """
        return self.query('dashboard', 'get_summary')

    def get_trade_history(
        self,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        symbol: Optional[str] = None,
        limit: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Get trade history with optional filters.

        Args:
            start_date: Filter trades after this date (ISO format)
            end_date: Filter trades before this date (ISO format)
            symbol: Filter by symbol
            limit: Max number of trades to return

        Returns:
            List of trade dictionaries

        Example:
            trades = service.get_trade_history(
                start_date="2024-12-01",
                limit=50
            )
        """
        params = {}
        if start_date:
            params['start_date'] = start_date
        if end_date:
            params['end_date'] = end_date
        if symbol:
            params['symbol'] = symbol
        if limit:
            params['limit'] = limit

        result = self.query('dashboard', 'get_trade_history', params)
        return result.get('trades', [])

    # ========================================================================
    # CONVENIENCE METHODS - POSITIONS
    # ========================================================================

    def get_active_positions(self, include_closed: bool = False) -> List[Dict[str, Any]]:
        """
        Get all active positions.

        Args:
            include_closed: Include closed positions (default: False)

        Returns:
            List of position dictionaries

        Example:
            positions = service.get_active_positions()
            # Returns: [
            #   {
            #     'symbol': 'AAPL',
            #     'strike': 170.0,
            #     'expiration': '2025-01-19',
            #     'profit_pct': 30.5,
            #     ...
            #   },
            #   ...
            # ]
        """
        params = {'include_closed': include_closed}
        result = self.query('positions', 'fetch_active', params)
        return result.get('positions', [])

    def get_position_details(self, position_id: str) -> Dict[str, Any]:
        """
        Get detailed information for a specific position.

        Args:
            position_id: Position ID

        Returns:
            Position details dictionary
        """
        return self.query('positions', 'get_details', {'position_id': position_id})

    def calculate_portfolio_theta(self) -> float:
        """
        Calculate total theta decay across all positions.

        Returns:
            Total daily theta decay

        Example:
            theta = service.calculate_portfolio_theta()
            # Returns: 18.50  (earning $18.50/day from theta)
        """
        result = self.query('positions', 'calculate_theta')
        return result.get('total_theta', 0.0)

    # ========================================================================
    # CONVENIENCE METHODS - OPPORTUNITIES
    # ========================================================================

    def scan_csp_opportunities(
        self,
        min_score: int = 70,
        dte_range: Optional[List[int]] = None,
        symbols: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """
        Scan for CSP opportunities.

        Args:
            min_score: Minimum opportunity score (0-100)
            dte_range: Days to expiration range [min, max]
            symbols: Limit scan to specific symbols (optional)

        Returns:
            List of opportunity dictionaries

        Example:
            opps = service.scan_csp_opportunities(
                min_score=80,
                dte_range=[20, 45]
            )
        """
        params = {'min_score': min_score}

        if dte_range:
            params['dte_min'] = dte_range[0]
            params['dte_max'] = dte_range[1]

        if symbols:
            params['symbols'] = symbols

        result = self.query('opportunities', 'scan_csp', params)
        return result.get('opportunities', [])

    def evaluate_trade_setup(self, symbol: str, strike: float, dte: int) -> Dict[str, Any]:
        """
        Evaluate a specific trade setup.

        Args:
            symbol: Stock symbol
            strike: Option strike price
            dte: Days to expiration

        Returns:
            Trade evaluation with scoring and analysis
        """
        params = {'symbol': symbol, 'strike': strike, 'dte': dte}
        return self.query('opportunities', 'evaluate_setup', params)

    # ========================================================================
    # CONVENIENCE METHODS - TRADINGVIEW WATCHLISTS
    # ========================================================================

    def get_watchlists(self) -> List[Dict[str, Any]]:
        """
        Get all TradingView watchlists.

        Returns:
            List of watchlist dictionaries
        """
        result = self.query('tradingview', 'fetch_watchlists')
        return result.get('watchlists', [])

    def get_watchlist_symbols(self, watchlist_name: str) -> List[str]:
        """
        Get symbols in a specific watchlist.

        Args:
            watchlist_name: Name of watchlist

        Returns:
            List of symbol strings
        """
        result = self.query('tradingview', 'get_symbols', {'watchlist': watchlist_name})
        return result.get('symbols', [])

    # ========================================================================
    # CONVENIENCE METHODS - KALSHI (PREDICTION MARKETS)
    # ========================================================================

    def get_kalshi_markets(self, category: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Get Kalshi prediction markets.

        Args:
            category: Filter by category (e.g., 'NFL', 'Politics')

        Returns:
            List of market dictionaries
        """
        params = {'category': category} if category else {}
        result = self.query('kalshi', 'fetch_markets', params)
        return result.get('markets', [])

    def get_nfl_predictions(self, week: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Get NFL game predictions.

        Args:
            week: NFL week number (optional, defaults to current week)

        Returns:
            List of game prediction dictionaries
        """
        params = {'week': week} if week else {}
        result = self.query('kalshi', 'fetch_nfl_predictions', params)
        return result.get('games', [])

    # ========================================================================
    # CONVENIENCE METHODS - XTRADES
    # ========================================================================

    def get_recent_alerts(self, limit: int = 50) -> List[Dict[str, Any]]:
        """
        Get recent Xtrades alerts.

        Args:
            limit: Max number of alerts

        Returns:
            List of alert dictionaries
        """
        result = self.query('xtrades', 'fetch_recent_alerts', {'limit': limit})
        return result.get('alerts', [])

    def get_profile_alerts(self, profile_name: str) -> List[Dict[str, Any]]:
        """
        Get alerts from a specific Xtrades profile.

        Args:
            profile_name: Profile username

        Returns:
            List of alert dictionaries
        """
        result = self.query('xtrades', 'fetch_profile_alerts', {'profile': profile_name})
        return result.get('alerts', [])

    # ========================================================================
    # CONVENIENCE METHODS - AI RESEARCH
    # ========================================================================

    def research_stock(self, symbol: str) -> Dict[str, Any]:
        """
        Run comprehensive AI research on a stock.

        Args:
            symbol: Stock symbol

        Returns:
            Research results with:
                - fundamental_score: 0-100
                - technical_score: 0-100
                - sentiment_score: 0-100
                - options_score: 0-100
                - overall_rating: BUY/HOLD/SELL
                - reasoning: Text explanation
        """
        return self.query('ai_research', 'analyze_stock', {'symbol': symbol})

    def get_stock_rating(self, symbol: str) -> str:
        """
        Get AI rating for a stock (quick version).

        Args:
            symbol: Stock symbol

        Returns:
            Rating string: 'BUY', 'HOLD', 'SELL', or 'UNKNOWN'
        """
        result = self.query('ai_research', 'get_rating', {'symbol': symbol})
        return result.get('rating', 'UNKNOWN')

    # ========================================================================
    # CONVENIENCE METHODS - EARNINGS CALENDAR
    # ========================================================================

    def get_earnings_date(self, symbol: str) -> Optional[str]:
        """
        Get upcoming earnings date for a symbol.

        Args:
            symbol: Stock symbol

        Returns:
            Earnings date (ISO format) or None if not available
        """
        result = self.query('earnings', 'fetch_date', {'symbol': symbol})
        return result.get('earnings_date')

    def get_upcoming_earnings(self, days_ahead: int = 7) -> List[Dict[str, Any]]:
        """
        Get all upcoming earnings in the next N days.

        Args:
            days_ahead: Number of days to look ahead

        Returns:
            List of earnings dictionaries
        """
        result = self.query('earnings', 'fetch_upcoming', {'days_ahead': days_ahead})
        return result.get('earnings', [])

    # ========================================================================
    # CONVENIENCE METHODS - CALENDAR SPREADS
    # ========================================================================

    def scan_calendar_spreads(self, min_score: int = 60) -> List[Dict[str, Any]]:
        """
        Scan for calendar spread opportunities.

        Args:
            min_score: Minimum spread quality score

        Returns:
            List of spread opportunities
        """
        result = self.query('calendar_spreads', 'scan', {'min_score': min_score})
        return result.get('spreads', [])

    # ========================================================================
    # CONVENIENCE METHODS - DATABASE SCAN
    # ========================================================================

    def get_database_stats(self) -> Dict[str, Any]:
        """
        Get stock database statistics.

        Returns:
            Stats dictionary with symbol counts by sector, price range, etc.
        """
        return self.query('database_scan', 'get_stats')

    def scan_database(
        self,
        sector: Optional[str] = None,
        min_price: Optional[float] = None,
        max_price: Optional[float] = None
    ) -> List[Dict[str, Any]]:
        """
        Scan stock database for opportunities.

        Args:
            sector: Filter by sector (e.g., 'Technology')
            min_price: Minimum stock price
            max_price: Maximum stock price

        Returns:
            List of opportunities
        """
        params = {}
        if sector:
            params['sector'] = sector
        if min_price:
            params['min_price'] = min_price
        if max_price:
            params['max_price'] = max_price

        result = self.query('database_scan', 'scan', params)
        return result.get('opportunities', [])

    # ========================================================================
    # CROSS-FEATURE INTELLIGENCE METHODS
    # ========================================================================

    def find_watchlist_opportunities(
        self,
        watchlist_name: str,
        min_score: int = 70
    ) -> List[Dict[str, Any]]:
        """
        Find CSP opportunities within a TradingView watchlist.

        This is an example of cross-feature intelligence:
        - Gets symbols from TradingView connector
        - Scans those symbols using Opportunities connector

        Args:
            watchlist_name: TradingView watchlist name
            min_score: Minimum opportunity score

        Returns:
            List of opportunities from watchlist symbols only
        """
        # Step 1: Get watchlist symbols
        symbols = self.get_watchlist_symbols(watchlist_name)

        if not symbols:
            logger.warning(f"Watchlist '{watchlist_name}' has no symbols")
            return []

        # Step 2: Scan those symbols for opportunities
        opportunities = self.scan_csp_opportunities(
            symbols=symbols,
            min_score=min_score
        )

        logger.info(f"Found {len(opportunities)} opportunities in watchlist '{watchlist_name}'")
        return opportunities

    def check_position_earnings_risk(self) -> List[Dict[str, Any]]:
        """
        Cross-reference active positions with upcoming earnings.

        This is cross-feature intelligence:
        - Gets positions from Positions connector
        - Gets earnings dates from Earnings connector
        - Identifies positions that expire after earnings

        Returns:
            List of positions at risk with earnings details
        """
        positions = self.get_active_positions()
        risky_positions = []

        for position in positions:
            symbol = position.get('symbol')
            expiration_str = position.get('expiration_date')

            if not symbol or not expiration_str:
                continue

            # Get earnings date
            earnings_date_str = self.get_earnings_date(symbol)

            if not earnings_date_str:
                continue

            # Parse dates
            from datetime import datetime
            try:
                expiration = datetime.fromisoformat(expiration_str)
                earnings_date = datetime.fromisoformat(earnings_date_str)

                # Check if earnings is before expiration (risk!)
                if earnings_date < expiration:
                    days_until_earnings = (earnings_date - datetime.now()).days

                    risky_positions.append({
                        'position': position,
                        'earnings_date': earnings_date_str,
                        'expiration_date': expiration_str,
                        'days_until_earnings': days_until_earnings
                    })
            except Exception as e:
                logger.warning(f"Error parsing dates for {symbol}: {e}")
                continue

        logger.info(f"Found {len(risky_positions)} positions with earnings risk")
        return risky_positions

    def get_ai_recommended_opportunities(
        self,
        rating_filter: str = "bullish",
        min_score: int = 70
    ) -> List[Dict[str, Any]]:
        """
        Find opportunities in stocks with specific AI ratings.

        Cross-feature intelligence:
        - Queries AI Research for stocks with desired rating
        - Scans those stocks for opportunities

        Args:
            rating_filter: 'bullish', 'bearish', or 'neutral'
            min_score: Minimum opportunity score

        Returns:
            High-conviction opportunities (AI agrees + good setup)
        """
        # TODO: Implement when AI Research connector has rating query
        # This would query cached AI research results
        # Then scan those symbols for opportunities
        logger.warning("AI recommended opportunities not yet implemented")
        return []

    # ========================================================================
    # MONITORING AND ADMIN
    # ========================================================================

    def get_all_cache_stats(self) -> List[Dict[str, Any]]:
        """
        Get cache statistics for all connectors.

        Returns:
            List of cache stats for each connector
        """
        self.initialize()

        stats = []
        for name, connector in self._connectors.items():
            stats.append(connector.get_cache_stats())

        return stats

    def invalidate_all_caches(self) -> None:
        """Invalidate all caches across all connectors"""
        self.initialize()

        for connector in self._connectors.values():
            connector.invalidate_cache()

        logger.info("Invalidated all caches across all connectors")

    def get_service_status(self) -> Dict[str, Any]:
        """
        Get overall service status and health metrics.

        Returns:
            Status dictionary with connector stats, cache performance, etc.
        """
        self.initialize()

        cache_stats = self.get_all_cache_stats()
        total_hits = sum(s['cache_hits'] for s in cache_stats)
        total_misses = sum(s['cache_misses'] for s in cache_stats)
        total_requests = total_hits + total_misses
        overall_hit_rate = (total_hits / total_requests * 100) if total_requests > 0 else 0

        return {
            'initialized': self._initialized,
            'connector_count': len(self._connectors),
            'connectors': list(self._connectors.keys()),
            'cache_stats': cache_stats,
            'overall_cache_hit_rate_pct': round(overall_hit_rate, 2),
            'total_requests': total_requests
        }


# ============================================================================
# SINGLETON INSTANCE
# ============================================================================

_data_service_instance: Optional[DataIntegrationService] = None
_instance_lock = threading.Lock()


def get_data_service() -> DataIntegrationService:
    """
    Get singleton instance of DataIntegrationService.

    This ensures only one instance exists across the application.

    Returns:
        DataIntegrationService singleton instance

    Example:
        service = get_data_service()
        balance = service.get_portfolio_balance()
    """
    global _data_service_instance

    if _data_service_instance is None:
        with _instance_lock:
            if _data_service_instance is None:
                _data_service_instance = DataIntegrationService()

    return _data_service_instance


# ============================================================================
# MODULE EXPORTS
# ============================================================================

__all__ = [
    'DataIntegrationService',
    'DataConnector',
    'CachePolicy',
    'DataSourceType',
    'get_data_service'
]

"""
Universe Service - Unified access to stocks and ETFs universe data
Provides filtering, caching, and efficient database access patterns
"""

import logging
from typing import Dict, List, Optional, Set, Any, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import asyncio
from functools import lru_cache
import threading

from src.database.connection_pool import get_db_connection

logger = logging.getLogger(__name__)


class AssetType(Enum):
    STOCK = "stock"
    ETF = "etf"
    ALL = "all"


@dataclass
class StockInfo:
    """Stock information from universe"""
    symbol: str
    company_name: Optional[str] = None
    exchange: Optional[str] = None
    sector: Optional[str] = None
    industry: Optional[str] = None
    current_price: Optional[float] = None
    market_cap: Optional[float] = None
    volume: Optional[int] = None
    avg_volume_10d: Optional[int] = None
    pe_ratio: Optional[float] = None
    dividend_yield: Optional[float] = None
    beta: Optional[float] = None
    week_52_high: Optional[float] = None
    week_52_low: Optional[float] = None
    sma_50: Optional[float] = None
    sma_200: Optional[float] = None
    rsi_14: Optional[float] = None
    has_options: bool = False
    is_active: bool = True
    last_updated: Optional[datetime] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "symbol": self.symbol,
            "company_name": self.company_name,
            "exchange": self.exchange,
            "sector": self.sector,
            "industry": self.industry,
            "current_price": self.current_price,
            "market_cap": self.market_cap,
            "volume": self.volume,
            "avg_volume_10d": self.avg_volume_10d,
            "pe_ratio": self.pe_ratio,
            "dividend_yield": self.dividend_yield,
            "beta": self.beta,
            "week_52_high": self.week_52_high,
            "week_52_low": self.week_52_low,
            "sma_50": self.sma_50,
            "sma_200": self.sma_200,
            "rsi_14": self.rsi_14,
            "has_options": self.has_options,
            "is_active": self.is_active,
            "last_updated": self.last_updated.isoformat() if self.last_updated else None
        }


@dataclass
class ETFInfo:
    """ETF information from universe"""
    symbol: str
    fund_name: Optional[str] = None
    exchange: Optional[str] = None
    category: Optional[str] = None
    fund_family: Optional[str] = None
    current_price: Optional[float] = None
    total_assets: Optional[float] = None
    expense_ratio: Optional[float] = None
    volume: Optional[int] = None
    avg_volume_10d: Optional[int] = None
    yield_ttm: Optional[float] = None
    ytd_return: Optional[float] = None
    beta: Optional[float] = None
    week_52_high: Optional[float] = None
    week_52_low: Optional[float] = None
    sma_50: Optional[float] = None
    sma_200: Optional[float] = None
    rsi_14: Optional[float] = None
    has_options: bool = False
    is_active: bool = True
    last_updated: Optional[datetime] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "symbol": self.symbol,
            "fund_name": self.fund_name,
            "exchange": self.exchange,
            "category": self.category,
            "fund_family": self.fund_family,
            "current_price": self.current_price,
            "total_assets": self.total_assets,
            "expense_ratio": self.expense_ratio,
            "volume": self.volume,
            "avg_volume_10d": self.avg_volume_10d,
            "yield_ttm": self.yield_ttm,
            "ytd_return": self.ytd_return,
            "beta": self.beta,
            "week_52_high": self.week_52_high,
            "week_52_low": self.week_52_low,
            "sma_50": self.sma_50,
            "sma_200": self.sma_200,
            "rsi_14": self.rsi_14,
            "has_options": self.has_options,
            "is_active": self.is_active,
            "last_updated": self.last_updated.isoformat() if self.last_updated else None
        }


@dataclass
class UniverseFilter:
    """Filter criteria for universe queries"""
    asset_type: AssetType = AssetType.ALL
    min_price: Optional[float] = None
    max_price: Optional[float] = None
    min_market_cap: Optional[float] = None
    max_market_cap: Optional[float] = None
    min_volume: Optional[int] = None
    sectors: Optional[List[str]] = None
    industries: Optional[List[str]] = None
    categories: Optional[List[str]] = None  # For ETFs
    has_options_only: bool = False
    min_beta: Optional[float] = None
    max_beta: Optional[float] = None
    exchanges: Optional[List[str]] = None
    symbols: Optional[List[str]] = None  # Filter to specific symbols
    active_only: bool = True
    limit: int = 1000
    offset: int = 0

    def to_cache_key(self) -> str:
        """Generate cache key from filter parameters"""
        parts = [
            f"type:{self.asset_type.value}",
            f"price:{self.min_price}-{self.max_price}",
            f"mcap:{self.min_market_cap}-{self.max_market_cap}",
            f"vol:{self.min_volume}",
            f"sectors:{','.join(sorted(self.sectors)) if self.sectors else 'all'}",
            f"options:{self.has_options_only}",
            f"active:{self.active_only}",
            f"limit:{self.limit}",
            f"offset:{self.offset}"
        ]
        return "|".join(parts)


class UniverseService:
    """
    Unified service for accessing stocks and ETFs universe data.
    Implements caching, filtering, and efficient database access.
    """

    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return

        self._initialized = True
        self._cache: Dict[str, Tuple[Any, datetime]] = {}
        self._cache_ttl = timedelta(minutes=5)
        self._symbol_lookup: Dict[str, StockInfo | ETFInfo] = {}
        self._optionable_symbols: Set[str] = set()
        self._sectors: List[str] = []
        self._categories: List[str] = []
        self._last_refresh = None

        logger.info("UniverseService initialized")

    def _get_cached(self, key: str) -> Optional[Any]:
        """Get cached value if not expired"""
        if key in self._cache:
            value, timestamp = self._cache[key]
            if datetime.now() - timestamp < self._cache_ttl:
                return value
            del self._cache[key]
        return None

    def _set_cached(self, key: str, value: Any) -> None:
        """Set cache value with current timestamp"""
        self._cache[key] = (value, datetime.now())

    def invalidate_cache(self) -> None:
        """Clear all cached data"""
        self._cache.clear()
        self._symbol_lookup.clear()
        self._optionable_symbols.clear()
        self._sectors.clear()
        self._categories.clear()
        self._last_refresh = None
        logger.info("Universe cache invalidated")

    def get_universe_stats(self) -> Dict[str, Any]:
        """Get summary statistics for the universe"""
        cache_key = "universe_stats"
        cached = self._get_cached(cache_key)
        if cached:
            return cached

        try:
            with get_db_connection() as conn:
                cursor = conn.cursor()

                # Stock counts
                cursor.execute("""
                    SELECT
                        COUNT(*) as total,
                        COUNT(*) FILTER (WHERE has_options = true) as optionable,
                        COUNT(*) FILTER (WHERE is_active = true) as active,
                        COUNT(DISTINCT sector) as sectors,
                        AVG(current_price) as avg_price,
                        AVG(market_cap) as avg_market_cap
                    FROM stocks_universe
                """)
                stock_row = cursor.fetchone()

                # ETF counts
                cursor.execute("""
                    SELECT
                        COUNT(*) as total,
                        COUNT(*) FILTER (WHERE has_options = true) as optionable,
                        COUNT(*) FILTER (WHERE is_active = true) as active,
                        COUNT(DISTINCT category) as categories,
                        AVG(current_price) as avg_price,
                        SUM(total_assets) as total_aum
                    FROM etfs_universe
                """)
                etf_row = cursor.fetchone()

                stats = {
                    "stocks": {
                        "total": stock_row[0] or 0,
                        "optionable": stock_row[1] or 0,
                        "active": stock_row[2] or 0,
                        "sectors": stock_row[3] or 0,
                        "avg_price": round(float(stock_row[4] or 0), 2),
                        "avg_market_cap": round(float(stock_row[5] or 0) / 1e9, 2)  # In billions
                    },
                    "etfs": {
                        "total": etf_row[0] or 0,
                        "optionable": etf_row[1] or 0,
                        "active": etf_row[2] or 0,
                        "categories": etf_row[3] or 0,
                        "avg_price": round(float(etf_row[4] or 0), 2),
                        "total_aum_billions": round(float(etf_row[5] or 0) / 1e9, 2)
                    },
                    "generated_at": datetime.now().isoformat()
                }

                self._set_cached(cache_key, stats)
                return stats

        except Exception as e:
            logger.error(f"Error getting universe stats: {e}")
            return {"error": str(e), "stocks": {}, "etfs": {}}

    def get_sectors(self) -> List[str]:
        """Get all unique sectors"""
        if self._sectors:
            return self._sectors

        try:
            with get_db_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT DISTINCT sector
                    FROM stocks_universe
                    WHERE sector IS NOT NULL AND sector <> ''
                    ORDER BY sector
                """)
                self._sectors = [row[0] for row in cursor.fetchall()]
                return self._sectors
        except Exception as e:
            logger.error(f"Error getting sectors: {e}")
            return []

    def get_categories(self) -> List[str]:
        """Get all unique ETF categories"""
        if self._categories:
            return self._categories

        try:
            with get_db_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT DISTINCT category
                    FROM etfs_universe
                    WHERE category IS NOT NULL AND category <> ''
                    ORDER BY category
                """)
                self._categories = [row[0] for row in cursor.fetchall()]
                return self._categories
        except Exception as e:
            logger.error(f"Error getting categories: {e}")
            return []

    def get_optionable_symbols(self, asset_type: AssetType = AssetType.ALL) -> Set[str]:
        """Get all symbols that have options available"""
        cache_key = f"optionable_{asset_type.value}"
        cached = self._get_cached(cache_key)
        if cached:
            return cached

        symbols = set()
        try:
            with get_db_connection() as conn:
                cursor = conn.cursor()

                if asset_type in (AssetType.STOCK, AssetType.ALL):
                    cursor.execute("""
                        SELECT symbol FROM stocks_universe
                        WHERE has_options = true AND is_active = true
                    """)
                    symbols.update(row[0] for row in cursor.fetchall())

                if asset_type in (AssetType.ETF, AssetType.ALL):
                    cursor.execute("""
                        SELECT symbol FROM etfs_universe
                        WHERE has_options = true AND is_active = true
                    """)
                    symbols.update(row[0] for row in cursor.fetchall())

                self._set_cached(cache_key, symbols)
                return symbols

        except Exception as e:
            logger.error(f"Error getting optionable symbols: {e}")
            return set()

    def get_stocks(self, filter: Optional[UniverseFilter] = None) -> List[StockInfo]:
        """Get stocks matching filter criteria"""
        if filter is None:
            filter = UniverseFilter(asset_type=AssetType.STOCK)

        cache_key = f"stocks_{filter.to_cache_key()}"
        cached = self._get_cached(cache_key)
        if cached:
            return cached

        try:
            with get_db_connection() as conn:
                cursor = conn.cursor()

                # Build dynamic query
                conditions = []
                params = []

                if filter.active_only:
                    conditions.append("is_active = true")

                if filter.has_options_only:
                    conditions.append("has_options = true")

                if filter.min_price is not None:
                    conditions.append("current_price >= %s")
                    params.append(filter.min_price)

                if filter.max_price is not None:
                    conditions.append("current_price <= %s")
                    params.append(filter.max_price)

                if filter.min_market_cap is not None:
                    conditions.append("market_cap >= %s")
                    params.append(filter.min_market_cap)

                if filter.max_market_cap is not None:
                    conditions.append("market_cap <= %s")
                    params.append(filter.max_market_cap)

                if filter.min_volume is not None:
                    conditions.append("avg_volume_10d >= %s")
                    params.append(filter.min_volume)

                if filter.sectors:
                    conditions.append("sector = ANY(%s)")
                    params.append(filter.sectors)

                if filter.industries:
                    conditions.append("industry = ANY(%s)")
                    params.append(filter.industries)

                if filter.exchanges:
                    conditions.append("exchange = ANY(%s)")
                    params.append(filter.exchanges)

                if filter.symbols:
                    conditions.append("symbol = ANY(%s)")
                    params.append([s.upper() for s in filter.symbols])

                if filter.min_beta is not None:
                    conditions.append("beta >= %s")
                    params.append(filter.min_beta)

                if filter.max_beta is not None:
                    conditions.append("beta <= %s")
                    params.append(filter.max_beta)

                where_clause = " AND ".join(conditions) if conditions else "1=1"

                query = f"""
                    SELECT
                        symbol, company_name, exchange, sector, industry,
                        current_price, market_cap, volume, avg_volume_10d,
                        pe_ratio, dividend_yield, beta,
                        week_52_high, week_52_low, sma_50, sma_200, rsi_14,
                        has_options, is_active, last_updated
                    FROM stocks_universe
                    WHERE {where_clause}
                    ORDER BY market_cap DESC NULLS LAST
                    LIMIT %s OFFSET %s
                """
                params.extend([filter.limit, filter.offset])

                cursor.execute(query, params)
                rows = cursor.fetchall()

                stocks = []
                for row in rows:
                    stock = StockInfo(
                        symbol=row[0],
                        company_name=row[1],
                        exchange=row[2],
                        sector=row[3],
                        industry=row[4],
                        current_price=float(row[5]) if row[5] else None,
                        market_cap=float(row[6]) if row[6] else None,
                        volume=int(row[7]) if row[7] else None,
                        avg_volume_10d=int(row[8]) if row[8] else None,
                        pe_ratio=float(row[9]) if row[9] else None,
                        dividend_yield=float(row[10]) if row[10] else None,
                        beta=float(row[11]) if row[11] else None,
                        week_52_high=float(row[12]) if row[12] else None,
                        week_52_low=float(row[13]) if row[13] else None,
                        sma_50=float(row[14]) if row[14] else None,
                        sma_200=float(row[15]) if row[15] else None,
                        rsi_14=float(row[16]) if row[16] else None,
                        has_options=bool(row[17]),
                        is_active=bool(row[18]),
                        last_updated=row[19]
                    )
                    stocks.append(stock)

                self._set_cached(cache_key, stocks)
                return stocks

        except Exception as e:
            logger.error(f"Error getting stocks: {e}")
            return []

    def get_etfs(self, filter: Optional[UniverseFilter] = None) -> List[ETFInfo]:
        """Get ETFs matching filter criteria"""
        if filter is None:
            filter = UniverseFilter(asset_type=AssetType.ETF)

        cache_key = f"etfs_{filter.to_cache_key()}"
        cached = self._get_cached(cache_key)
        if cached:
            return cached

        try:
            with get_db_connection() as conn:
                cursor = conn.cursor()

                conditions = []
                params = []

                if filter.active_only:
                    conditions.append("is_active = true")

                if filter.has_options_only:
                    conditions.append("has_options = true")

                if filter.min_price is not None:
                    conditions.append("current_price >= %s")
                    params.append(filter.min_price)

                if filter.max_price is not None:
                    conditions.append("current_price <= %s")
                    params.append(filter.max_price)

                if filter.min_volume is not None:
                    conditions.append("avg_volume_10d >= %s")
                    params.append(filter.min_volume)

                if filter.categories:
                    conditions.append("category = ANY(%s)")
                    params.append(filter.categories)

                if filter.exchanges:
                    conditions.append("exchange = ANY(%s)")
                    params.append(filter.exchanges)

                if filter.symbols:
                    conditions.append("symbol = ANY(%s)")
                    params.append([s.upper() for s in filter.symbols])

                where_clause = " AND ".join(conditions) if conditions else "1=1"

                query = f"""
                    SELECT
                        symbol, fund_name, exchange, category, fund_family,
                        current_price, total_assets, expense_ratio,
                        volume, avg_volume_10d, yield_ttm, ytd_return, beta,
                        week_52_high, week_52_low, sma_50, sma_200, rsi_14,
                        has_options, is_active, last_updated
                    FROM etfs_universe
                    WHERE {where_clause}
                    ORDER BY total_assets DESC NULLS LAST
                    LIMIT %s OFFSET %s
                """
                params.extend([filter.limit, filter.offset])

                cursor.execute(query, params)
                rows = cursor.fetchall()

                etfs = []
                for row in rows:
                    etf = ETFInfo(
                        symbol=row[0],
                        fund_name=row[1],
                        exchange=row[2],
                        category=row[3],
                        fund_family=row[4],
                        current_price=float(row[5]) if row[5] else None,
                        total_assets=float(row[6]) if row[6] else None,
                        expense_ratio=float(row[7]) if row[7] else None,
                        volume=int(row[8]) if row[8] else None,
                        avg_volume_10d=int(row[9]) if row[9] else None,
                        yield_ttm=float(row[10]) if row[10] else None,
                        ytd_return=float(row[11]) if row[11] else None,
                        beta=float(row[12]) if row[12] else None,
                        week_52_high=float(row[13]) if row[13] else None,
                        week_52_low=float(row[14]) if row[14] else None,
                        sma_50=float(row[15]) if row[15] else None,
                        sma_200=float(row[16]) if row[16] else None,
                        rsi_14=float(row[17]) if row[17] else None,
                        has_options=bool(row[18]),
                        is_active=bool(row[19]),
                        last_updated=row[20]
                    )
                    etfs.append(etf)

                self._set_cached(cache_key, etfs)
                return etfs

        except Exception as e:
            logger.error(f"Error getting ETFs: {e}")
            return []

    def get_symbol_info(self, symbol: str) -> Optional[StockInfo | ETFInfo]:
        """Get info for a specific symbol (stock or ETF)"""
        symbol = symbol.upper()

        # Check cache first
        if symbol in self._symbol_lookup:
            return self._symbol_lookup[symbol]

        try:
            with get_db_connection() as conn:
                cursor = conn.cursor()

                # Try stocks first
                cursor.execute("""
                    SELECT
                        symbol, company_name, exchange, sector, industry,
                        current_price, market_cap, volume, avg_volume_10d,
                        pe_ratio, dividend_yield, beta,
                        week_52_high, week_52_low, sma_50, sma_200, rsi_14,
                        has_options, is_active, last_updated
                    FROM stocks_universe
                    WHERE symbol = %s
                """, (symbol,))
                row = cursor.fetchone()

                if row:
                    stock = StockInfo(
                        symbol=row[0],
                        company_name=row[1],
                        exchange=row[2],
                        sector=row[3],
                        industry=row[4],
                        current_price=float(row[5]) if row[5] else None,
                        market_cap=float(row[6]) if row[6] else None,
                        volume=int(row[7]) if row[7] else None,
                        avg_volume_10d=int(row[8]) if row[8] else None,
                        pe_ratio=float(row[9]) if row[9] else None,
                        dividend_yield=float(row[10]) if row[10] else None,
                        beta=float(row[11]) if row[11] else None,
                        week_52_high=float(row[12]) if row[12] else None,
                        week_52_low=float(row[13]) if row[13] else None,
                        sma_50=float(row[14]) if row[14] else None,
                        sma_200=float(row[15]) if row[15] else None,
                        rsi_14=float(row[16]) if row[16] else None,
                        has_options=bool(row[17]),
                        is_active=bool(row[18]),
                        last_updated=row[19]
                    )
                    self._symbol_lookup[symbol] = stock
                    return stock

                # Try ETFs
                cursor.execute("""
                    SELECT
                        symbol, fund_name, exchange, category, fund_family,
                        current_price, total_assets, expense_ratio,
                        volume, avg_volume_10d, yield_ttm, ytd_return, beta,
                        week_52_high, week_52_low, sma_50, sma_200, rsi_14,
                        has_options, is_active, last_updated
                    FROM etfs_universe
                    WHERE symbol = %s
                """, (symbol,))
                row = cursor.fetchone()

                if row:
                    etf = ETFInfo(
                        symbol=row[0],
                        fund_name=row[1],
                        exchange=row[2],
                        category=row[3],
                        fund_family=row[4],
                        current_price=float(row[5]) if row[5] else None,
                        total_assets=float(row[6]) if row[6] else None,
                        expense_ratio=float(row[7]) if row[7] else None,
                        volume=int(row[8]) if row[8] else None,
                        avg_volume_10d=int(row[9]) if row[9] else None,
                        yield_ttm=float(row[10]) if row[10] else None,
                        ytd_return=float(row[11]) if row[11] else None,
                        beta=float(row[12]) if row[12] else None,
                        week_52_high=float(row[13]) if row[13] else None,
                        week_52_low=float(row[14]) if row[14] else None,
                        sma_50=float(row[15]) if row[15] else None,
                        sma_200=float(row[16]) if row[16] else None,
                        rsi_14=float(row[17]) if row[17] else None,
                        has_options=bool(row[18]),
                        is_active=bool(row[19]),
                        last_updated=row[20]
                    )
                    self._symbol_lookup[symbol] = etf
                    return etf

                return None

        except Exception as e:
            logger.error(f"Error getting symbol info for {symbol}: {e}")
            return None

    def validate_symbols(self, symbols: List[str],
                        require_options: bool = True,
                        max_price: Optional[float] = None) -> Tuple[List[str], List[str]]:
        """
        Validate symbols against universe and return valid/invalid lists.
        Useful for pre-filtering before expensive operations like options scanning.
        """
        valid = []
        invalid = []

        optionable = self.get_optionable_symbols() if require_options else None

        for symbol in symbols:
            symbol = symbol.upper()
            info = self.get_symbol_info(symbol)

            if info is None:
                invalid.append(symbol)
                continue

            if require_options and symbol not in optionable:
                invalid.append(symbol)
                continue

            if max_price and info.current_price and info.current_price > max_price:
                invalid.append(symbol)
                continue

            valid.append(symbol)

        return valid, invalid

    def get_scannable_symbols(self,
                             max_price: float = 500.0,
                             min_volume: int = 100000,
                             sectors: Optional[List[str]] = None,
                             include_etfs: bool = True,
                             limit: int = 500) -> List[str]:
        """
        Get symbols suitable for premium scanning.
        Pre-filters based on options availability, price, and volume.
        """
        symbols = []

        # Get qualifying stocks
        stock_filter = UniverseFilter(
            asset_type=AssetType.STOCK,
            max_price=max_price,
            min_volume=min_volume,
            sectors=sectors,
            has_options_only=True,
            active_only=True,
            limit=limit
        )
        stocks = self.get_stocks(stock_filter)
        symbols.extend(s.symbol for s in stocks)

        # Get qualifying ETFs
        if include_etfs:
            etf_filter = UniverseFilter(
                asset_type=AssetType.ETF,
                max_price=max_price,
                min_volume=min_volume,
                has_options_only=True,
                active_only=True,
                limit=limit // 2
            )
            etfs = self.get_etfs(etf_filter)
            symbols.extend(e.symbol for e in etfs)

        return symbols[:limit]


# Singleton accessor
def get_universe_service() -> UniverseService:
    return UniverseService()

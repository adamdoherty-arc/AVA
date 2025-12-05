"""
Async Market Data Utilities

Modern async wrappers for yfinance and other market data providers.
Uses asyncio.to_thread() to prevent blocking the event loop.

Usage:
    from backend.utils.async_market_data import (
        async_get_ticker,
        async_get_history,
        async_get_options_chain,
        async_batch_tickers
    )

    # Single ticker
    ticker = await async_get_ticker("AAPL")
    history = await async_get_history(ticker, period="1mo")

    # Batch tickers (concurrent)
    tickers = await async_batch_tickers(["AAPL", "GOOGL", "MSFT"])
"""

import asyncio
from typing import Optional, List, Dict, Any, Union
from datetime import datetime, timedelta
from functools import lru_cache
import structlog

logger = structlog.get_logger(__name__)

# Lazy import yfinance to avoid import overhead
_yf = None

def _get_yfinance():
    """Lazy load yfinance module."""
    global _yf
    if _yf is None:
        import yfinance as yf
        _yf = yf
    return _yf


# ============================================================================
# Core Async Wrappers
# ============================================================================

async def async_get_ticker(symbol: str) -> Any:
    """
    Get a yfinance Ticker object asynchronously.

    Args:
        symbol: Stock ticker symbol (e.g., "AAPL")

    Returns:
        yfinance.Ticker object
    """
    yf = _get_yfinance()
    return await asyncio.to_thread(yf.Ticker, symbol.upper())


async def async_get_history(
    ticker_or_symbol: Union[Any, str],
    period: str = "1mo",
    interval: str = "1d",
    start: Optional[datetime] = None,
    end: Optional[datetime] = None,
    actions: bool = True,
    auto_adjust: bool = True
) -> Any:
    """
    Get historical price data asynchronously.

    Args:
        ticker_or_symbol: yfinance.Ticker object or symbol string
        period: Time period (1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max)
        interval: Data interval (1m, 2m, 5m, 15m, 30m, 60m, 90m, 1h, 1d, 5d, 1wk, 1mo, 3mo)
        start: Start date (alternative to period)
        end: End date (alternative to period)
        actions: Include dividends and stock splits
        auto_adjust: Auto-adjust OHLC prices

    Returns:
        pandas.DataFrame with OHLC data
    """
    if isinstance(ticker_or_symbol, str):
        ticker = await async_get_ticker(ticker_or_symbol)
    else:
        ticker = ticker_or_symbol

    def _fetch():
        if start and end:
            return ticker.history(
                start=start, end=end, interval=interval,
                actions=actions, auto_adjust=auto_adjust
            )
        return ticker.history(
            period=period, interval=interval,
            actions=actions, auto_adjust=auto_adjust
        )

    return await asyncio.to_thread(_fetch)


async def async_get_info(ticker_or_symbol: Union[Any, str]) -> Dict[str, Any]:
    """
    Get ticker info asynchronously.

    Args:
        ticker_or_symbol: yfinance.Ticker object or symbol string

    Returns:
        Dict with ticker information
    """
    if isinstance(ticker_or_symbol, str):
        ticker = await async_get_ticker(ticker_or_symbol)
    else:
        ticker = ticker_or_symbol

    return await asyncio.to_thread(lambda: ticker.info)


async def async_get_options_dates(ticker_or_symbol: Union[Any, str]) -> List[str]:
    """
    Get available options expiration dates asynchronously.

    Args:
        ticker_or_symbol: yfinance.Ticker object or symbol string

    Returns:
        List of expiration date strings
    """
    if isinstance(ticker_or_symbol, str):
        ticker = await async_get_ticker(ticker_or_symbol)
    else:
        ticker = ticker_or_symbol

    return await asyncio.to_thread(lambda: list(ticker.options))


async def async_get_options_chain(
    ticker_or_symbol: Union[Any, str],
    date: Optional[str] = None
) -> Any:
    """
    Get options chain asynchronously.

    Args:
        ticker_or_symbol: yfinance.Ticker object or symbol string
        date: Expiration date string (YYYY-MM-DD)

    Returns:
        Options object with .calls and .puts DataFrames
    """
    if isinstance(ticker_or_symbol, str):
        ticker = await async_get_ticker(ticker_or_symbol)
    else:
        ticker = ticker_or_symbol

    def _fetch():
        if date:
            return ticker.option_chain(date)
        # Get nearest expiration
        dates = ticker.options
        if dates:
            return ticker.option_chain(dates[0])
        return None

    return await asyncio.to_thread(_fetch)


async def async_get_financials(ticker_or_symbol: Union[Any, str]) -> Dict[str, Any]:
    """
    Get financial statements asynchronously.

    Args:
        ticker_or_symbol: yfinance.Ticker object or symbol string

    Returns:
        Dict with income_stmt, balance_sheet, cash_flow DataFrames
    """
    if isinstance(ticker_or_symbol, str):
        ticker = await async_get_ticker(ticker_or_symbol)
    else:
        ticker = ticker_or_symbol

    def _fetch():
        return {
            "income_stmt": ticker.income_stmt,
            "balance_sheet": ticker.balance_sheet,
            "cash_flow": ticker.cash_flow,
            "quarterly_income_stmt": ticker.quarterly_income_stmt,
            "quarterly_balance_sheet": ticker.quarterly_balance_sheet,
            "quarterly_cash_flow": ticker.quarterly_cash_flow
        }

    return await asyncio.to_thread(_fetch)


# ============================================================================
# Batch Operations (Concurrent)
# ============================================================================

async def async_batch_tickers(symbols: List[str]) -> Dict[str, Any]:
    """
    Get multiple Ticker objects concurrently.

    Args:
        symbols: List of ticker symbols

    Returns:
        Dict mapping symbol -> Ticker object
    """
    tasks = [async_get_ticker(sym) for sym in symbols]
    tickers = await asyncio.gather(*tasks, return_exceptions=True)

    result = {}
    for sym, ticker in zip(symbols, tickers):
        if isinstance(ticker, Exception):
            logger.warning("ticker_fetch_error", symbol=sym, error=str(ticker))
            result[sym] = None
        else:
            result[sym] = ticker

    return result


async def async_batch_history(
    symbols: List[str],
    period: str = "1mo",
    interval: str = "1d"
) -> Dict[str, Any]:
    """
    Get historical data for multiple symbols concurrently.

    Args:
        symbols: List of ticker symbols
        period: Time period
        interval: Data interval

    Returns:
        Dict mapping symbol -> DataFrame
    """
    async def _get_one(symbol: str):
        ticker = await async_get_ticker(symbol)
        return await async_get_history(ticker, period=period, interval=interval)

    tasks = [_get_one(sym) for sym in symbols]
    histories = await asyncio.gather(*tasks, return_exceptions=True)

    result = {}
    for sym, hist in zip(symbols, histories):
        if isinstance(hist, Exception):
            logger.warning("history_fetch_error", symbol=sym, error=str(hist))
            result[sym] = None
        else:
            result[sym] = hist

    return result


async def async_batch_info(symbols: List[str]) -> Dict[str, Dict[str, Any]]:
    """
    Get info for multiple symbols concurrently.

    Args:
        symbols: List of ticker symbols

    Returns:
        Dict mapping symbol -> info dict
    """
    async def _get_one(symbol: str):
        return await async_get_info(symbol)

    tasks = [_get_one(sym) for sym in symbols]
    infos = await asyncio.gather(*tasks, return_exceptions=True)

    result = {}
    for sym, info in zip(symbols, infos):
        if isinstance(info, Exception):
            logger.warning("info_fetch_error", symbol=sym, error=str(info))
            result[sym] = {}
        else:
            result[sym] = info

    return result


# ============================================================================
# Convenience Functions
# ============================================================================

async def async_get_current_price(symbol: str) -> Optional[float]:
    """
    Get current stock price asynchronously.

    Args:
        symbol: Ticker symbol

    Returns:
        Current price or None if unavailable
    """
    try:
        info = await async_get_info(symbol)
        return info.get("currentPrice") or info.get("regularMarketPrice")
    except Exception as e:
        logger.warning("price_fetch_error", symbol=symbol, error=str(e))
        return None


async def async_get_earnings_dates(ticker_or_symbol: Union[Any, str]) -> Any:
    """
    Get earnings dates asynchronously.

    Args:
        ticker_or_symbol: yfinance.Ticker object or symbol string

    Returns:
        DataFrame with earnings dates
    """
    if isinstance(ticker_or_symbol, str):
        ticker = await async_get_ticker(ticker_or_symbol)
    else:
        ticker = ticker_or_symbol

    return await asyncio.to_thread(lambda: ticker.earnings_dates)


async def async_get_recommendations(ticker_or_symbol: Union[Any, str]) -> Any:
    """
    Get analyst recommendations asynchronously.

    Args:
        ticker_or_symbol: yfinance.Ticker object or symbol string

    Returns:
        DataFrame with recommendations
    """
    if isinstance(ticker_or_symbol, str):
        ticker = await async_get_ticker(ticker_or_symbol)
    else:
        ticker = ticker_or_symbol

    return await asyncio.to_thread(lambda: ticker.recommendations)


# ============================================================================
# Cached Variants (for repeated calls within short timeframes)
# ============================================================================

class AsyncMarketDataCache:
    """
    In-memory cache for market data with TTL.

    Usage:
        cache = AsyncMarketDataCache(ttl_seconds=300)
        price = await cache.get_price("AAPL")
    """

    def __init__(self, ttl_seconds: int = 300):
        self.ttl = ttl_seconds
        self._cache: Dict[str, tuple[Any, datetime]] = {}

    def _is_valid(self, key: str) -> bool:
        if key not in self._cache:
            return False
        _, timestamp = self._cache[key]
        return datetime.now() - timestamp < timedelta(seconds=self.ttl)

    async def get_price(self, symbol: str) -> Optional[float]:
        """Get cached price or fetch fresh."""
        key = f"price:{symbol}"
        if self._is_valid(key):
            return self._cache[key][0]

        price = await async_get_current_price(symbol)
        if price is not None:
            self._cache[key] = (price, datetime.now())
        return price

    async def get_info(self, symbol: str) -> Dict[str, Any]:
        """Get cached info or fetch fresh."""
        key = f"info:{symbol}"
        if self._is_valid(key):
            return self._cache[key][0]

        info = await async_get_info(symbol)
        self._cache[key] = (info, datetime.now())
        return info

    async def get_history(
        self,
        symbol: str,
        period: str = "1mo",
        interval: str = "1d"
    ) -> Any:
        """Get cached history or fetch fresh."""
        key = f"history:{symbol}:{period}:{interval}"
        if self._is_valid(key):
            return self._cache[key][0]

        history = await async_get_history(symbol, period=period, interval=interval)
        self._cache[key] = (history, datetime.now())
        return history

    def clear(self) -> None:
        """Clear all cached data."""
        self._cache.clear()

    def clear_symbol(self, symbol: str):
        """Clear cached data for a specific symbol."""
        keys_to_remove = [k for k in self._cache if symbol in k]
        for k in keys_to_remove:
            del self._cache[k]


# Global cache instance (5 minute TTL)
market_data_cache = AsyncMarketDataCache(ttl_seconds=300)

"""
YFinance Rate-Limited Wrapper
Centralized wrapper for all yfinance calls with built-in rate limiting and caching

PERFORMANCE: Prevents API throttling and enables response caching
USAGE: Replace all direct yf.Ticker() calls with get_ticker()
"""

import yfinance as yf
import logging
import time
from functools import wraps, lru_cache
from typing import Any, Optional, Dict
import threading

logger = logging.getLogger(__name__)


class YFinanceRateLimiter:
    """
    Rate limiter for YFinance API calls

    Prevents 429 errors by limiting requests to 2000/hour (Yahoo's limit)
    Conservative: 5 requests/second = 300/minute = 18,000/hour
    """

    def __init__(self, max_calls_per_second: float = 5.0):
        """
        Initialize rate limiter

        Args:
            max_calls_per_second: Maximum API calls per second (default: 5)
        """
        self.max_calls = max_calls_per_second
        self.min_interval = 1.0 / max_calls_per_second  # 0.2 seconds between calls
        self.last_call_time = 0
        self.lock = threading.Lock()
        self.total_calls = 0
        self.throttled_calls = 0

    def wait_if_needed(self) -> None:
        """Wait if we're calling too frequently"""
        with self.lock:
            now = time.time()
            time_since_last = now - self.last_call_time

            if time_since_last < self.min_interval:
                sleep_time = self.min_interval - time_since_last
                logger.debug(f"Rate limiting: sleeping {sleep_time:.3f}s")
                time.sleep(sleep_time)
                self.throttled_calls += 1

            self.last_call_time = time.time()
            self.total_calls += 1

    def get_stats(self) -> Dict[str, int]:
        """Get rate limiter statistics"""
        return {
            'total_calls': self.total_calls,
            'throttled_calls': self.throttled_calls,
            'throttle_rate': (self.throttled_calls / max(self.total_calls, 1)) * 100
        }


# Global rate limiter instance
_rate_limiter = YFinanceRateLimiter(max_calls_per_second=5.0)


# PERFORMANCE: LRU cache for ticker objects (reduces redundant API calls)
@lru_cache(maxsize=1000)
def _get_cached_ticker(symbol: str) -> yf.Ticker:
    """
    Get cached ticker object

    Args:
        symbol: Stock symbol

    Returns:
        yfinance Ticker object
    """
    return yf.Ticker(symbol)


def get_ticker(symbol: str, use_cache: bool = True) -> Optional[yf.Ticker]:
    """
    Get a yfinance Ticker with rate limiting

    PERFORMANCE: Rate limited and cached wrapper for yf.Ticker()

    Args:
        symbol: Stock symbol (e.g., 'AAPL')
        use_cache: Use cached ticker object (default: True)

    Returns:
        yfinance Ticker object or None on error

    Example:
        >>> ticker = get_ticker('AAPL')
        >>> info = ticker.info
    """
    try:
        # Apply rate limiting
        _rate_limiter.wait_if_needed()

        # Get ticker (cached or new)
        if use_cache:
            ticker = _get_cached_ticker(symbol.upper())
        else:
            ticker = yf.Ticker(symbol.upper())

        return ticker

    except Exception as e:
        logger.error(f"Error fetching ticker for {symbol}: {e}")
        return None


def get_ticker_info(symbol: str, use_cache: bool = True) -> Optional[Dict[str, Any]]:
    """
    Get ticker info with rate limiting

    PERFORMANCE: Cached and rate-limited wrapper

    Args:
        symbol: Stock symbol
        use_cache: Use cached data (default: True)

    Returns:
        Ticker info dictionary or None on error
    """
    try:
        ticker = get_ticker(symbol, use_cache=use_cache)
        if ticker:
            return ticker.info
        return None
    except Exception as e:
        logger.error(f"Error fetching info for {symbol}: {e}")
        return None


def get_ticker_history(
    symbol: str,
    period: str = "1mo",
    interval: str = "1d",
    use_cache: bool = True
) -> Optional[Any]:
    """
    Get ticker price history with rate limiting

    Args:
        symbol: Stock symbol
        period: Data period (1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max)
        interval: Data interval (1m, 2m, 5m, 15m, 30m, 60m, 90m, 1h, 1d, 5d, 1wk, 1mo, 3mo)
        use_cache: Use cached ticker (default: True)

    Returns:
        DataFrame with price history or None on error
    """
    try:
        ticker = get_ticker(symbol, use_cache=use_cache)
        if ticker:
            return ticker.history(period=period, interval=interval)
        return None
    except Exception as e:
        logger.error(f"Error fetching history for {symbol}: {e}")
        return None


def get_rate_limiter_stats() -> Dict[str, int]:
    """
    Get rate limiter statistics

    Returns:
        Dictionary with total_calls, throttled_calls, throttle_rate
    """
    return _rate_limiter.get_stats()


def clear_ticker_cache():
    """Clear the ticker object cache"""
    _get_cached_ticker.cache_clear()
    logger.info("YFinance ticker cache cleared")


# Convenience function for backward compatibility
def rate_limited_ticker(symbol: str) -> Optional[yf.Ticker]:
    """
    DEPRECATED: Use get_ticker() instead

    Get a rate-limited ticker object
    """
    logger.warning("rate_limited_ticker() is deprecated, use get_ticker() instead")
    return get_ticker(symbol)


if __name__ == "__main__":
    # Test the rate limiter
    import pandas as pd

    print("Testing YFinance Rate Limiter...")
    print("=" * 50)

    # Test with multiple symbols
    symbols = ['AAPL', 'GOOGL', 'MSFT', 'TSLA', 'NVDA']

    start_time = time.time()

    for symbol in symbols:
        print(f"\nFetching {symbol}...")
        ticker = get_ticker(symbol)
        if ticker:
            info = ticker.info
            print(f"  Price: ${info.get('currentPrice', 'N/A')}")
            print(f"  Market Cap: ${info.get('marketCap', 'N/A'):,}")

    elapsed = time.time() - start_time

    stats = get_rate_limiter_stats()
    print("\n" + "=" * 50)
    print("Rate Limiter Stats:")
    print(f"  Total Calls: {stats['total_calls']}")
    print(f"  Throttled Calls: {stats['throttled_calls']}")
    print(f"  Throttle Rate: {stats['throttle_rate']:.1f}%")
    print(f"  Elapsed Time: {elapsed:.2f}s")
    print(f"  Avg Time/Call: {elapsed / stats['total_calls']:.3f}s")

"""
Async Wrappers for yfinance API Calls

Provides non-blocking async wrappers for synchronous yfinance API calls.
All yfinance calls are blocking I/O - this module wraps them with
asyncio.to_thread() to prevent event loop blocking.

Features:
- Circuit breaker for resilience against yfinance failures
- Non-blocking async calls
- Automatic error handling and logging

Usage:
    from backend.infrastructure.async_yfinance import AsyncYFinance

    yf_client = AsyncYFinance()
    hist = await yf_client.get_history("AAPL", period="3mo")
    info = await yf_client.get_info("AAPL")
"""

import asyncio
import logging
from typing import Any, Dict, List, Optional
from dataclasses import dataclass
from datetime import datetime

from backend.infrastructure.circuit_breaker import yfinance_breaker

logger = logging.getLogger(__name__)


@dataclass
class StockData:
    """Container for stock data from yfinance."""
    symbol: str
    current_price: float
    historical_prices: List[float]
    iv_estimate: float
    returns: List[float]
    hist_df: Any  # pandas DataFrame


class AsyncYFinance:
    """
    Async wrapper for yfinance API.

    All methods are non-blocking and safe to use in async context.
    """

    def __init__(self) -> None:
        self._yf = None
        self._available = True

    def _ensure_import(self) -> None:
        """Lazy import yfinance."""
        if self._yf is None:
            try:
                import yfinance as yf
                self._yf = yf
            except ImportError:
                self._available = False
                raise ImportError("yfinance not available")
        return self._yf

    async def get_history(
        self,
        symbol: str,
        period: str = "3mo",
        interval: str = "1d"
    ) -> Any:
        """
        Get historical price data for a symbol.

        Args:
            symbol: Stock symbol
            period: Data period (1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max)
            interval: Data interval (1m, 2m, 5m, 15m, 30m, 60m, 90m, 1h, 1d, 5d, 1wk, 1mo, 3mo)

        Returns:
            pandas DataFrame with OHLCV data
        """
        def _fetch():
            yf = self._ensure_import()
            ticker = yf.Ticker(symbol)
            return ticker.history(period=period, interval=interval)

        # Use proper async function (not lambda) for circuit breaker
        async def _async_fetch():
            return await asyncio.to_thread(_fetch)

        return await yfinance_breaker.call(_async_fetch)

    async def get_info(self, symbol: str) -> Dict[str, Any]:
        """Get stock info/fundamentals."""
        def _fetch():
            yf = self._ensure_import()
            ticker = yf.Ticker(symbol)
            return ticker.info

        async def _async_fetch():
            return await asyncio.to_thread(_fetch)

        return await yfinance_breaker.call(_async_fetch)

    async def get_options_expirations(self, symbol: str) -> List[str]:
        """Get available options expiration dates."""
        def _fetch():
            yf = self._ensure_import()
            ticker = yf.Ticker(symbol)
            return list(ticker.options)

        return await asyncio.to_thread(_fetch)

    async def get_options_chain(
        self,
        symbol: str,
        expiration: str
    ) -> Dict[str, Any]:
        """Get options chain for a specific expiration."""
        def _fetch():
            yf = self._ensure_import()
            ticker = yf.Ticker(symbol)
            chain = ticker.option_chain(expiration)
            return {
                "calls": chain.calls,
                "puts": chain.puts
            }

        return await asyncio.to_thread(_fetch)

    async def get_stock_data(
        self,
        symbol: str,
        period: str = "3mo"
    ) -> StockData:
        """
        Get comprehensive stock data for analysis.

        Returns StockData with price, history, IV estimate, and returns.
        """
        def _fetch():
            yf = self._ensure_import()
            ticker = yf.Ticker(symbol)
            hist = ticker.history(period=period)

            if len(hist) == 0:
                raise ValueError(f"No data found for {symbol}")

            current_price = float(hist['Close'].iloc[-1])
            historical_prices = hist['Close'].tolist()

            # Calculate returns and IV estimate
            returns = hist['Close'].pct_change().dropna()
            iv_estimate = float(returns.std() * (252 ** 0.5)) * 100

            return StockData(
                symbol=symbol,
                current_price=current_price,
                historical_prices=historical_prices,
                iv_estimate=iv_estimate,
                returns=returns.tolist(),
                hist_df=hist
            )

        async def _async_fetch():
            return await asyncio.to_thread(_fetch)

        return await yfinance_breaker.call(_async_fetch)

    async def batch_get_history(
        self,
        symbols: List[str],
        period: str = "3mo",
        interval: str = "1d"
    ) -> Dict[str, Any]:
        """
        Batch fetch historical data for multiple symbols.

        More efficient than individual calls for many symbols.
        """
        def _fetch():
            yf = self._ensure_import()
            # yfinance supports downloading multiple symbols at once
            data = yf.download(
                symbols,
                period=period,
                interval=interval,
                group_by='ticker',
                threads=True
            )
            return data

        return await asyncio.to_thread(_fetch)

    async def get_current_price(self, symbol: str) -> float:
        """Get current/latest price for a symbol."""
        def _fetch():
            yf = self._ensure_import()
            ticker = yf.Ticker(symbol)
            # Use fast_info for speed
            try:
                price = ticker.fast_info.last_price
                if price is not None and price > 0:
                    return price
                raise ValueError("Invalid price from fast_info")
            except (AttributeError, KeyError, ValueError, TypeError) as e:
                # Fallback to history for expected fast_info failures
                logger.debug(f"{symbol}: fast_info failed ({e}), falling back to history")
                hist = ticker.history(period="1d")
                if len(hist) > 0:
                    return float(hist['Close'].iloc[-1])
                raise ValueError(f"No price found for {symbol}")

        return await asyncio.to_thread(_fetch)

    async def batch_get_prices(
        self,
        symbols: List[str]
    ) -> Dict[str, float]:
        """
        Get current prices for multiple symbols efficiently.
        """
        import pandas as pd  # Import at function level for availability in _fetch

        def _fetch():
            yf = self._ensure_import()
            prices = {}
            # Use download for batch efficiency
            data = yf.download(
                symbols,
                period="1d",
                interval="1d",
                threads=True
            )
            if len(symbols) == 1:
                # Single symbol returns different structure
                if len(data) > 0:
                    prices[symbols[0]] = float(data['Close'].iloc[-1])
            else:
                for symbol in symbols:
                    try:
                        if symbol in data['Close'].columns:
                            price = data['Close'][symbol].iloc[-1]
                            if not pd.isna(price):
                                prices[symbol] = float(price)
                    except (KeyError, IndexError, TypeError) as e:
                        # Expected errors for missing/invalid data
                        logger.debug(f"Batch price skip {symbol}: {e}")
                        continue
            return prices

        return await asyncio.to_thread(_fetch)

    @property
    def is_available(self) -> bool:
        """Check if yfinance is available."""
        if self._yf is None:
            try:
                self._ensure_import()
            except ImportError:
                return False
        return self._available


# =============================================================================
# Singleton Instance
# =============================================================================

import threading

_async_yf: Optional[AsyncYFinance] = None
_async_yf_lock = threading.Lock()


def get_async_yfinance() -> AsyncYFinance:
    """Get async yfinance client singleton (thread-safe)."""
    global _async_yf
    if _async_yf is None:
        with _async_yf_lock:
            # Double-check pattern for thread safety
            if _async_yf is None:
                _async_yf = AsyncYFinance()
    return _async_yf

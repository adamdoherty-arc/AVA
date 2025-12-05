"""
Async Wrappers for Robinhood API Calls

Provides non-blocking async wrappers for synchronous robin_stocks API calls.
All robin_stocks calls are blocking I/O - this module wraps them with
asyncio.to_thread() to prevent event loop blocking.

Usage:
    from backend.infrastructure.async_robinhood import AsyncRobinhood

    rh = AsyncRobinhood()
    await rh.ensure_login()
    positions = await rh.get_open_option_positions()
"""

import asyncio
import os
import logging
from typing import Any, Dict, List, Optional, Callable, TypeVar
from functools import wraps
from datetime import datetime

import robin_stocks.robinhood as rh

logger = logging.getLogger(__name__)

T = TypeVar('T')


def async_wrap(func: Callable[..., T]) -> Callable[..., T]:
    """
    Decorator that wraps a sync function to run in a thread pool.

    Usage:
        @async_wrap
        def sync_function(arg):
            return blocking_io_call(arg)

        # Now can be called with await
        result = await sync_function(arg)
    """
    @wraps(func)
    async def wrapper(*args, **kwargs) -> T:
        return await asyncio.to_thread(func, *args, **kwargs)
    return wrapper


class AsyncRobinhood:
    """
    Async wrapper for robin_stocks.robinhood API.

    All methods are non-blocking and safe to use in async context.
    """

    def __init__(self) -> None:
        self._logged_in = False
        self._login_lock = asyncio.Lock()
        self._last_login_attempt: Optional[datetime] = None

    async def ensure_login(self, force: bool = False) -> bool:
        """
        Ensure we're logged in to Robinhood.

        Uses environment variables for credentials.
        Thread-safe with asyncio lock.
        """
        async with self._login_lock:
            if self._logged_in and not force:
                return True

            try:
                username = os.getenv("ROBINHOOD_USERNAME", "")
                password = os.getenv("ROBINHOOD_PASSWORD", "")
                totp = os.getenv("ROBINHOOD_TOTP", "")

                if not username or not password:
                    logger.warning("Robinhood credentials not found in env")
                    return False

                login_result = await asyncio.to_thread(
                    lambda: rh.login(
                        username=username,
                        password=password,
                        mfa_code=totp if totp else None,
                        store_session=True
                    )
                )

                self._logged_in = bool(login_result)
                self._last_login_attempt = datetime.now()

                if self._logged_in:
                    logger.info("Robinhood login successful")
                else:
                    logger.warning("Robinhood login failed")

                return self._logged_in

            except Exception as e:
                logger.error(f"Robinhood login error: {e}")
                return False

    # =========================================================================
    # Account Information
    # =========================================================================

    async def get_account_profile(self) -> Dict[str, Any]:
        """Get account profile information."""
        return await asyncio.to_thread(rh.load_account_profile)

    async def get_portfolio_profile(self) -> Dict[str, Any]:
        """Get portfolio profile."""
        return await asyncio.to_thread(rh.load_portfolio_profile)

    async def get_phoenix_account(self) -> Dict[str, Any]:
        """Get phoenix account info (unified account data)."""
        return await asyncio.to_thread(rh.load_phoenix_account)

    # =========================================================================
    # Stock Positions
    # =========================================================================

    async def get_stock_positions(self) -> List[Dict[str, Any]]:
        """Get all stock positions."""
        return await asyncio.to_thread(rh.get_open_stock_positions)

    async def get_stock_position(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Get position for a specific stock."""
        return await asyncio.to_thread(
            lambda: rh.get_open_stock_positions(symbol)
        )

    # =========================================================================
    # Option Positions
    # =========================================================================

    async def get_open_option_positions(self) -> List[Dict[str, Any]]:
        """Get all open option positions."""
        return await asyncio.to_thread(rh.get_open_option_positions)

    async def get_option_instrument_data(
        self,
        option_id: str
    ) -> Optional[Dict[str, Any]]:
        """Get instrument data for an option by ID."""
        return await asyncio.to_thread(
            lambda: rh.get_option_instrument_data_by_id(option_id)
        )

    async def get_option_market_data(
        self,
        option_id: str
    ) -> Optional[List[Dict[str, Any]]]:
        """Get market data for an option by ID."""
        return await asyncio.to_thread(
            lambda: rh.get_option_market_data_by_id(option_id)
        )

    async def get_option_market_data_by_symbols(
        self,
        symbols: List[str]
    ) -> List[Dict[str, Any]]:
        """Get market data for multiple option symbols."""
        return await asyncio.to_thread(
            lambda: rh.get_option_market_data(*symbols)
        )

    # =========================================================================
    # Quotes and Prices
    # =========================================================================

    async def get_quotes(
        self,
        symbols: List[str]
    ) -> List[Dict[str, Any]]:
        """Get quotes for multiple symbols."""
        return await asyncio.to_thread(
            lambda: rh.get_quotes(symbols)
        )

    async def get_latest_price(self, symbol: str) -> Optional[float]:
        """Get latest price for a single symbol."""
        prices = await asyncio.to_thread(
            lambda: rh.get_latest_price(symbol)
        )
        if prices and len(prices) > 0 and prices[0]:
            return float(prices[0])
        return None

    async def get_fundamentals(
        self,
        symbols: List[str]
    ) -> List[Dict[str, Any]]:
        """Get fundamental data for symbols."""
        return await asyncio.to_thread(
            lambda: rh.get_fundamentals(symbols)
        )

    # =========================================================================
    # Historical Data
    # =========================================================================

    async def get_stock_historicals(
        self,
        symbols: List[str],
        interval: str = "day",
        span: str = "month"
    ) -> List[Dict[str, Any]]:
        """
        Get historical data for stocks.

        Args:
            symbols: List of stock symbols
            interval: 5minute, 10minute, hour, day, week
            span: day, week, month, 3month, year, 5year
        """
        return await asyncio.to_thread(
            lambda: rh.get_stock_historicals(
                symbols,
                interval=interval,
                span=span
            )
        )

    # =========================================================================
    # Option Chains
    # =========================================================================

    async def find_options_by_expiration(
        self,
        symbol: str,
        expiration_date: str,
        option_type: str = "both"
    ) -> List[Dict[str, Any]]:
        """
        Find options by expiration date.

        Args:
            symbol: Stock symbol
            expiration_date: Date string YYYY-MM-DD
            option_type: "call", "put", or "both"
        """
        return await asyncio.to_thread(
            lambda: rh.find_options_by_expiration(
                symbol,
                expirationDate=expiration_date,
                optionType=option_type
            )
        )

    async def find_options_by_strike(
        self,
        symbol: str,
        strike_price: float,
        option_type: str = "both"
    ) -> List[Dict[str, Any]]:
        """Find options by strike price."""
        return await asyncio.to_thread(
            lambda: rh.find_options_by_strike(
                symbol,
                strike=strike_price,
                optionType=option_type
            )
        )

    async def get_chains(self, symbol: str) -> Dict[str, Any]:
        """Get full option chain info for symbol."""
        return await asyncio.to_thread(
            lambda: rh.get_chains(symbol)
        )

    # =========================================================================
    # Batch Operations (Parallel)
    # =========================================================================

    async def batch_get_option_data(
        self,
        option_ids: List[str],
        max_concurrent: int = 5
    ) -> Dict[str, Dict[str, Any]]:
        """
        Fetch instrument and market data for multiple options in parallel.

        More efficient than sequential calls.
        Returns dict keyed by option_id.
        """
        semaphore = asyncio.Semaphore(max_concurrent)

        async def fetch_one(opt_id: str) -> tuple:
            async with semaphore:
                try:
                    instrument = await self.get_option_instrument_data(opt_id)
                    market = await self.get_option_market_data(opt_id)
                    return (opt_id, {
                        "instrument": instrument or {},
                        "market_data": market[0] if market else {}
                    })
                except Exception as e:
                    logger.error(f"Error fetching option {opt_id}: {e}")
                    return (opt_id, None)

        tasks = [fetch_one(oid) for oid in option_ids]
        results = await asyncio.gather(*tasks)

        return {
            opt_id: data
            for opt_id, data in results
            if data is not None
        }

    async def batch_get_quotes(
        self,
        symbols: List[str],
        chunk_size: int = 50
    ) -> Dict[str, Dict[str, Any]]:
        """
        Get quotes for many symbols, chunked to avoid API limits.

        Robinhood allows ~50 symbols per request.
        """
        results = {}

        # Chunk symbols
        chunks = [
            symbols[i:i + chunk_size]
            for i in range(0, len(symbols), chunk_size)
        ]

        for chunk in chunks:
            quotes = await self.get_quotes(chunk)
            for quote in quotes:
                if quote:
                    symbol = quote.get("symbol")
                    if symbol:
                        results[symbol] = quote

        return results


# =============================================================================
# Singleton Instance
# =============================================================================

_async_rh: Optional[AsyncRobinhood] = None


def get_async_robinhood() -> AsyncRobinhood:
    """Get async Robinhood client singleton."""
    global _async_rh
    if _async_rh is None:
        _async_rh = AsyncRobinhood()
    return _async_rh

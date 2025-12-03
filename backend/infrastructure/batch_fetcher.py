"""
Parallel Batch Fetching Infrastructure

Provides:
- Parallel API call batching for YFinance, Robinhood
- Automatic retry with exponential backoff
- Progress tracking
- Result aggregation

Solves N+1 query problem by fetching multiple items in parallel.
"""

import asyncio
import logging
from typing import Any, Callable, Dict, List, Optional, TypeVar
from dataclasses import dataclass
from datetime import datetime

logger = logging.getLogger(__name__)

T = TypeVar('T')


@dataclass
class BatchResult:
    """Result of a batch fetch operation"""
    successful: Dict[str, Any]
    failed: Dict[str, str]
    total_time_ms: float

    @property
    def success_rate(self) -> float:
        total = len(self.successful) + len(self.failed)
        return (len(self.successful) / total * 100) if total > 0 else 0.0


class ParallelBatchFetcher:
    """
    Fetches multiple items in parallel with rate limiting.

    Features:
    - Configurable concurrency
    - Automatic retry with backoff
    - Timeout handling
    - Progress callbacks

    Usage:
        fetcher = ParallelBatchFetcher(max_concurrent=10)

        results = await fetcher.fetch_batch(
            items=["AAPL", "GOOGL", "MSFT"],
            fetch_func=get_stock_metadata,
            key_func=lambda x: x
        )
    """

    def __init__(
        self,
        max_concurrent: int = 10,
        retry_count: int = 2,
        retry_delay: float = 1.0,
        timeout: float = 30.0
    ):
        self.max_concurrent = max_concurrent
        self.retry_count = retry_count
        self.retry_delay = retry_delay
        self.timeout = timeout

    async def fetch_batch(
        self,
        items: List[Any],
        fetch_func: Callable[[Any], T],
        key_func: Optional[Callable[[Any], str]] = None,
        progress_callback: Optional[Callable[[int, int], None]] = None
    ) -> BatchResult:
        """
        Fetch multiple items in parallel.

        Args:
            items: List of items to fetch
            fetch_func: Async or sync function to fetch each item
            key_func: Function to extract key from item (default: str)
            progress_callback: Called with (completed, total) after each item

        Returns:
            BatchResult with successful and failed items
        """
        start_time = datetime.now()
        key_func = key_func or str

        # Create semaphore for concurrency control
        semaphore = asyncio.Semaphore(self.max_concurrent)

        successful: Dict[str, Any] = {}
        failed: Dict[str, str] = {}
        completed = 0
        lock = asyncio.Lock()

        async def fetch_one(item: Any) -> tuple[str, Any, Optional[str]]:
            """Fetch single item with retry"""
            nonlocal completed
            key = key_func(item)

            for attempt in range(self.retry_count + 1):
                try:
                    async with semaphore:
                        # Handle both sync and async functions
                        if asyncio.iscoroutinefunction(fetch_func):
                            result = await asyncio.wait_for(
                                fetch_func(item),
                                timeout=self.timeout
                            )
                        else:
                            result = await asyncio.wait_for(
                                asyncio.to_thread(fetch_func, item),
                                timeout=self.timeout
                            )

                        async with lock:
                            completed += 1
                            if progress_callback:
                                progress_callback(completed, len(items))

                        return (key, result, None)

                except asyncio.TimeoutError:
                    error = f"Timeout after {self.timeout}s"
                except Exception as e:
                    error = str(e)

                # Retry with backoff
                if attempt < self.retry_count:
                    await asyncio.sleep(self.retry_delay * (2 ** attempt))

            async with lock:
                completed += 1
                if progress_callback:
                    progress_callback(completed, len(items))

            return (key, None, error)

        # Run all fetches in parallel
        tasks = [fetch_one(item) for item in items]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Aggregate results
        for result in results:
            if isinstance(result, Exception):
                logger.error(f"Batch fetch exception: {result}")
                continue

            key, value, error = result
            if error:
                failed[key] = error
            else:
                successful[key] = value

        total_time = (datetime.now() - start_time).total_seconds() * 1000

        logger.info(
            f"Batch fetch completed: {len(successful)}/{len(items)} successful "
            f"in {total_time:.0f}ms"
        )

        return BatchResult(
            successful=successful,
            failed=failed,
            total_time_ms=total_time
        )


# =============================================================================
# Specialized Batch Fetchers
# =============================================================================

class YFinanceBatchFetcher:
    """
    Optimized batch fetcher for YFinance data.

    Uses yfinance's built-in multi-symbol support when possible.
    """

    def __init__(self, max_concurrent: int = 20):
        self._fetcher = ParallelBatchFetcher(
            max_concurrent=max_concurrent,
            retry_count=2,
            retry_delay=0.5,
            timeout=10.0
        )
        self._yf = None  # Lazy-loaded yfinance module

    @property
    def yf(self):
        """Lazy-load yfinance module once."""
        if self._yf is None:
            import yfinance
            self._yf = yfinance
        return self._yf

    async def fetch_metadata_batch(
        self,
        symbols: List[str],
        progress_callback: Optional[Callable[[int, int], None]] = None
    ) -> BatchResult:
        """
        Fetch metadata for multiple symbols in parallel.

        Args:
            symbols: List of ticker symbols
            progress_callback: Progress update callback

        Returns:
            BatchResult with metadata dictionaries
        """
        yf = self.yf  # Use lazy-loaded module

        def fetch_single(symbol: str) -> Dict[str, Any]:
            """Fetch single ticker metadata"""
            ticker = yf.Ticker(symbol)
            info = ticker.info

            # Extract key fields
            return {
                "symbol": symbol,
                "name": info.get("shortName", symbol),
                "sector": info.get("sector", "Unknown"),
                "industry": info.get("industry", "Unknown"),
                "market_cap": info.get("marketCap"),
                "pe_ratio": info.get("trailingPE"),
                "forward_pe": info.get("forwardPE"),
                "dividend_yield": info.get("dividendYield"),
                "beta": info.get("beta"),
                "52w_high": info.get("fiftyTwoWeekHigh"),
                "52w_low": info.get("fiftyTwoWeekLow"),
                "current_price": info.get("regularMarketPrice"),
                "analyst_target": info.get("targetMeanPrice"),
                "analyst_rating": info.get("recommendationKey"),
                "fetched_at": datetime.now().isoformat()
            }

        return await self._fetcher.fetch_batch(
            items=symbols,
            fetch_func=fetch_single,
            key_func=lambda x: x.upper(),
            progress_callback=progress_callback
        )

    async def fetch_prices_batch(
        self,
        symbols: List[str]
    ) -> Dict[str, float]:
        """
        Fetch current prices for multiple symbols efficiently.

        Uses yfinance download() for bulk fetching.
        """
        yf = self.yf  # Use lazy-loaded module

        try:
            # Use bulk download
            data = await asyncio.to_thread(
                lambda: yf.download(
                    " ".join(symbols),
                    period="1d",
                    progress=False,
                    threads=True
                )
            )

            prices = {}
            if len(symbols) == 1:
                # Single symbol returns different structure
                if not data.empty:
                    prices[symbols[0]] = float(data["Close"].iloc[-1])
            else:
                # Multiple symbols
                if "Close" in data.columns:
                    for symbol in symbols:
                        if symbol in data["Close"].columns:
                            price = data["Close"][symbol].iloc[-1]
                            if not pd.isna(price):
                                prices[symbol] = float(price)

            return prices

        except Exception as e:
            logger.error(f"Bulk price fetch failed: {e}")
            return {}


class RobinhoodBatchFetcher:
    """
    Optimized batch fetcher for Robinhood data.

    Uses Robinhood's batch endpoints where available.
    """

    def __init__(self, max_concurrent: int = 5):
        # Lower concurrency for Robinhood rate limits
        self._fetcher = ParallelBatchFetcher(
            max_concurrent=max_concurrent,
            retry_count=1,
            retry_delay=2.0,
            timeout=15.0
        )

    async def fetch_option_market_data(
        self,
        option_ids: List[str]
    ) -> BatchResult:
        """
        Fetch market data for multiple options.
        """
        import robin_stocks.robinhood as rh

        def fetch_single(opt_id: str) -> Dict[str, Any]:
            data = rh.get_option_market_data_by_id(opt_id)
            if data and len(data) > 0:
                return data[0]
            return {}

        return await self._fetcher.fetch_batch(
            items=option_ids,
            fetch_func=fetch_single,
            key_func=lambda x: x
        )

    async def fetch_option_instrument_data(
        self,
        option_ids: List[str]
    ) -> BatchResult:
        """
        Fetch instrument data for multiple options in parallel.

        Returns option chain info (symbol, strike, expiration, type).
        """
        import robin_stocks.robinhood as rh

        def fetch_single(opt_id: str) -> Dict[str, Any]:
            return rh.get_option_instrument_data_by_id(opt_id) or {}

        return await self._fetcher.fetch_batch(
            items=option_ids,
            fetch_func=fetch_single,
            key_func=lambda x: x
        )

    async def fetch_all_option_data(
        self,
        option_ids: List[str]
    ) -> Dict[str, Dict[str, Any]]:
        """
        Fetch both instrument and market data for options in parallel.

        Returns combined data keyed by option_id.
        More efficient than sequential calls.
        """
        import robin_stocks.robinhood as rh
        from datetime import datetime

        # Fetch instrument and market data in parallel
        instrument_task = self.fetch_option_instrument_data(option_ids)
        market_task = self.fetch_option_market_data(option_ids)

        instrument_result, market_result = await asyncio.gather(
            instrument_task, market_task
        )

        # Combine results
        combined = {}
        for opt_id in option_ids:
            instrument = instrument_result.successful.get(opt_id, {})
            market = market_result.successful.get(opt_id, {})

            if instrument:
                combined[opt_id] = {
                    "instrument": instrument,
                    "market_data": market,
                    "fetched_at": datetime.now().isoformat()
                }

        logger.info(
            f"Batch option fetch: {len(combined)}/{len(option_ids)} complete "
            f"(instrument: {instrument_result.success_rate:.0f}%, "
            f"market: {market_result.success_rate:.0f}%)"
        )

        return combined

    async def fetch_instruments_by_url(
        self,
        instrument_urls: List[str]
    ) -> Dict[str, Dict[str, Any]]:
        """
        Fetch instrument data for multiple stock instrument URLs in parallel.

        Args:
            instrument_urls: List of Robinhood instrument URLs

        Returns:
            Dict mapping URL to instrument data
        """
        import robin_stocks.robinhood as rh

        def fetch_single(url: str) -> Dict[str, Any]:
            return rh.get_instrument_by_url(url) or {}

        result = await self._fetcher.fetch_batch(
            items=instrument_urls,
            fetch_func=fetch_single,
            key_func=lambda x: x
        )

        return result.successful

    async def fetch_latest_prices(
        self,
        symbols: List[str]
    ) -> Dict[str, float]:
        """
        Fetch latest prices for multiple symbols.

        Uses Robinhood's batch quote endpoint.
        """
        import robin_stocks.robinhood as rh

        try:
            # Use batch quote - more efficient than individual calls
            quotes = await asyncio.to_thread(
                lambda: rh.get_quotes(symbols)
            )

            prices = {}
            for quote in quotes:
                if quote:
                    symbol = quote.get("symbol")
                    price = quote.get("last_trade_price") or quote.get("last_extended_hours_trade_price")
                    if symbol and price:
                        prices[symbol] = float(price)

            return prices

        except Exception as e:
            logger.error(f"Robinhood batch price fetch failed: {e}")
            return {}


# =============================================================================
# Convenience Functions
# =============================================================================

_yf_fetcher: Optional[YFinanceBatchFetcher] = None
_rh_fetcher: Optional[RobinhoodBatchFetcher] = None


def get_yfinance_fetcher() -> YFinanceBatchFetcher:
    """Get YFinance batch fetcher singleton"""
    global _yf_fetcher
    if _yf_fetcher is None:
        _yf_fetcher = YFinanceBatchFetcher()
    return _yf_fetcher


def get_robinhood_fetcher() -> RobinhoodBatchFetcher:
    """Get Robinhood batch fetcher singleton"""
    global _rh_fetcher
    if _rh_fetcher is None:
        _rh_fetcher = RobinhoodBatchFetcher()
    return _rh_fetcher


# Import pandas for price batch (optional)
try:
    import pandas as pd
except ImportError:
    pd = None

"""
Backend Utilities Module

Modern async utilities for non-blocking I/O operations.
"""

from backend.utils.async_market_data import (
    async_get_ticker,
    async_get_history,
    async_get_info,
    async_get_options_dates,
    async_get_options_chain,
    async_get_financials,
    async_batch_tickers,
    async_batch_history,
    async_batch_info,
    async_get_current_price,
    async_get_earnings_dates,
    async_get_recommendations,
    AsyncMarketDataCache,
    market_data_cache
)

from backend.utils.async_file_io import (
    async_read_file,
    async_write_file,
    async_read_json,
    async_write_json,
    async_file_exists,
    async_list_dir
)

__all__ = [
    # Market data
    "async_get_ticker",
    "async_get_history",
    "async_get_info",
    "async_get_options_dates",
    "async_get_options_chain",
    "async_get_financials",
    "async_batch_tickers",
    "async_batch_history",
    "async_batch_info",
    "async_get_current_price",
    "async_get_earnings_dates",
    "async_get_recommendations",
    "AsyncMarketDataCache",
    "market_data_cache",
    # File I/O
    "async_read_file",
    "async_write_file",
    "async_read_json",
    "async_write_json",
    "async_file_exists",
    "async_list_dir"
]

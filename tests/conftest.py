"""
Shared pytest fixtures for Magnus test suite.

This file provides common fixtures used across multiple test modules.
"""

import pytest
import asyncio
from unittest.mock import Mock, MagicMock, AsyncMock, patch
from typing import Dict, List, Any, Optional
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ============================================================================
# ASYNC EVENT LOOP
# ============================================================================

@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the event loop for the test session."""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


# ============================================================================
# MOCK YFINANCE FIXTURES
# ============================================================================

@pytest.fixture
def mock_ticker_info() -> Dict[str, Any]:
    """Mock yfinance ticker.info response."""
    return {
        'currentPrice': 150.50,
        'regularMarketPrice': 150.50,
        'symbol': 'AAPL',
        'shortName': 'Apple Inc.',
        'sector': 'Technology',
        'industry': 'Consumer Electronics',
        'marketCap': 2500000000000,
    }


@pytest.fixture
def mock_ticker_info_low_price() -> Dict[str, Any]:
    """Mock yfinance ticker.info response for low price stock."""
    return {
        'currentPrice': 5.25,
        'regularMarketPrice': 5.25,
        'symbol': 'F',
        'shortName': 'Ford Motor Company',
        'sector': 'Consumer Cyclical',
        'industry': 'Auto Manufacturers',
        'marketCap': 42000000000,
    }


@pytest.fixture
def mock_options_chain():
    """Mock yfinance options chain response."""
    import pandas as pd

    puts_data = {
        'strike': [145.0, 150.0, 155.0],
        'bid': [2.50, 4.00, 6.50],
        'ask': [2.75, 4.25, 6.75],
        'volume': [100, 250, 50],
        'openInterest': [500, 1200, 300],
        'impliedVolatility': [0.35, 0.32, 0.30],
    }

    calls_data = {
        'strike': [155.0, 160.0, 165.0],
        'bid': [3.00, 2.00, 1.00],
        'ask': [3.25, 2.25, 1.25],
        'volume': [150, 200, 75],
        'openInterest': [600, 800, 400],
        'impliedVolatility': [0.33, 0.30, 0.28],
    }

    class MockOptionsChain:
        puts = pd.DataFrame(puts_data)
        calls = pd.DataFrame(calls_data)

    return MockOptionsChain()


@pytest.fixture
def mock_yfinance_ticker(mock_ticker_info, mock_options_chain):
    """Create a fully mocked yfinance Ticker object."""
    ticker = Mock()
    ticker.info = mock_ticker_info
    ticker.options = ['2025-01-17', '2025-02-21', '2025-03-21']
    ticker.option_chain = Mock(return_value=mock_options_chain)
    return ticker


# ============================================================================
# DATABASE FIXTURES
# ============================================================================

@pytest.fixture
def mock_db_connection():
    """Mock async database connection."""
    conn = AsyncMock()
    conn.fetch = AsyncMock(return_value=[])
    conn.fetchrow = AsyncMock(return_value=None)
    conn.fetchval = AsyncMock(return_value=0)
    conn.execute = AsyncMock(return_value="INSERT 0 1")
    return conn


@pytest.fixture
def mock_database_manager(mock_db_connection):
    """Mock AsyncDatabaseManager."""
    db = AsyncMock()
    db.fetch = mock_db_connection.fetch
    db.fetchrow = mock_db_connection.fetchrow
    db.fetchval = mock_db_connection.fetchval
    db.execute = mock_db_connection.execute
    db.transaction = MagicMock()
    return db


@pytest.fixture
def sample_premium_opportunities() -> List[Dict[str, Any]]:
    """Sample premium opportunities data from database."""
    return [
        {
            'symbol': 'AAPL',
            'company_name': 'Apple Inc.',
            'option_type': 'PUT',
            'strike': 145.0,
            'expiration': '2025-01-17',
            'dte': 30,
            'stock_price': 150.50,
            'bid': 2.50,
            'ask': 2.75,
            'mid': 2.625,
            'premium': 2.625,
            'premium_pct': 1.81,
            'annualized_return': 22.0,
            'monthly_return': 1.81,
            'delta': -0.25,
            'gamma': 0.02,
            'theta': -0.05,
            'vega': 0.30,
            'implied_volatility': 0.35,
            'volume': 250,
            'open_interest': 1200,
            'break_even': 142.375,
            'pop': 0.75,
            'last_updated': '2025-12-05T10:00:00Z',
        },
        {
            'symbol': 'F',
            'company_name': 'Ford Motor Company',
            'option_type': 'PUT',
            'strike': 10.0,
            'expiration': '2025-01-17',
            'dte': 30,
            'stock_price': 10.50,
            'bid': 0.20,
            'ask': 0.25,
            'mid': 0.225,
            'premium': 0.225,
            'premium_pct': 2.25,
            'annualized_return': 27.0,
            'monthly_return': 2.25,
            'delta': -0.30,
            'gamma': 0.05,
            'theta': -0.02,
            'vega': 0.10,
            'implied_volatility': 0.45,
            'volume': 500,
            'open_interest': 2000,
            'break_even': 9.775,
            'pop': 0.70,
            'last_updated': '2025-12-05T10:00:00Z',
        },
    ]


@pytest.fixture
def sample_scan_history() -> List[Dict[str, Any]]:
    """Sample scan history data."""
    return [
        {
            'scan_id': 'scan_abc123',
            'symbols': ['AAPL', 'MSFT', 'NVDA'],
            'symbol_count': 3,
            'dte': 30,
            'max_price': 500.0,
            'min_premium_pct': 1.0,
            'result_count': 15,
            'created_at': '2025-12-05T09:00:00Z',
        },
        {
            'scan_id': 'scan_def456',
            'symbols': ['F', 'GM', 'RIVN'],
            'symbol_count': 3,
            'dte': 14,
            'max_price': 50.0,
            'min_premium_pct': 2.0,
            'result_count': 8,
            'created_at': '2025-12-05T08:00:00Z',
        },
    ]


# ============================================================================
# FASTAPI TEST CLIENT FIXTURES
# ============================================================================

@pytest.fixture
def test_client():
    """Create FastAPI test client."""
    from fastapi.testclient import TestClient
    from backend.main import app

    return TestClient(app)


@pytest.fixture
async def async_test_client():
    """Create async HTTP client for testing."""
    import httpx

    async with httpx.AsyncClient(base_url="http://test") as client:
        yield client


# ============================================================================
# SCANNER FIXTURES
# ============================================================================

@pytest.fixture
def fresh_circuit_breaker():
    """Create a fresh CircuitBreaker instance for testing."""
    from src.premium_scanner_v2 import CircuitBreaker
    return CircuitBreaker(failure_threshold=5, reset_timeout=30)


@pytest.fixture
def fresh_rate_limiter():
    """Create a fresh RateLimiter instance for testing."""
    from src.premium_scanner_v2 import RateLimiter
    return RateLimiter(min_interval=0.1)


@pytest.fixture
def fresh_symbol_cache():
    """Create a fresh SymbolCache instance for testing."""
    from src.premium_scanner_v2 import SymbolCache
    return SymbolCache(ttl_seconds=60)


@pytest.fixture
def fresh_scanner():
    """Create a fresh PremiumScannerV2 instance for testing."""
    from src.premium_scanner_v2 import PremiumScannerV2
    return PremiumScannerV2(
        max_workers=5,
        symbol_timeout=10,
        min_volume=50,
        min_open_interest=25
    )


# ============================================================================
# WATCHLIST FIXTURES
# ============================================================================

@pytest.fixture
def sample_watchlists() -> List[Dict[str, Any]]:
    """Sample watchlist data."""
    return [
        {
            'id': 'popular',
            'name': 'Popular Stocks',
            'source': 'predefined',
            'symbols': ['AAPL', 'MSFT', 'NVDA', 'TSLA', 'AMD'],
        },
        {
            'id': 'tech',
            'name': 'Tech Leaders',
            'source': 'predefined',
            'symbols': ['GOOGL', 'META', 'AMZN', 'NFLX', 'CRM'],
        },
        {
            'id': 'tradingview-main',
            'name': 'TradingView Watchlist',
            'source': 'tradingview',
            'symbols': ['SPY', 'QQQ', 'IWM'],
        },
    ]


# ============================================================================
# TIME MANIPULATION
# ============================================================================

@pytest.fixture
def freeze_time():
    """Fixture to freeze time for deterministic testing."""
    import time
    original_time = time.time
    frozen_time = 1733400000.0  # Fixed timestamp

    def mock_time():
        return frozen_time

    time.time = mock_time
    yield mock_time, frozen_time
    time.time = original_time


# ============================================================================
# CLEANUP
# ============================================================================

@pytest.fixture(autouse=True)
def reset_singletons():
    """Reset singleton instances between tests."""
    yield
    # Reset scanner singleton
    import src.premium_scanner_v2 as scanner_module
    scanner_module._scanner_v2 = None
    # Reset circuit breaker and rate limiter
    scanner_module._yfinance_circuit_breaker.failures = 0
    scanner_module._yfinance_circuit_breaker.state = "closed"
    scanner_module._yfinance_circuit_breaker.last_failure_time = None
    scanner_module._yfinance_rate_limiter.last_call_time = 0

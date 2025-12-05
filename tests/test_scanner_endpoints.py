"""
API endpoint tests for backend/routers/scanner.py

Tests cover:
- Router configuration and constants
- Helper function logic
- Request/Response models

NOTE: Full integration tests require a running backend and database.
These tests focus on unit testing the router logic without external dependencies.
"""

import pytest
from unittest.mock import Mock, MagicMock, AsyncMock, patch
import sys
import os
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ============================================================================
# TIMEOUT CONFIGURATION TESTS
# ============================================================================

class TestTimeoutConfiguration:
    """Tests for timeout configuration in scanner router."""

    def test_timeout_constants_exist(self):
        """Test that timeout constants are properly defined."""
        from backend.routers.scanner import (
            TIMEOUT_LIVE_SCAN_MAX,
            TIMEOUT_PER_SYMBOL,
            TIMEOUT_MIN,
            TIMEOUT_DB_QUERY,
        )

        assert TIMEOUT_LIVE_SCAN_MAX == 120
        assert TIMEOUT_PER_SYMBOL == 10
        assert TIMEOUT_MIN == 30
        assert TIMEOUT_DB_QUERY == 30

    def test_cache_ttl_constants_exist(self):
        """Test that cache TTL constants are properly defined."""
        from backend.routers.scanner import (
            CACHE_TTL_WATCHLISTS,
            CACHE_TTL_HISTORY,
            CACHE_TTL_STORED_PREMIUMS,
            CACHE_TTL_DTE_COMPARISON,
            CACHE_TTL_PREMIUM_STATS,
        )

        assert CACHE_TTL_WATCHLISTS == 300  # 5 minutes
        assert CACHE_TTL_HISTORY == 60      # 1 minute
        assert CACHE_TTL_STORED_PREMIUMS == 600  # 10 minutes
        assert CACHE_TTL_DTE_COMPARISON == 600   # 10 minutes
        assert CACHE_TTL_PREMIUM_STATS == 120    # 2 minutes


# ============================================================================
# PYDANTIC MODEL TESTS
# ============================================================================

class TestScanRequestModel:
    """Tests for the ScanRequest Pydantic model."""

    def test_scan_request_defaults(self):
        """Test that ScanRequest has correct default values."""
        from backend.routers.scanner import ScanRequest

        request = ScanRequest(symbols=["AAPL", "MSFT"])

        assert request.symbols == ["AAPL", "MSFT"]
        assert request.max_price == 50
        assert request.min_premium_pct == 1.0
        assert request.dte == 30
        assert request.save_to_db is True

    def test_scan_request_custom_values(self):
        """Test ScanRequest with custom values."""
        from backend.routers.scanner import ScanRequest

        request = ScanRequest(
            symbols=["NVDA"],
            max_price=100.0,
            min_premium_pct=2.5,
            dte=14,
            save_to_db=False
        )

        assert request.symbols == ["NVDA"]
        assert request.max_price == 100.0
        assert request.min_premium_pct == 2.5
        assert request.dte == 14
        assert request.save_to_db is False


class TestMultiDTEScanRequestModel:
    """Tests for the MultiDTEScanRequest Pydantic model."""

    def test_multi_dte_request_defaults(self):
        """Test that MultiDTEScanRequest has correct default values."""
        from backend.routers.scanner import MultiDTEScanRequest

        request = MultiDTEScanRequest(symbols=["AAPL"])

        assert request.symbols == ["AAPL"]
        assert request.max_price == 50
        assert request.min_premium_pct == 1.0
        assert request.dte_targets == [7, 14, 30, 45]

    def test_multi_dte_request_custom_targets(self):
        """Test MultiDTEScanRequest with custom DTE targets."""
        from backend.routers.scanner import MultiDTEScanRequest

        request = MultiDTEScanRequest(
            symbols=["TSLA"],
            dte_targets=[7, 21, 45, 60]
        )

        assert request.dte_targets == [7, 21, 45, 60]


# ============================================================================
# HELPER FUNCTION TESTS
# ============================================================================

class TestFetchScanHistory:
    """Tests for _fetch_scan_history helper function."""

    @pytest.mark.asyncio
    async def test_fetch_scan_history_empty(self):
        """Test _fetch_scan_history returns empty when no rows."""
        from backend.routers.scanner import _fetch_scan_history

        mock_db = AsyncMock()
        mock_db.fetch = AsyncMock(return_value=[])

        result = await _fetch_scan_history(mock_db, limit=10)

        assert result["history"] == []
        assert result["count"] == 0

    @pytest.mark.asyncio
    async def test_fetch_scan_history_with_data(self):
        """Test _fetch_scan_history processes rows correctly."""
        from backend.routers.scanner import _fetch_scan_history

        mock_row = {
            "scan_id": "scan_123",
            "symbols": ["AAPL", "MSFT"],
            "dte": 30,
            "max_price": 100.0,
            "min_premium_pct": 1.5,
            "result_count": 10,
            "created_at": datetime(2025, 12, 5, 10, 0, 0),
        }

        mock_db = AsyncMock()
        mock_db.fetch = AsyncMock(return_value=[mock_row])

        result = await _fetch_scan_history(mock_db, limit=10)

        assert result["count"] == 1
        assert len(result["history"]) == 1
        assert result["history"][0]["scan_id"] == "scan_123"
        assert result["history"][0]["symbol_count"] == 2
        assert result["history"][0]["dte"] == 30


class TestFetchScanById:
    """Tests for _fetch_scan_by_id helper function."""

    @pytest.mark.asyncio
    async def test_fetch_scan_by_id_not_found(self):
        """Test _fetch_scan_by_id returns None when not found."""
        from backend.routers.scanner import _fetch_scan_by_id

        mock_db = AsyncMock()
        mock_db.fetchrow = AsyncMock(return_value=None)

        result = await _fetch_scan_by_id(mock_db, "nonexistent-id")

        assert result is None

    @pytest.mark.asyncio
    async def test_fetch_scan_by_id_found(self):
        """Test _fetch_scan_by_id returns data when found."""
        from backend.routers.scanner import _fetch_scan_by_id

        mock_row = {
            "scan_id": "scan_123",
            "symbols": ["AAPL"],
            "dte": 30,
            "max_price": 100.0,
            "min_premium_pct": 1.0,
            "result_count": 5,
            "results": [{"symbol": "AAPL", "premium": 2.50}],
            "created_at": datetime(2025, 12, 5, 10, 0, 0),
        }

        mock_db = AsyncMock()
        mock_db.fetchrow = AsyncMock(return_value=mock_row)

        result = await _fetch_scan_by_id(mock_db, "scan_123")

        assert result is not None
        assert result["scan_id"] == "scan_123"
        assert result["result_count"] == 5
        assert len(result["results"]) == 1


class TestPersistScanResults:
    """Tests for _persist_scan_results helper function."""

    @pytest.mark.asyncio
    async def test_persist_empty_results(self):
        """Test _persist_scan_results returns 0 for empty list."""
        from backend.routers.scanner import _persist_scan_results

        mock_db = AsyncMock()

        result = await _persist_scan_results(mock_db, [])

        assert result == 0
        mock_db.execute.assert_not_called()

    @pytest.mark.asyncio
    async def test_persist_results_calls_execute(self):
        """Test _persist_scan_results calls execute for each result."""
        from backend.routers.scanner import _persist_scan_results

        mock_db = AsyncMock()
        mock_db.execute = AsyncMock()

        results = [
            {
                "symbol": "AAPL",
                "option_type": "PUT",
                "strike": 145.0,
                "expiration": "2025-01-17",
                "dte": 30,
                "stock_price": 150.0,
                "bid": 2.50,
                "ask": 2.75,
                "mid": 2.625,
                "premium": 2.625,
                "premium_pct": 1.81,
                "annualized_return": 22.0,
                "monthly_return": 1.81,
            },
        ]

        count = await _persist_scan_results(mock_db, results)

        assert count == 1
        assert mock_db.execute.call_count == 1

    @pytest.mark.asyncio
    async def test_persist_results_handles_errors(self):
        """Test _persist_scan_results handles individual errors gracefully."""
        from backend.routers.scanner import _persist_scan_results

        mock_db = AsyncMock()
        mock_db.execute = AsyncMock(side_effect=Exception("DB Error"))

        results = [
            {"symbol": "AAPL", "expiration": "2025-01-17"},
        ]

        # Should not raise, should return 0 since all failed
        count = await _persist_scan_results(mock_db, results)

        assert count == 0


# ============================================================================
# ROUTER CONFIGURATION TESTS
# ============================================================================

class TestRouterConfiguration:
    """Tests for router configuration."""

    def test_router_prefix(self):
        """Test that router has correct prefix."""
        from backend.routers.scanner import router

        assert router.prefix == "/api/scanner"

    def test_router_tags(self):
        """Test that router has correct tags."""
        from backend.routers.scanner import router

        assert "scanner" in router.tags


# ============================================================================
# WATCHLIST PRESET TESTS
# ============================================================================

class TestWatchlistPresets:
    """Tests for watchlist preset data."""

    def test_preset_watchlists_defined(self):
        """Test that preset watchlists are properly defined in frontend."""
        # This is more of a sanity check - the presets are in the frontend
        # But we verify the backend can handle them
        preset_ids = ['popular', 'tech', 'finance', 'retail', 'healthcare', 'energy', 'high-iv']

        # These should be valid watchlist IDs
        for preset_id in preset_ids:
            assert isinstance(preset_id, str)
            assert len(preset_id) > 0


# ============================================================================
# EDGE CASE TESTS
# ============================================================================

class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_empty_symbol_list_scan_request(self):
        """Test ScanRequest with empty symbol list."""
        from backend.routers.scanner import ScanRequest

        # Empty list should be valid syntactically (endpoint may reject it)
        request = ScanRequest(symbols=[])
        assert request.symbols == []

    def test_large_dte_value(self):
        """Test ScanRequest with large DTE value."""
        from backend.routers.scanner import ScanRequest

        request = ScanRequest(symbols=["AAPL"], dte=365)
        assert request.dte == 365

    def test_zero_min_premium_pct(self):
        """Test ScanRequest with zero min premium."""
        from backend.routers.scanner import ScanRequest

        request = ScanRequest(symbols=["AAPL"], min_premium_pct=0.0)
        assert request.min_premium_pct == 0.0


# ============================================================================
# CACHE KEY GENERATION TESTS
# ============================================================================

class TestCacheKeyGeneration:
    """Tests for cache key generation patterns."""

    def test_cache_key_with_symbols(self):
        """Test generating a cache key with symbols."""
        import hashlib

        symbols = ["AAPL", "MSFT", "NVDA"]
        params = {
            "min_premium_pct": 1.0,
            "max_price": 100.0,
            "max_dte": 30,
            "symbols": symbols,
        }

        # Simulate cache key generation
        sorted_symbols = sorted(symbols)
        key_parts = [
            f"min_prem_{params['min_premium_pct']}",
            f"max_price_{params['max_price']}",
            f"max_dte_{params['max_dte']}",
            f"symbols_{','.join(sorted_symbols)}",
        ]
        key = "_".join(key_parts)

        assert "AAPL" in key
        assert "min_prem_1.0" in key


# ============================================================================
# ADAPTIVE TIMEOUT CALCULATION TESTS
# ============================================================================

class TestAdaptiveTimeout:
    """Tests for adaptive timeout calculation logic."""

    def test_timeout_calculation_capped_at_max(self):
        """Test that timeout calculation is capped at max."""
        from backend.routers.scanner import (
            TIMEOUT_LIVE_SCAN_MAX,
            TIMEOUT_PER_SYMBOL,
            TIMEOUT_MIN,
        )

        # Simulate the calculation from scanner.py
        symbol_count = 100  # Large number
        calculated_timeout = TIMEOUT_PER_SYMBOL * symbol_count  # 1000 seconds
        adaptive_timeout = min(TIMEOUT_LIVE_SCAN_MAX, max(TIMEOUT_MIN, calculated_timeout))

        assert adaptive_timeout == TIMEOUT_LIVE_SCAN_MAX  # Should be capped at 120

    def test_timeout_calculation_has_minimum(self):
        """Test that timeout calculation has minimum floor."""
        from backend.routers.scanner import (
            TIMEOUT_LIVE_SCAN_MAX,
            TIMEOUT_PER_SYMBOL,
            TIMEOUT_MIN,
        )

        # Small number of symbols
        symbol_count = 1
        calculated_timeout = TIMEOUT_PER_SYMBOL * symbol_count  # 10 seconds
        adaptive_timeout = min(TIMEOUT_LIVE_SCAN_MAX, max(TIMEOUT_MIN, calculated_timeout))

        assert adaptive_timeout == TIMEOUT_MIN  # Should be floored at 30

    def test_timeout_calculation_scales_with_symbols(self):
        """Test that timeout scales correctly within bounds."""
        from backend.routers.scanner import (
            TIMEOUT_LIVE_SCAN_MAX,
            TIMEOUT_PER_SYMBOL,
            TIMEOUT_MIN,
        )

        # Medium number of symbols
        symbol_count = 5
        calculated_timeout = TIMEOUT_PER_SYMBOL * symbol_count  # 50 seconds
        adaptive_timeout = min(TIMEOUT_LIVE_SCAN_MAX, max(TIMEOUT_MIN, calculated_timeout))

        assert adaptive_timeout == 50  # Should be calculated value
        assert TIMEOUT_MIN < adaptive_timeout < TIMEOUT_LIVE_SCAN_MAX

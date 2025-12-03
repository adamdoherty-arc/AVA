"""
Positions Connector for Magnus Financial Assistant
Provides access to Robinhood option positions
"""

import robin_stocks.robinhood as rh
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
import logging
import os
from dotenv import load_dotenv

from .connector_base import BaseConnector

load_dotenv()
logger = logging.getLogger(__name__)


class PositionsConnector(BaseConnector):
    """
    Connector for accessing Robinhood option positions.

    Features:
    - Get all open option positions
    - Get position details by symbol
    - Calculate P&L for positions
    - Filter positions by criteria (delta, DTE, P&L)
    - Get position Greeks
    """

    def __init__(self, cache_ttl: int = 60):
        """
        Initialize positions connector.

        Args:
            cache_ttl: Cache time-to-live in seconds (default: 60 seconds)
        """
        super().__init__(cache_ttl)
        self.logged_in = False
        self._ensure_login()

    def _ensure_login(self) -> None:
        """Ensure Robinhood is logged in."""
        if self.logged_in:
            return

        try:
            username = os.getenv("ROBINHOOD_USERNAME")
            password = os.getenv("ROBINHOOD_PASSWORD")

            if not username or not password:
                logger.error("Robinhood credentials not found in environment")
                return

            rh.login(username, password)
            self.logged_in = True
            logger.info("Successfully logged in to Robinhood")

        except Exception as e:
            logger.error(f"Failed to login to Robinhood: {e}")
            self.logged_in = False

    def get_data(self, **kwargs) -> Dict[str, Any]:
        """
        Get positions data based on query parameters.

        Supported kwargs:
        - symbol: str - Filter by underlying symbol
        - min_delta: float - Minimum delta
        - max_delta: float - Maximum delta
        - min_dte: int - Minimum days to expiration
        - max_dte: int - Maximum days to expiration
        - min_pnl_pct: float - Minimum P&L percentage
        - max_pnl_pct: float - Maximum P&L percentage
        - risky_only: bool - Only show risky positions

        Returns:
            Dictionary with positions data
        """
        # Get filter parameters
        symbol = kwargs.get('symbol')
        min_delta = kwargs.get('min_delta')
        max_delta = kwargs.get('max_delta')
        min_dte = kwargs.get('min_dte')
        max_dte = kwargs.get('max_dte')
        min_pnl_pct = kwargs.get('min_pnl_pct')
        max_pnl_pct = kwargs.get('max_pnl_pct')
        risky_only = kwargs.get('risky_only', False)

        # Build cache key
        cache_key = f"positions_{symbol or 'all'}"
        if risky_only:
            cache_key += "_risky"

        # Get all positions (with caching)
        def fetch_positions():
            return self._fetch_all_positions()

        positions_data = self.get_cached_or_fetch(cache_key, fetch_positions)

        if "error" in positions_data:
            return positions_data

        # Apply filters
        filtered_positions = self._apply_filters(
            positions_data.get("positions", []),
            symbol=symbol,
            min_delta=min_delta,
            max_delta=max_delta,
            min_dte=min_dte,
            max_dte=max_dte,
            min_pnl_pct=min_pnl_pct,
            max_pnl_pct=max_pnl_pct,
            risky_only=risky_only
        )

        return self.format_success_response(
            filtered_positions,
            metadata={
                "total_positions": len(positions_data.get("positions", [])),
                "filtered_count": len(filtered_positions),
                "filters_applied": {
                    "symbol": symbol,
                    "min_delta": min_delta,
                    "max_delta": max_delta,
                    "min_dte": min_dte,
                    "max_dte": max_dte,
                    "min_pnl_pct": min_pnl_pct,
                    "max_pnl_pct": max_pnl_pct,
                    "risky_only": risky_only
                }
            }
        )

    def _fetch_all_positions(self) -> Dict[str, Any]:
        """Fetch all option positions from Robinhood."""
        try:
            self._ensure_login()

            if not self.logged_in:
                return {"error": "Not logged in to Robinhood"}

            # Get option positions
            positions = rh.options.get_open_option_positions()

            if not positions:
                return {"positions": []}

            # Process each position
            processed_positions = []

            for pos in positions:
                try:
                    # Get basic position info
                    instrument_url = pos.get("option")
                    if not instrument_url:
                        continue

                    instrument_data = rh.helper.request_get(instrument_url)
                    if not instrument_data:
                        continue

                    # Extract key information
                    chain_symbol = instrument_data.get("chain_symbol", "")
                    strike = float(instrument_data.get("strike_price", 0))
                    expiration_date = instrument_data.get("expiration_date", "")
                    option_type = instrument_data.get("type", "")  # 'call' or 'put'

                    # Get position details
                    quantity = float(pos.get("quantity", 0))
                    average_price = float(pos.get("average_price", 0))

                    # Calculate current value
                    # Note: In real implementation, you'd get current option price
                    # For now, use a placeholder
                    current_price = float(pos.get("price", average_price))

                    # Calculate P&L
                    cost_basis = quantity * average_price * 100  # Options are per 100 shares
                    current_value = quantity * current_price * 100
                    pnl = current_value - cost_basis
                    pnl_pct = (pnl / cost_basis * 100) if cost_basis > 0 else 0

                    # Calculate days to expiration
                    try:
                        exp_date = datetime.strptime(expiration_date, "%Y-%m-%d")
                        dte = (exp_date - datetime.now()).days
                    except:
                        dte = 0

                    # Get Greeks (if available)
                    greeks = pos.get("greeks", {})
                    delta = float(greeks.get("delta", 0)) if greeks else 0
                    theta = float(greeks.get("theta", 0)) if greeks else 0
                    gamma = float(greeks.get("gamma", 0)) if greeks else 0
                    vega = float(greeks.get("vega", 0)) if greeks else 0

                    # Determine if position is risky
                    is_risky = (
                        abs(delta) > 0.5 or  # High delta
                        dte < 7 or  # Less than 7 days to expiration
                        pnl_pct < -20  # Down more than 20%
                    )

                    processed_position = {
                        "symbol": chain_symbol,
                        "strike": strike,
                        "expiration": expiration_date,
                        "dte": dte,
                        "type": option_type,
                        "quantity": quantity,
                        "average_price": average_price,
                        "current_price": current_price,
                        "cost_basis": cost_basis,
                        "current_value": current_value,
                        "pnl": pnl,
                        "pnl_pct": pnl_pct,
                        "greeks": {
                            "delta": delta,
                            "theta": theta,
                            "gamma": gamma,
                            "vega": vega
                        },
                        "is_risky": is_risky,
                        "raw_data": pos  # Keep raw data for reference
                    }

                    processed_positions.append(processed_position)

                except Exception as e:
                    logger.error(f"Error processing position: {e}")
                    continue

            return {"positions": processed_positions}

        except Exception as e:
            logger.error(f"Error fetching positions: {e}")
            return self.format_error_response(e, "fetch_all_positions")

    def _apply_filters(
        self,
        positions: List[Dict],
        symbol: Optional[str] = None,
        min_delta: Optional[float] = None,
        max_delta: Optional[float] = None,
        min_dte: Optional[int] = None,
        max_dte: Optional[int] = None,
        min_pnl_pct: Optional[float] = None,
        max_pnl_pct: Optional[float] = None,
        risky_only: bool = False
    ) -> List[Dict]:
        """Apply filters to positions list."""
        filtered = positions

        # Filter by symbol
        if symbol:
            filtered = [p for p in filtered if p["symbol"].upper() == symbol.upper()]

        # Filter by delta
        if min_delta is not None:
            filtered = [p for p in filtered if abs(p["greeks"]["delta"]) >= min_delta]
        if max_delta is not None:
            filtered = [p for p in filtered if abs(p["greeks"]["delta"]) <= max_delta]

        # Filter by DTE
        if min_dte is not None:
            filtered = [p for p in filtered if p["dte"] >= min_dte]
        if max_dte is not None:
            filtered = [p for p in filtered if p["dte"] <= max_dte]

        # Filter by P&L percentage
        if min_pnl_pct is not None:
            filtered = [p for p in filtered if p["pnl_pct"] >= min_pnl_pct]
        if max_pnl_pct is not None:
            filtered = [p for p in filtered if p["pnl_pct"] <= max_pnl_pct]

        # Filter risky only
        if risky_only:
            filtered = [p for p in filtered if p["is_risky"]]

        return filtered

    def validate_response(self, data: Dict[str, Any]) -> bool:
        """
        Validate positions response.

        Args:
            data: Response data

        Returns:
            True if valid
        """
        if "error" in data:
            return False

        if "positions" not in data:
            return False

        return True

    # Convenience methods

    def get_all_positions(self) -> List[Dict]:
        """Get all open positions."""
        result = self.get_data()
        if result.get("success"):
            return result["data"]
        return []

    def get_position_by_symbol(self, symbol: str) -> List[Dict]:
        """Get positions for a specific symbol."""
        result = self.get_data(symbol=symbol)
        if result.get("success"):
            return result["data"]
        return []

    def get_risky_positions(self) -> List[Dict]:
        """Get only risky positions."""
        result = self.get_data(risky_only=True)
        if result.get("success"):
            return result["data"]
        return []

    def get_position_summary(self) -> Dict[str, Any]:
        """Get summary statistics for all positions."""
        positions = self.get_all_positions()

        if not positions:
            return {
                "total_positions": 0,
                "total_value": 0,
                "total_pnl": 0,
                "total_pnl_pct": 0,
                "risky_count": 0
            }

        total_value = sum(p["current_value"] for p in positions)
        total_cost = sum(p["cost_basis"] for p in positions)
        total_pnl = total_value - total_cost
        total_pnl_pct = (total_pnl / total_cost * 100) if total_cost > 0 else 0
        risky_count = sum(1 for p in positions if p["is_risky"])

        return {
            "total_positions": len(positions),
            "total_value": total_value,
            "total_cost": total_cost,
            "total_pnl": total_pnl,
            "total_pnl_pct": total_pnl_pct,
            "risky_count": risky_count,
            "risky_pct": (risky_count / len(positions) * 100) if positions else 0
        }


# Register connector on import
def _register():
    """Register the connector in the global registry."""
    from .connector_base import register_connector
    connector = PositionsConnector()
    register_connector("positions", connector)


# Auto-register
_register()

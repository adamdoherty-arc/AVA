"""
P&L Calculator - Validates profit/loss calculations

Validates:
- Stock position P&L
- Option position P&L
- Total portfolio P&L
- Percentage returns
"""

from dataclasses import dataclass
from typing import List, Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)


@dataclass
class StockPosition:
    """Stock position data"""
    symbol: str
    quantity: float
    avg_price: float
    current_price: float
    reported_pl: Optional[float] = None
    reported_pl_pct: Optional[float] = None


@dataclass
class OptionPosition:
    """Option position data"""
    symbol: str
    quantity: int
    avg_price: float
    current_price: float
    multiplier: int = 100  # Standard options multiplier
    reported_pl: Optional[float] = None
    reported_pl_pct: Optional[float] = None


class PLCalculator:
    """
    Validates P&L calculations for positions.

    Catches common errors:
    - Floating point precision issues
    - Wrong multiplier usage
    - Sign errors
    - Percentage calculation errors
    """

    # Tolerance for P&L validation
    ABSOLUTE_TOLERANCE = 0.02    # $0.02 absolute difference
    PERCENTAGE_TOLERANCE = 0.01  # 1% relative difference

    def calculate_stock_pl(self, position: StockPosition) -> Dict[str, float]:
        """
        Calculate stock position P&L

        Returns:
            Dict with pl, pl_pct, cost_basis, current_value
        """
        cost_basis = position.quantity * position.avg_price
        current_value = position.quantity * position.current_price
        pl = current_value - cost_basis

        # Avoid division by zero
        if cost_basis != 0:
            pl_pct = (pl / cost_basis) * 100
        else:
            pl_pct = 0.0

        return {
            'pl': pl,
            'pl_pct': pl_pct,
            'cost_basis': cost_basis,
            'current_value': current_value,
        }

    def calculate_option_pl(self, position: OptionPosition) -> Dict[str, float]:
        """
        Calculate option position P&L

        Returns:
            Dict with pl, pl_pct, cost_basis, current_value
        """
        # Options P&L includes multiplier
        cost_basis = position.quantity * position.avg_price * position.multiplier
        current_value = position.quantity * position.current_price * position.multiplier
        pl = current_value - cost_basis

        # Avoid division by zero
        if cost_basis != 0:
            pl_pct = (pl / abs(cost_basis)) * 100
        else:
            pl_pct = 0.0

        return {
            'pl': pl,
            'pl_pct': pl_pct,
            'cost_basis': cost_basis,
            'current_value': current_value,
        }

    def validate_stock_pl(self, position: StockPosition) -> List[Dict[str, Any]]:
        """
        Validate stock position P&L calculation

        Returns:
            List of validation issues (empty if valid)
        """
        issues = []

        if position.reported_pl is None and position.reported_pl_pct is None:
            return issues  # Nothing to validate

        calculated = self.calculate_stock_pl(position)

        # Validate absolute P&L
        if position.reported_pl is not None:
            diff = abs(position.reported_pl - calculated['pl'])
            if diff > self.ABSOLUTE_TOLERANCE:
                # Check for common errors
                error_type = self._identify_pl_error(
                    position.reported_pl,
                    calculated['pl'],
                    position.quantity,
                    position.avg_price,
                    position.current_price,
                )
                issues.append({
                    'field': 'pl',
                    'symbol': position.symbol,
                    'expected': round(calculated['pl'], 2),
                    'reported': round(position.reported_pl, 2),
                    'difference': round(diff, 2),
                    'error_type': error_type,
                    'message': f"P&L mismatch for {position.symbol}: "
                              f"expected ${calculated['pl']:.2f}, got ${position.reported_pl:.2f}"
                })

        # Validate percentage P&L
        if position.reported_pl_pct is not None:
            diff_pct = abs(position.reported_pl_pct - calculated['pl_pct'])
            if diff_pct > self.PERCENTAGE_TOLERANCE * 100:
                issues.append({
                    'field': 'pl_pct',
                    'symbol': position.symbol,
                    'expected': round(calculated['pl_pct'], 2),
                    'reported': round(position.reported_pl_pct, 2),
                    'difference': round(diff_pct, 2),
                    'message': f"P&L % mismatch for {position.symbol}: "
                              f"expected {calculated['pl_pct']:.2f}%, got {position.reported_pl_pct:.2f}%"
                })

        return issues

    def validate_option_pl(self, position: OptionPosition) -> List[Dict[str, Any]]:
        """
        Validate option position P&L calculation

        Returns:
            List of validation issues (empty if valid)
        """
        issues = []

        if position.reported_pl is None and position.reported_pl_pct is None:
            return issues  # Nothing to validate

        calculated = self.calculate_option_pl(position)

        # Validate absolute P&L
        if position.reported_pl is not None:
            diff = abs(position.reported_pl - calculated['pl'])
            if diff > self.ABSOLUTE_TOLERANCE:
                # Check if multiplier was forgotten
                without_multiplier = self._calculate_without_multiplier(position)
                if abs(position.reported_pl - without_multiplier) < self.ABSOLUTE_TOLERANCE:
                    error_type = 'missing_multiplier'
                else:
                    error_type = 'calculation_error'

                issues.append({
                    'field': 'pl',
                    'symbol': position.symbol,
                    'expected': round(calculated['pl'], 2),
                    'reported': round(position.reported_pl, 2),
                    'difference': round(diff, 2),
                    'error_type': error_type,
                    'message': f"Option P&L mismatch for {position.symbol}: "
                              f"expected ${calculated['pl']:.2f}, got ${position.reported_pl:.2f}"
                              f"{' (multiplier may be missing)' if error_type == 'missing_multiplier' else ''}"
                })

        return issues

    def _calculate_without_multiplier(self, position: OptionPosition) -> float:
        """Calculate option P&L without multiplier (common error)"""
        cost_basis = position.quantity * position.avg_price
        current_value = position.quantity * position.current_price
        return current_value - cost_basis

    def _identify_pl_error(
        self,
        reported: float,
        expected: float,
        quantity: float,
        avg_price: float,
        current_price: float,
    ) -> str:
        """Identify the type of P&L calculation error"""

        # Check for sign error
        if abs(reported + expected) < self.ABSOLUTE_TOLERANCE:
            return 'sign_error'

        # Check for quantity=1 error
        single_pl = current_price - avg_price
        if abs(reported - single_pl) < self.ABSOLUTE_TOLERANCE:
            return 'quantity_ignored'

        # Check for inverted calculation (cost - current instead of current - cost)
        inverted = (quantity * avg_price) - (quantity * current_price)
        if abs(reported - inverted) < self.ABSOLUTE_TOLERANCE:
            return 'inverted_calculation'

        return 'unknown'

    def validate_portfolio_total(
        self,
        positions: List[Dict[str, Any]],
        reported_total: float,
    ) -> List[Dict[str, Any]]:
        """
        Validate total portfolio P&L matches sum of positions

        Returns:
            List of validation issues
        """
        issues = []

        calculated_total = 0.0
        for pos in positions:
            if 'pl' in pos:
                calculated_total += pos['pl']

        diff = abs(reported_total - calculated_total)
        if diff > self.ABSOLUTE_TOLERANCE:
            issues.append({
                'field': 'total_pl',
                'expected': round(calculated_total, 2),
                'reported': round(reported_total, 2),
                'difference': round(diff, 2),
                'message': f"Portfolio total P&L mismatch: "
                          f"expected ${calculated_total:.2f}, got ${reported_total:.2f}"
            })

        return issues

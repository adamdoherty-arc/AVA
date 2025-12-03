"""
Greeks Validator - Validates options Greeks calculations

Validates:
- Delta (sensitivity to underlying price)
- Gamma (rate of change of delta)
- Theta (time decay)
- Vega (sensitivity to volatility)
- Rho (sensitivity to interest rates)
"""

import math
from dataclasses import dataclass
from datetime import datetime, date
from typing import Optional, Tuple, List, Dict, Any


@dataclass
class GreeksValues:
    """Container for Greek values"""
    delta: float
    gamma: float
    theta: float
    vega: float
    rho: Optional[float] = None


class GreeksValidator:
    """
    Validates options Greeks calculations using Black-Scholes model.

    Compares reported Greeks against expected values and flags discrepancies.
    """

    # Tolerance for Greek validation (as percentage)
    DELTA_TOLERANCE = 0.05   # 5%
    GAMMA_TOLERANCE = 0.10   # 10%
    THETA_TOLERANCE = 0.10   # 10%
    VEGA_TOLERANCE = 0.10    # 10%
    RHO_TOLERANCE = 0.15     # 15%

    def __init__(self, risk_free_rate: float = 0.05):
        """
        Initialize validator

        Args:
            risk_free_rate: Risk-free interest rate (default 5%)
        """
        self.risk_free_rate = risk_free_rate

    @staticmethod
    def _norm_cdf(x: float) -> float:
        """Standard normal cumulative distribution function"""
        return (1.0 + math.erf(x / math.sqrt(2.0))) / 2.0

    @staticmethod
    def _norm_pdf(x: float) -> float:
        """Standard normal probability density function"""
        return math.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)

    def _calculate_d1_d2(
        self,
        stock_price: float,
        strike: float,
        time_to_expiry: float,
        volatility: float,
    ) -> Tuple[float, float]:
        """Calculate d1 and d2 for Black-Scholes"""
        if time_to_expiry <= 0 or volatility <= 0:
            return 0, 0

        sqrt_t = math.sqrt(time_to_expiry)
        d1 = (math.log(stock_price / strike) +
              (self.risk_free_rate + 0.5 * volatility ** 2) * time_to_expiry) / \
             (volatility * sqrt_t)
        d2 = d1 - volatility * sqrt_t

        return d1, d2

    def calculate_greeks(
        self,
        stock_price: float,
        strike: float,
        days_to_expiry: int,
        volatility: float,
        is_call: bool = True,
    ) -> GreeksValues:
        """
        Calculate theoretical Greeks using Black-Scholes

        Args:
            stock_price: Current stock price
            strike: Option strike price
            days_to_expiry: Days until expiration
            volatility: Implied volatility (as decimal, e.g., 0.30 for 30%)
            is_call: True for call, False for put

        Returns:
            GreeksValues with calculated Greeks
        """
        # Convert days to years
        time_to_expiry = days_to_expiry / 365.0

        if time_to_expiry <= 0:
            # Expired option
            return GreeksValues(
                delta=1.0 if is_call and stock_price > strike else 0.0,
                gamma=0.0,
                theta=0.0,
                vega=0.0,
                rho=0.0,
            )

        d1, d2 = self._calculate_d1_d2(stock_price, strike, time_to_expiry, volatility)
        sqrt_t = math.sqrt(time_to_expiry)

        # Calculate Greeks
        if is_call:
            delta = self._norm_cdf(d1)
            rho = strike * time_to_expiry * math.exp(-self.risk_free_rate * time_to_expiry) * \
                  self._norm_cdf(d2) / 100
        else:
            delta = self._norm_cdf(d1) - 1
            rho = -strike * time_to_expiry * math.exp(-self.risk_free_rate * time_to_expiry) * \
                  self._norm_cdf(-d2) / 100

        gamma = self._norm_pdf(d1) / (stock_price * volatility * sqrt_t)

        theta_common = -(stock_price * volatility * self._norm_pdf(d1)) / (2 * sqrt_t)
        if is_call:
            theta = (theta_common -
                     self.risk_free_rate * strike * math.exp(-self.risk_free_rate * time_to_expiry) *
                     self._norm_cdf(d2)) / 365
        else:
            theta = (theta_common +
                     self.risk_free_rate * strike * math.exp(-self.risk_free_rate * time_to_expiry) *
                     self._norm_cdf(-d2)) / 365

        vega = stock_price * sqrt_t * self._norm_pdf(d1) / 100

        return GreeksValues(
            delta=delta,
            gamma=gamma,
            theta=theta,
            vega=vega,
            rho=rho,
        )

    def validate_greeks(
        self,
        reported: Dict[str, float],
        stock_price: float,
        strike: float,
        days_to_expiry: int,
        volatility: float,
        is_call: bool = True,
    ) -> List[Dict[str, Any]]:
        """
        Validate reported Greeks against calculated values

        Args:
            reported: Dict with reported Greek values
            stock_price: Current stock price
            strike: Option strike price
            days_to_expiry: Days until expiration
            volatility: Implied volatility
            is_call: True for call, False for put

        Returns:
            List of validation issues (empty if all valid)
        """
        issues = []

        # Calculate expected Greeks
        expected = self.calculate_greeks(
            stock_price, strike, days_to_expiry, volatility, is_call
        )

        # Validate each Greek
        validations = [
            ('delta', expected.delta, self.DELTA_TOLERANCE),
            ('gamma', expected.gamma, self.GAMMA_TOLERANCE),
            ('theta', expected.theta, self.THETA_TOLERANCE),
            ('vega', expected.vega, self.VEGA_TOLERANCE),
        ]

        for greek_name, expected_value, tolerance in validations:
            reported_value = reported.get(greek_name)

            if reported_value is None:
                issues.append({
                    'greek': greek_name,
                    'error': 'missing',
                    'message': f"Missing {greek_name} value",
                })
                continue

            # Check if difference exceeds tolerance
            if expected_value != 0:
                diff_pct = abs(reported_value - expected_value) / abs(expected_value)
            else:
                diff_pct = abs(reported_value) if reported_value != 0 else 0

            if diff_pct > tolerance:
                issues.append({
                    'greek': greek_name,
                    'error': 'mismatch',
                    'expected': round(expected_value, 4),
                    'reported': round(reported_value, 4),
                    'difference_pct': round(diff_pct * 100, 2),
                    'tolerance_pct': tolerance * 100,
                    'message': f"{greek_name} mismatch: expected {expected_value:.4f}, "
                              f"got {reported_value:.4f} ({diff_pct*100:.1f}% diff)"
                })

        return issues

    def validate_delta_range(self, delta: float, is_call: bool) -> Optional[str]:
        """
        Validate delta is within valid range

        Call delta: 0 to 1
        Put delta: -1 to 0
        """
        if is_call:
            if delta < 0 or delta > 1:
                return f"Call delta {delta:.4f} out of range [0, 1]"
        else:
            if delta < -1 or delta > 0:
                return f"Put delta {delta:.4f} out of range [-1, 0]"
        return None

    def validate_theta_sign(self, theta: float) -> Optional[str]:
        """
        Validate theta is negative (time decay)

        Theta should almost always be negative (except deep ITM European options)
        """
        if theta > 0.01:  # Small positive theta can occur for deep ITM
            return f"Unusual positive theta: {theta:.4f}"
        return None

    def validate_vega_positive(self, vega: float) -> Optional[str]:
        """Validate vega is positive (volatility sensitivity)"""
        if vega < 0:
            return f"Vega should be positive: {vega:.4f}"
        return None

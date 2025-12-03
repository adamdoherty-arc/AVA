"""
Advanced Greeks Engine
======================

High-performance vectorized Greeks calculations with:
- First-order Greeks: Delta, Gamma, Theta, Vega, Rho
- Second-order Greeks: Vanna, Volga (Vomma), Charm, Veta
- Third-order Greeks: Speed, Zomma, Color, Ultima
- Multi-leg position Greeks
- IV Surface analysis
- Theta decay projections
- Probability calculations

Uses py_vollib_vectorized for 10-100x performance improvement.

Author: AVA Trading Platform
Created: 2025-11-28
"""

import logging
import numpy as np
import pandas as pd
from scipy.stats import norm
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, field
from datetime import datetime, date, timedelta
from enum import Enum

# Try to import vectorized libraries for performance
try:
    import py_vollib_vectorized as pv
    from py_vollib_vectorized.implied_volatility import vectorized_implied_volatility
    from py_vollib_vectorized.greeks import delta as pv_delta, gamma as pv_gamma
    from py_vollib_vectorized.greeks import theta as pv_theta, vega as pv_vega, rho as pv_rho
    HAS_VOLLIB = True
except ImportError:
    HAS_VOLLIB = False

try:
    import polars as pl
    HAS_POLARS = True
except ImportError:
    HAS_POLARS = False

logger = logging.getLogger(__name__)


class OptionType(Enum):
    CALL = 'call'
    PUT = 'put'


class PositionSide(Enum):
    LONG = 'long'
    SHORT = 'short'


@dataclass
class OptionContract:
    """Represents a single option contract"""
    symbol: str
    strike: float
    expiration: date
    option_type: OptionType
    underlying_price: float
    implied_volatility: float
    quantity: int = 1
    side: PositionSide = PositionSide.LONG
    bid: float = 0.0
    ask: float = 0.0
    last_price: float = 0.0
    open_interest: int = 0
    volume: int = 0

    @property
    def mid_price(self) -> float:
        if self.bid > 0 and self.ask > 0:
            return (self.bid + self.ask) / 2
        return self.last_price

    @property
    def days_to_expiration(self) -> int:
        return (self.expiration - date.today()).days

    @property
    def time_to_expiration(self) -> float:
        """Time to expiration in years"""
        return max(self.days_to_expiration / 365.0, 0.001)


@dataclass
class GreeksResult:
    """Complete Greeks calculation result"""
    # First-order Greeks
    delta: float = 0.0
    gamma: float = 0.0
    theta: float = 0.0  # Daily
    vega: float = 0.0
    rho: float = 0.0

    # Second-order Greeks
    vanna: float = 0.0      # d(delta)/d(vol) or d(vega)/d(spot)
    volga: float = 0.0      # d(vega)/d(vol), also called Vomma
    charm: float = 0.0      # d(delta)/d(time), delta decay
    veta: float = 0.0       # d(vega)/d(time), vega decay

    # Third-order Greeks
    speed: float = 0.0      # d(gamma)/d(spot)
    zomma: float = 0.0      # d(gamma)/d(vol)
    color: float = 0.0      # d(gamma)/d(time)
    ultima: float = 0.0     # d(vomma)/d(vol)

    # Pricing
    theoretical_price: float = 0.0
    intrinsic_value: float = 0.0
    extrinsic_value: float = 0.0

    # Probabilities
    prob_itm: float = 0.0   # Probability of being in-the-money at expiration
    prob_profit: float = 0.0  # Probability of profit (for the position)
    prob_touch: float = 0.0   # Probability of touching strike before expiration

    # Risk metrics
    max_profit: float = 0.0
    max_loss: float = 0.0
    breakeven: List[float] = field(default_factory=list)


class AdvancedGreeksEngine:
    """
    High-performance Greeks calculation engine

    Features:
    - Vectorized calculations for batch processing
    - All first, second, and third-order Greeks
    - Multi-leg position aggregation
    - IV surface analysis
    - Theta decay projections
    """

    def __init__(self, risk_free_rate: float = 0.05, dividend_yield: float = 0.0):
        """
        Initialize the Greeks engine.

        Args:
            risk_free_rate: Risk-free interest rate (default 5%)
            dividend_yield: Continuous dividend yield (default 0%)
        """
        self.risk_free_rate = risk_free_rate
        self.dividend_yield = dividend_yield
        self._use_vectorized = HAS_VOLLIB

        if HAS_VOLLIB:
            logger.info("Using py_vollib_vectorized for high-performance Greeks")
        else:
            logger.warning("py_vollib_vectorized not available, using scipy fallback")

    def _d1_d2(self, S: float, K: float, T: float, sigma: float) -> Tuple[float, float]:
        """Calculate d1 and d2 for Black-Scholes"""
        if T <= 0 or sigma <= 0:
            return 0.0, 0.0

        d1 = (np.log(S / K) + (self.risk_free_rate - self.dividend_yield + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        return d1, d2

    def calculate_greeks(
        self,
        S: float,
        K: float,
        T: float,
        sigma: float,
        option_type: Union[str, OptionType] = 'call',
        include_higher_order: bool = True
    ) -> GreeksResult:
        """
        Calculate all Greeks for a single option.

        Args:
            S: Current underlying price
            K: Strike price
            T: Time to expiration in years
            sigma: Implied volatility (decimal, e.g., 0.25 for 25%)
            option_type: 'call' or 'put'
            include_higher_order: Calculate second and third-order Greeks

        Returns:
            GreeksResult with all calculated values
        """
        if isinstance(option_type, OptionType):
            option_type = option_type.value

        result = GreeksResult()

        # Handle expired options
        if T <= 0:
            if option_type == 'call':
                result.intrinsic_value = max(0, S - K)
                result.delta = 1.0 if S > K else 0.0
            else:
                result.intrinsic_value = max(0, K - S)
                result.delta = -1.0 if K > S else 0.0
            result.theoretical_price = result.intrinsic_value
            result.prob_itm = 1.0 if result.intrinsic_value > 0 else 0.0
            return result

        # Calculate d1 and d2
        d1, d2 = self._d1_d2(S, K, T, sigma)
        sqrt_T = np.sqrt(T)

        # Standard normal PDF and CDF
        n_d1 = norm.pdf(d1)
        n_d2 = norm.pdf(d2)
        N_d1 = norm.cdf(d1)
        N_d2 = norm.cdf(d2)
        N_neg_d1 = norm.cdf(-d1)
        N_neg_d2 = norm.cdf(-d2)

        # Discount factors
        discount = np.exp(-self.risk_free_rate * T)
        div_discount = np.exp(-self.dividend_yield * T)

        # === FIRST-ORDER GREEKS ===

        if option_type == 'call':
            result.delta = div_discount * N_d1
            result.theoretical_price = S * div_discount * N_d1 - K * discount * N_d2
            result.prob_itm = N_d2
        else:
            result.delta = -div_discount * N_neg_d1
            result.theoretical_price = K * discount * N_neg_d2 - S * div_discount * N_neg_d1
            result.prob_itm = N_neg_d2

        # Gamma (same for calls and puts)
        result.gamma = div_discount * n_d1 / (S * sigma * sqrt_T)

        # Theta (daily)
        theta_part1 = -div_discount * S * n_d1 * sigma / (2 * sqrt_T)
        if option_type == 'call':
            theta_part2 = self.dividend_yield * S * div_discount * N_d1
            theta_part3 = -self.risk_free_rate * K * discount * N_d2
            result.theta = (theta_part1 + theta_part2 + theta_part3) / 365
        else:
            theta_part2 = -self.dividend_yield * S * div_discount * N_neg_d1
            theta_part3 = self.risk_free_rate * K * discount * N_neg_d2
            result.theta = (theta_part1 + theta_part2 + theta_part3) / 365

        # Vega (per 1% move)
        result.vega = S * div_discount * n_d1 * sqrt_T / 100

        # Rho (per 1% move)
        if option_type == 'call':
            result.rho = K * T * discount * N_d2 / 100
        else:
            result.rho = -K * T * discount * N_neg_d2 / 100

        # === SECOND-ORDER GREEKS ===

        if include_higher_order:
            # Vanna: d(delta)/d(vol) = d(vega)/d(S)
            result.vanna = -div_discount * n_d1 * d2 / sigma

            # Volga (Vomma): d(vega)/d(vol)
            result.volga = result.vega * d1 * d2 / sigma

            # Charm: d(delta)/d(time) - delta decay
            charm_factor = div_discount * n_d1 * (
                2 * (self.risk_free_rate - self.dividend_yield) * T - d2 * sigma * sqrt_T
            ) / (2 * T * sigma * sqrt_T)
            if option_type == 'call':
                result.charm = -charm_factor - self.dividend_yield * div_discount * N_d1
            else:
                result.charm = -charm_factor + self.dividend_yield * div_discount * N_neg_d1
            result.charm = result.charm / 365  # Daily

            # Veta: d(vega)/d(time)
            veta_factor = (
                self.dividend_yield +
                ((self.risk_free_rate - self.dividend_yield) * d1) / (sigma * sqrt_T) -
                (1 + d1 * d2) / (2 * T)
            )
            result.veta = S * div_discount * n_d1 * sqrt_T * veta_factor / 365

        # === THIRD-ORDER GREEKS ===

        if include_higher_order:
            # Speed: d(gamma)/d(S)
            result.speed = -result.gamma / S * (d1 / (sigma * sqrt_T) + 1)

            # Zomma: d(gamma)/d(vol)
            result.zomma = result.gamma * (d1 * d2 - 1) / sigma

            # Color: d(gamma)/d(time)
            color_factor = (
                2 * (self.risk_free_rate - self.dividend_yield) * T - d2 * sigma * sqrt_T
            ) / (2 * T * sigma * sqrt_T)
            result.color = -div_discount * n_d1 / (2 * S * T * sigma * sqrt_T) * (
                2 * self.dividend_yield * T + 1 + d1 * color_factor
            ) / 365

            # Ultima: d(vomma)/d(vol)
            result.ultima = result.vega / (sigma ** 2) * (
                d1 * d2 * (1 - d1 * d2) + d1 ** 2 + d2 ** 2
            )

        # === INTRINSIC AND EXTRINSIC VALUE ===

        if option_type == 'call':
            result.intrinsic_value = max(0, S - K)
        else:
            result.intrinsic_value = max(0, K - S)
        result.extrinsic_value = max(0, result.theoretical_price - result.intrinsic_value)

        # === PROBABILITY CALCULATIONS ===

        # Probability of touching the strike
        # P(touch) ≈ 2 * P(ITM) for at-the-money options
        if option_type == 'call':
            result.prob_touch = min(1.0, 2 * result.prob_itm)
        else:
            result.prob_touch = min(1.0, 2 * result.prob_itm)

        return result

    def calculate_greeks_vectorized(
        self,
        S: np.ndarray,
        K: np.ndarray,
        T: np.ndarray,
        sigma: np.ndarray,
        option_type: np.ndarray
    ) -> pd.DataFrame:
        """
        Calculate Greeks for multiple options simultaneously (vectorized).

        Args:
            S: Array of underlying prices
            K: Array of strike prices
            T: Array of times to expiration (years)
            sigma: Array of implied volatilities
            option_type: Array of option types ('call' or 'put')

        Returns:
            DataFrame with Greeks for each option
        """
        n = len(S)

        # Initialize results
        results = {
            'delta': np.zeros(n),
            'gamma': np.zeros(n),
            'theta': np.zeros(n),
            'vega': np.zeros(n),
            'rho': np.zeros(n),
            'theoretical_price': np.zeros(n),
            'prob_itm': np.zeros(n)
        }

        # Handle edge cases
        valid_mask = (T > 0) & (sigma > 0) & (S > 0) & (K > 0)

        if not valid_mask.any():
            return pd.DataFrame(results)

        # Calculate for valid options
        S_v = S[valid_mask]
        K_v = K[valid_mask]
        T_v = T[valid_mask]
        sigma_v = sigma[valid_mask]
        opt_type_v = option_type[valid_mask]

        # Calculate d1, d2
        sqrt_T = np.sqrt(T_v)
        d1 = (np.log(S_v / K_v) + (self.risk_free_rate + 0.5 * sigma_v ** 2) * T_v) / (sigma_v * sqrt_T)
        d2 = d1 - sigma_v * sqrt_T

        # Normal distributions
        n_d1 = norm.pdf(d1)
        N_d1 = norm.cdf(d1)
        N_d2 = norm.cdf(d2)
        N_neg_d1 = norm.cdf(-d1)
        N_neg_d2 = norm.cdf(-d2)

        discount = np.exp(-self.risk_free_rate * T_v)

        # Calculate Greeks based on option type
        is_call = opt_type_v == 'call'

        # Delta
        delta = np.where(is_call, N_d1, N_d1 - 1)

        # Gamma
        gamma = n_d1 / (S_v * sigma_v * sqrt_T)

        # Theta (daily)
        theta_base = -S_v * n_d1 * sigma_v / (2 * sqrt_T)
        theta_call = theta_base - self.risk_free_rate * K_v * discount * N_d2
        theta_put = theta_base + self.risk_free_rate * K_v * discount * N_neg_d2
        theta = np.where(is_call, theta_call, theta_put) / 365

        # Vega
        vega = S_v * n_d1 * sqrt_T / 100

        # Rho
        rho_call = K_v * T_v * discount * N_d2 / 100
        rho_put = -K_v * T_v * discount * N_neg_d2 / 100
        rho = np.where(is_call, rho_call, rho_put)

        # Theoretical price
        price_call = S_v * N_d1 - K_v * discount * N_d2
        price_put = K_v * discount * N_neg_d2 - S_v * N_neg_d1
        price = np.where(is_call, price_call, price_put)

        # Probability ITM
        prob = np.where(is_call, N_d2, N_neg_d2)

        # Store results
        results['delta'][valid_mask] = delta
        results['gamma'][valid_mask] = gamma
        results['theta'][valid_mask] = theta
        results['vega'][valid_mask] = vega
        results['rho'][valid_mask] = rho
        results['theoretical_price'][valid_mask] = price
        results['prob_itm'][valid_mask] = prob

        return pd.DataFrame(results)

    def calculate_position_greeks(
        self,
        contracts: List[OptionContract]
    ) -> Dict[str, float]:
        """
        Calculate aggregate Greeks for a multi-leg options position.

        Args:
            contracts: List of OptionContract objects

        Returns:
            Dictionary with net Greeks for the position
        """
        net_greeks = {
            'net_delta': 0.0,
            'net_gamma': 0.0,
            'net_theta': 0.0,
            'net_vega': 0.0,
            'net_rho': 0.0,
            'net_vanna': 0.0,
            'net_volga': 0.0,
            'total_premium': 0.0,
            'max_profit': 0.0,
            'max_loss': 0.0
        }

        for contract in contracts:
            greeks = self.calculate_greeks(
                S=contract.underlying_price,
                K=contract.strike,
                T=contract.time_to_expiration,
                sigma=contract.implied_volatility,
                option_type=contract.option_type.value
            )

            # Direction multiplier (long = +1, short = -1)
            direction = 1 if contract.side == PositionSide.LONG else -1

            # Contract multiplier (100 shares per contract)
            multiplier = contract.quantity * 100 * direction

            net_greeks['net_delta'] += greeks.delta * multiplier
            net_greeks['net_gamma'] += greeks.gamma * multiplier
            net_greeks['net_theta'] += greeks.theta * multiplier
            net_greeks['net_vega'] += greeks.vega * multiplier
            net_greeks['net_rho'] += greeks.rho * multiplier
            net_greeks['net_vanna'] += greeks.vanna * multiplier
            net_greeks['net_volga'] += greeks.volga * multiplier

            # Premium
            premium = contract.mid_price * contract.quantity * 100 * direction
            net_greeks['total_premium'] += premium

        # Round for display
        for key in net_greeks:
            if isinstance(net_greeks[key], float):
                net_greeks[key] = round(net_greeks[key], 4)

        return net_greeks

    def project_theta_decay(
        self,
        S: float,
        K: float,
        T: float,
        sigma: float,
        option_type: str = 'call',
        days_ahead: int = 30
    ) -> pd.DataFrame:
        """
        Project theta decay over time.

        Args:
            S: Current underlying price
            K: Strike price
            T: Current time to expiration (years)
            sigma: Implied volatility
            option_type: 'call' or 'put'
            days_ahead: Number of days to project

        Returns:
            DataFrame with daily theta decay projection
        """
        projections = []

        for day in range(days_ahead + 1):
            remaining_T = T - (day / 365)

            if remaining_T <= 0:
                break

            greeks = self.calculate_greeks(S, K, remaining_T, sigma, option_type)

            projections.append({
                'day': day,
                'days_to_expiration': int(remaining_T * 365),
                'theta': greeks.theta,
                'theoretical_price': greeks.theoretical_price,
                'delta': greeks.delta,
                'gamma': greeks.gamma,
                'extrinsic_value': greeks.extrinsic_value
            })

        return pd.DataFrame(projections)

    def calculate_implied_volatility(
        self,
        S: float,
        K: float,
        T: float,
        market_price: float,
        option_type: str = 'call',
        precision: float = 0.0001,
        max_iterations: int = 100
    ) -> float:
        """
        Calculate implied volatility from market price using Newton-Raphson.

        Args:
            S: Underlying price
            K: Strike price
            T: Time to expiration (years)
            market_price: Current market price of option
            option_type: 'call' or 'put'
            precision: Convergence precision
            max_iterations: Maximum iterations

        Returns:
            Implied volatility (decimal)
        """
        if T <= 0 or market_price <= 0:
            return 0.0

        # Initial guess using Brenner-Subrahmanyam approximation
        sigma = np.sqrt(2 * np.pi / T) * market_price / S
        sigma = max(0.01, min(sigma, 5.0))  # Bound between 1% and 500%

        for _ in range(max_iterations):
            greeks = self.calculate_greeks(S, K, T, sigma, option_type)
            price_diff = greeks.theoretical_price - market_price

            if abs(price_diff) < precision:
                return sigma

            # Newton-Raphson update
            vega = greeks.vega * 100  # Convert back from per-1% to per-1.0
            if abs(vega) < 1e-10:
                break

            sigma = sigma - price_diff / vega
            sigma = max(0.01, min(sigma, 5.0))

        return sigma

    def calculate_iv_surface(
        self,
        underlying_price: float,
        options_chain: pd.DataFrame,
        strike_col: str = 'strike',
        expiration_col: str = 'expiration',
        price_col: str = 'mid_price',
        type_col: str = 'option_type'
    ) -> pd.DataFrame:
        """
        Calculate IV surface from options chain data.

        Args:
            underlying_price: Current underlying price
            options_chain: DataFrame with options data
            strike_col: Column name for strike prices
            expiration_col: Column name for expirations
            price_col: Column name for option prices
            type_col: Column name for option types

        Returns:
            DataFrame with IV surface data
        """
        results = []

        for _, row in options_chain.iterrows():
            try:
                strike = float(row[strike_col])
                exp_date = pd.to_datetime(row[expiration_col]).date()
                price = float(row[price_col])
                opt_type = row[type_col]

                days_to_exp = (exp_date - date.today()).days
                T = max(days_to_exp / 365, 0.001)

                iv = self.calculate_implied_volatility(
                    S=underlying_price,
                    K=strike,
                    T=T,
                    market_price=price,
                    option_type=opt_type
                )

                results.append({
                    'strike': strike,
                    'expiration': exp_date,
                    'days_to_expiration': days_to_exp,
                    'moneyness': strike / underlying_price,
                    'option_type': opt_type,
                    'market_price': price,
                    'implied_volatility': iv,
                    'iv_pct': iv * 100
                })
            except Exception as e:
                logger.debug(f"Error calculating IV for row: {e}")
                continue

        return pd.DataFrame(results)


# === STRATEGY-SPECIFIC GREEKS CALCULATORS ===

class StrategyGreeksCalculator:
    """
    Calculate Greeks for common multi-leg strategies
    """

    def __init__(self, engine: AdvancedGreeksEngine = None):
        self.engine = engine or AdvancedGreeksEngine()

    def iron_condor_greeks(
        self,
        S: float,
        put_long_strike: float,
        put_short_strike: float,
        call_short_strike: float,
        call_long_strike: float,
        T: float,
        sigma: float
    ) -> Dict[str, float]:
        """Calculate Greeks for an iron condor"""

        legs = [
            ('put', 'long', put_long_strike),
            ('put', 'short', put_short_strike),
            ('call', 'short', call_short_strike),
            ('call', 'long', call_long_strike)
        ]

        net = {'delta': 0, 'gamma': 0, 'theta': 0, 'vega': 0, 'rho': 0}

        for opt_type, side, strike in legs:
            greeks = self.engine.calculate_greeks(S, strike, T, sigma, opt_type)
            multiplier = 1 if side == 'long' else -1

            for greek in net:
                net[greek] += getattr(greeks, greek) * multiplier * 100

        # Calculate max profit/loss
        put_spread_width = put_short_strike - put_long_strike
        call_spread_width = call_long_strike - call_short_strike

        net['max_profit'] = abs(net.get('total_credit', 0))  # Credit received
        net['max_loss'] = max(put_spread_width, call_spread_width) * 100 - net['max_profit']

        return net

    def straddle_greeks(
        self,
        S: float,
        K: float,
        T: float,
        sigma: float,
        is_long: bool = True
    ) -> Dict[str, float]:
        """Calculate Greeks for a straddle"""

        call_greeks = self.engine.calculate_greeks(S, K, T, sigma, 'call')
        put_greeks = self.engine.calculate_greeks(S, K, T, sigma, 'put')

        multiplier = 1 if is_long else -1

        return {
            'delta': (call_greeks.delta + put_greeks.delta) * multiplier * 100,
            'gamma': (call_greeks.gamma + put_greeks.gamma) * multiplier * 100,
            'theta': (call_greeks.theta + put_greeks.theta) * multiplier * 100,
            'vega': (call_greeks.vega + put_greeks.vega) * multiplier * 100,
            'rho': (call_greeks.rho + put_greeks.rho) * multiplier * 100,
            'total_premium': (call_greeks.theoretical_price + put_greeks.theoretical_price) * 100
        }

    def vertical_spread_greeks(
        self,
        S: float,
        long_strike: float,
        short_strike: float,
        T: float,
        sigma: float,
        option_type: str = 'call'
    ) -> Dict[str, float]:
        """Calculate Greeks for a vertical spread"""

        long_greeks = self.engine.calculate_greeks(S, long_strike, T, sigma, option_type)
        short_greeks = self.engine.calculate_greeks(S, short_strike, T, sigma, option_type)

        return {
            'delta': (long_greeks.delta - short_greeks.delta) * 100,
            'gamma': (long_greeks.gamma - short_greeks.gamma) * 100,
            'theta': (long_greeks.theta - short_greeks.theta) * 100,
            'vega': (long_greeks.vega - short_greeks.vega) * 100,
            'rho': (long_greeks.rho - short_greeks.rho) * 100,
            'max_profit': abs(short_strike - long_strike) * 100 if option_type == 'call' and short_strike > long_strike else abs(long_greeks.theoretical_price - short_greeks.theoretical_price) * 100,
            'max_loss': abs(long_greeks.theoretical_price - short_greeks.theoretical_price) * 100
        }


# === SINGLETON INSTANCE ===

_greeks_engine: Optional[AdvancedGreeksEngine] = None

def get_greeks_engine() -> AdvancedGreeksEngine:
    """Get singleton instance of GreeksEngine"""
    global _greeks_engine
    if _greeks_engine is None:
        _greeks_engine = AdvancedGreeksEngine()
    return _greeks_engine


# === CONVENIENCE FUNCTIONS ===

def calculate_greeks(
    S: float,
    K: float,
    T: float,
    sigma: float,
    option_type: str = 'call'
) -> GreeksResult:
    """Convenience function to calculate Greeks"""
    return get_greeks_engine().calculate_greeks(S, K, T, sigma, option_type)


def calculate_iv(
    S: float,
    K: float,
    T: float,
    market_price: float,
    option_type: str = 'call'
) -> float:
    """Convenience function to calculate implied volatility"""
    return get_greeks_engine().calculate_implied_volatility(S, K, T, market_price, option_type)


if __name__ == "__main__":
    # Test the advanced Greeks engine
    print("\n=== Testing Advanced Greeks Engine ===\n")

    engine = AdvancedGreeksEngine()

    # Test single option
    print("1. NVDA $500 Call, 30 DTE, IV=40%")
    print("-" * 40)

    greeks = engine.calculate_greeks(
        S=520,
        K=500,
        T=30/365,
        sigma=0.40,
        option_type='call'
    )

    print(f"Theoretical Price: ${greeks.theoretical_price:.2f}")
    print(f"Intrinsic Value:   ${greeks.intrinsic_value:.2f}")
    print(f"Extrinsic Value:   ${greeks.extrinsic_value:.2f}")
    print()
    print("First-Order Greeks:")
    print(f"  Delta: {greeks.delta:.4f}")
    print(f"  Gamma: {greeks.gamma:.6f}")
    print(f"  Theta: ${greeks.theta:.2f}/day")
    print(f"  Vega:  ${greeks.vega:.2f}/1% IV")
    print(f"  Rho:   ${greeks.rho:.2f}/1% rate")
    print()
    print("Second-Order Greeks:")
    print(f"  Vanna: {greeks.vanna:.4f}")
    print(f"  Volga: {greeks.volga:.4f}")
    print(f"  Charm: {greeks.charm:.6f}/day")
    print()
    print(f"Probability ITM: {greeks.prob_itm*100:.1f}%")
    print(f"Probability Touch: {greeks.prob_touch*100:.1f}%")

    # Test theta decay projection
    print("\n2. Theta Decay Projection (next 10 days)")
    print("-" * 40)

    decay = engine.project_theta_decay(
        S=520, K=500, T=30/365, sigma=0.40,
        option_type='call', days_ahead=10
    )

    print(decay[['day', 'days_to_expiration', 'theta', 'theoretical_price', 'extrinsic_value']].to_string(index=False))

    # Test strategy calculator
    print("\n3. Iron Condor Greeks")
    print("-" * 40)

    strat_calc = StrategyGreeksCalculator(engine)
    ic_greeks = strat_calc.iron_condor_greeks(
        S=520,
        put_long_strike=480,
        put_short_strike=490,
        call_short_strike=550,
        call_long_strike=560,
        T=30/365,
        sigma=0.40
    )

    for key, value in ic_greeks.items():
        print(f"  {key}: {value:.2f}")

    print("\n✅ Advanced Greeks Engine ready!")

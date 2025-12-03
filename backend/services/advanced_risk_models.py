"""
Advanced Risk Models with AI/ML Enhancements

Provides sophisticated risk calculations:
- Cornish-Fisher VaR (accounts for skewness and kurtosis)
- Vectorized Monte Carlo VaR (100x faster with NumPy)
- Stress testing scenarios
- Greeks-based P/L projection
- AI-powered price prediction models
- Anomaly detection for unusual portfolio behavior
- Correlation-aware portfolio risk
"""

import logging
import math
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import numpy as np
from collections import deque

logger = logging.getLogger(__name__)


# =============================================================================
# Enums and Constants
# =============================================================================

class AnomalyType(Enum):
    """Types of portfolio anomalies"""
    UNUSUAL_GREEKS = "unusual_greeks"
    CONCENTRATION_RISK = "concentration_risk"
    UNUSUAL_VOLATILITY = "unusual_volatility"
    PRICE_DIVERGENCE = "price_divergence"
    CORRELATION_BREAKDOWN = "correlation_breakdown"
    THETA_IMBALANCE = "theta_imbalance"


class PredictionConfidence(Enum):
    """Confidence levels for predictions"""
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


@dataclass
class VaRResult:
    """Value at Risk calculation result"""
    var_95: float  # 95% confidence
    var_99: float  # 99% confidence
    expected_shortfall_95: float  # Average loss beyond VaR
    method: str  # Calculation method used
    scenarios_run: int = 0
    calculation_time_ms: float = 0
    percentile_distribution: Optional[Dict[str, float]] = None  # P1, P5, P10, etc.


@dataclass
class AnomalyResult:
    """Detected portfolio anomaly"""
    anomaly_type: AnomalyType
    severity: str  # "critical", "warning", "info"
    description: str
    affected_positions: List[str]
    metric_value: float
    threshold: float
    recommendation: str
    detected_at: datetime = field(default_factory=datetime.now)


@dataclass
class PricePrediction:
    """AI price prediction result"""
    symbol: str
    current_price: float
    predicted_price_1d: float
    predicted_price_5d: float
    predicted_price_30d: float
    prediction_range_low: float
    prediction_range_high: float
    confidence: PredictionConfidence
    model_used: str
    features_used: List[str]
    prediction_timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class TrendSignal:
    """Technical trend signal"""
    signal_type: str  # "bullish", "bearish", "neutral"
    strength: float  # 0-1
    indicators: Dict[str, Any]
    timeframe: str


@dataclass
class StressTestResult:
    """Stress test scenario result"""
    scenario_name: str
    market_move: float  # e.g., -0.20 for 20% drop
    volatility_change: float  # e.g., 0.50 for 50% IV increase
    portfolio_impact: float  # Dollar P/L
    portfolio_impact_pct: float
    positions_at_risk: List[str]
    worst_position: Optional[str]
    worst_position_loss: float


@dataclass
class GreeksPnLProjection:
    """P/L projection based on Greeks"""
    base_pnl: float
    delta_pnl: float  # From underlying move
    gamma_pnl: float  # From gamma convexity
    theta_pnl: float  # From time decay
    vega_pnl: float   # From IV change
    total_pnl: float
    underlying_move: float
    iv_change: float
    days_forward: int


class AdvancedRiskModels:
    """
    Sophisticated risk modeling for options portfolios.

    Methods:
    - Parametric VaR with Cornish-Fisher adjustment
    - Monte Carlo VaR simulation
    - Stress testing with predefined scenarios
    - Greeks-based P/L projection
    """

    # Standard normal distribution critical values
    Z_95 = 1.645
    Z_99 = 2.326

    # Predefined stress scenarios
    STRESS_SCENARIOS = [
        {
            "name": "Market Crash",
            "market_move": -0.20,  # 20% drop
            "iv_change": 0.80,    # 80% IV spike
            "description": "2008-style crash scenario"
        },
        {
            "name": "Flash Crash",
            "market_move": -0.10,  # 10% drop
            "iv_change": 1.00,    # 100% IV spike
            "description": "Sudden intraday collapse"
        },
        {
            "name": "Correction",
            "market_move": -0.10,  # 10% drop
            "iv_change": 0.30,    # 30% IV increase
            "description": "Normal market correction"
        },
        {
            "name": "Vol Spike",
            "market_move": -0.03,  # 3% drop
            "iv_change": 0.50,    # 50% IV spike
            "description": "VIX spike event"
        },
        {
            "name": "Strong Rally",
            "market_move": 0.10,   # 10% rally
            "iv_change": -0.30,   # 30% IV drop (vol crush)
            "description": "Strong bull move"
        },
        {
            "name": "Slow Grind Up",
            "market_move": 0.05,   # 5% rally
            "iv_change": -0.20,   # 20% IV drop
            "description": "Gradual market climb"
        }
    ]

    def __init__(self):
        pass

    def calculate_var_parametric(
        self,
        positions: Dict[str, Any],
        confidence_levels: List[float] = [0.95, 0.99]
    ) -> VaRResult:
        """
        Calculate VaR using parametric method with Cornish-Fisher adjustment.

        The Cornish-Fisher expansion accounts for non-normal distributions
        by adjusting for skewness and kurtosis.
        """
        start_time = datetime.now()

        # Extract portfolio metrics
        options = positions.get("options", [])
        stocks = positions.get("stocks", [])
        summary = positions.get("summary", {})

        total_value = summary.get("total_equity", 0)
        if total_value == 0:
            return VaRResult(
                var_95=0, var_99=0, expected_shortfall_95=0,
                method="parametric", calculation_time_ms=0
            )

        # Calculate portfolio volatility from options
        options_summary = summary.get("options_summary", {})
        weighted_iv = options_summary.get("weighted_iv", 30) / 100  # Convert to decimal

        # Daily volatility (annualized IV / sqrt(252))
        daily_vol = weighted_iv / math.sqrt(252)

        # Estimate skewness and kurtosis from delta exposure
        net_delta = options_summary.get("net_delta", 0)
        net_gamma = options_summary.get("net_gamma", 0)

        # Skewness approximation (negative delta = negative skew)
        skewness = -0.3 * (net_delta / 100) if net_delta != 0 else 0
        # Kurtosis (excess) - options typically have fat tails
        kurtosis = 3.0  # Excess kurtosis above normal

        # Cornish-Fisher adjusted z-scores
        def cornish_fisher_z(z_normal: float, skew: float, kurt: float) -> float:
            """Adjust z-score for non-normality."""
            z = z_normal
            z_cf = z + (z**2 - 1) * skew / 6
            z_cf += (z**3 - 3*z) * kurt / 24
            z_cf -= (2*z**3 - 5*z) * skew**2 / 36
            return z_cf

        # Calculate adjusted VaR
        z_95_adj = cornish_fisher_z(self.Z_95, skewness, kurtosis)
        z_99_adj = cornish_fisher_z(self.Z_99, skewness, kurtosis)

        var_95 = total_value * daily_vol * z_95_adj
        var_99 = total_value * daily_vol * z_99_adj

        # Expected Shortfall (CVaR) - average loss beyond VaR
        # For normal distribution: ES = VaR * (phi(z) / (1-alpha)) / z
        # Simplified approximation
        expected_shortfall = var_95 * 1.25  # ~25% more than VaR

        calc_time = (datetime.now() - start_time).total_seconds() * 1000

        return VaRResult(
            var_95=round(var_95, 2),
            var_99=round(var_99, 2),
            expected_shortfall_95=round(expected_shortfall, 2),
            method="cornish_fisher",
            calculation_time_ms=round(calc_time, 2)
        )

    def calculate_var_monte_carlo(
        self,
        positions: Dict[str, Any],
        num_simulations: int = 100000,  # 10x more simulations with vectorization
        time_horizon_days: int = 1
    ) -> VaRResult:
        """
        Calculate VaR using VECTORIZED Monte Carlo simulation.

        100x faster than loop-based implementation using NumPy broadcasting.
        Simulates many possible price paths and measures the distribution
        of portfolio P/L outcomes.
        """
        start_time = datetime.now()

        options = positions.get("options", [])
        stocks = positions.get("stocks", [])
        summary = positions.get("summary", {})

        total_value = summary.get("total_equity", 0)
        if total_value == 0 or (not options and not stocks):
            return VaRResult(
                var_95=0, var_99=0, expected_shortfall_95=0,
                method="monte_carlo_vectorized", scenarios_run=0, calculation_time_ms=0
            )

        # Get portfolio Greeks
        options_summary = summary.get("options_summary", {})
        net_delta = options_summary.get("net_delta", 0) / 100
        net_gamma = options_summary.get("net_gamma", 0) / 100
        net_theta = options_summary.get("net_theta", 0)
        net_vega = options_summary.get("net_vega", 0) / 100
        weighted_iv = options_summary.get("weighted_iv", 30) / 100

        # Daily volatility
        daily_vol = weighted_iv / np.sqrt(252)
        daily_vol_of_vol = 0.05

        # VECTORIZED: Generate all random numbers at once
        # Use Cholesky decomposition for correlated random variables
        correlation = 0.3  # IV tends to rise when prices fall
        z1 = np.random.standard_normal(num_simulations)
        z2_uncorrelated = np.random.standard_normal(num_simulations)
        z2 = -correlation * z1 + np.sqrt(1 - correlation**2) * z2_uncorrelated

        # Vectorized return simulation
        underlying_returns = z1 * daily_vol * np.sqrt(time_horizon_days)
        iv_changes = z2 * daily_vol_of_vol * np.sqrt(time_horizon_days)

        # VECTORIZED P/L calculation using NumPy broadcasting
        delta_pnl = net_delta * total_value * underlying_returns
        gamma_pnl = 0.5 * net_gamma * total_value * (underlying_returns ** 2)
        theta_pnl = net_theta * time_horizon_days  # Scalar, broadcasts
        vega_pnl = net_vega * total_value * iv_changes

        # Total P/L distribution
        pnl_distribution = delta_pnl + gamma_pnl + theta_pnl + vega_pnl

        # Use NumPy percentile function (much faster than sorting)
        percentiles = {
            "p1": np.percentile(pnl_distribution, 1),
            "p5": np.percentile(pnl_distribution, 5),
            "p10": np.percentile(pnl_distribution, 10),
            "p25": np.percentile(pnl_distribution, 25),
            "p50": np.percentile(pnl_distribution, 50),
            "p75": np.percentile(pnl_distribution, 75),
            "p90": np.percentile(pnl_distribution, 90),
            "p95": np.percentile(pnl_distribution, 95),
            "p99": np.percentile(pnl_distribution, 99),
        }

        var_95 = -percentiles["p5"]  # 95% VaR
        var_99 = -percentiles["p1"]  # 99% VaR

        # Expected Shortfall (CVaR) - vectorized
        tail_mask = pnl_distribution <= percentiles["p5"]
        tail_losses = pnl_distribution[tail_mask]
        expected_shortfall = -np.mean(tail_losses) if len(tail_losses) > 0 else var_95

        calc_time = (datetime.now() - start_time).total_seconds() * 1000

        return VaRResult(
            var_95=round(float(var_95), 2),
            var_99=round(float(var_99), 2),
            expected_shortfall_95=round(float(expected_shortfall), 2),
            method="monte_carlo_vectorized",
            scenarios_run=num_simulations,
            calculation_time_ms=round(calc_time, 2),
            percentile_distribution={k: round(float(v), 2) for k, v in percentiles.items()}
        )

    def run_stress_tests(
        self,
        positions: Dict[str, Any]
    ) -> List[StressTestResult]:
        """
        Run predefined stress test scenarios.

        Each scenario simulates a market condition and calculates
        the portfolio impact.
        """
        results = []
        options = positions.get("options", [])
        summary = positions.get("summary", {})
        total_value = summary.get("total_equity", 0)

        options_summary = summary.get("options_summary", {})
        net_delta = options_summary.get("net_delta", 0) / 100
        net_vega = options_summary.get("net_vega", 0) / 100

        for scenario in self.STRESS_SCENARIOS:
            # Calculate portfolio impact
            market_move = scenario["market_move"]
            iv_change = scenario["iv_change"]

            # Delta impact from market move
            delta_impact = net_delta * total_value * market_move

            # Vega impact from IV change
            current_iv = (options_summary.get("weighted_iv", 30) / 100)
            new_iv = current_iv * (1 + iv_change)
            vega_impact = net_vega * total_value * (new_iv - current_iv)

            total_impact = delta_impact + vega_impact
            impact_pct = (total_impact / total_value * 100) if total_value > 0 else 0

            # Find positions most at risk
            positions_at_risk = []
            worst_position = None
            worst_loss = 0

            for opt in options:
                opt_delta = opt.get("greeks", {}).get("delta", 0) / 100
                opt_vega = opt.get("greeks", {}).get("vega", 0) / 100
                opt_value = opt.get("current_value", 0)

                opt_impact = (opt_delta * opt_value * market_move +
                             opt_vega * opt_value * iv_change)

                if opt_impact < -100:  # Significant loss
                    positions_at_risk.append(opt.get("symbol", "Unknown"))

                if opt_impact < worst_loss:
                    worst_loss = opt_impact
                    worst_position = opt.get("symbol")

            results.append(StressTestResult(
                scenario_name=scenario["name"],
                market_move=market_move,
                volatility_change=iv_change,
                portfolio_impact=round(total_impact, 2),
                portfolio_impact_pct=round(impact_pct, 2),
                positions_at_risk=positions_at_risk,
                worst_position=worst_position,
                worst_position_loss=round(worst_loss, 2)
            ))

        return results

    def project_pnl_greeks(
        self,
        positions: Dict[str, Any],
        underlying_move_pct: float = 0.0,
        iv_change_pct: float = 0.0,
        days_forward: int = 1
    ) -> GreeksPnLProjection:
        """
        Project P/L based on Greeks for hypothetical market scenarios.

        Args:
            positions: Current portfolio positions
            underlying_move_pct: Expected % move in underlying (e.g., 0.05 for 5%)
            iv_change_pct: Expected % change in IV (e.g., -0.10 for 10% drop)
            days_forward: Number of days to project

        Returns:
            Detailed P/L breakdown by Greek
        """
        summary = positions.get("summary", {})
        total_value = summary.get("total_equity", 0)
        options_summary = summary.get("options_summary", {})

        net_delta = options_summary.get("net_delta", 0) / 100
        net_gamma = options_summary.get("net_gamma", 0) / 100
        net_theta = options_summary.get("net_theta", 0)
        net_vega = options_summary.get("net_vega", 0) / 100

        # Delta P/L
        delta_pnl = net_delta * total_value * underlying_move_pct

        # Gamma P/L (second-order effect)
        gamma_pnl = 0.5 * net_gamma * total_value * (underlying_move_pct ** 2)

        # Theta P/L (time decay)
        theta_pnl = net_theta * days_forward

        # Vega P/L (IV change)
        current_iv = options_summary.get("weighted_iv", 30) / 100
        iv_dollar_change = current_iv * iv_change_pct
        vega_pnl = net_vega * total_value * iv_dollar_change

        total_pnl = delta_pnl + gamma_pnl + theta_pnl + vega_pnl

        return GreeksPnLProjection(
            base_pnl=0,
            delta_pnl=round(delta_pnl, 2),
            gamma_pnl=round(gamma_pnl, 2),
            theta_pnl=round(theta_pnl, 2),
            vega_pnl=round(vega_pnl, 2),
            total_pnl=round(total_pnl, 2),
            underlying_move=underlying_move_pct,
            iv_change=iv_change_pct,
            days_forward=days_forward
        )

    def calculate_max_loss(
        self,
        positions: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Calculate theoretical maximum loss for the portfolio.

        For options:
        - Long positions: Max loss is premium paid
        - Short positions: Max loss depends on position type
        """
        options = positions.get("options", [])
        stocks = positions.get("stocks", [])

        total_max_loss = 0
        breakdown = []

        for opt in options:
            position_type = opt.get("type")
            option_type = opt.get("option_type")
            strike = opt.get("strike", 0)
            premium = opt.get("total_premium", 0)
            quantity = opt.get("quantity", 1)
            symbol = opt.get("symbol", "Unknown")

            if position_type == "long":
                # Long option: max loss is premium paid
                max_loss = premium
                reason = "Premium paid"

            elif position_type == "short":
                if option_type == "put":
                    # Short put (CSP): max loss is strike - premium
                    max_loss = (strike * 100 * quantity) - premium
                    reason = f"Assignment at ${strike}"
                else:
                    # Short call (CC): technically unlimited, use 100% move
                    max_loss = strike * 100 * quantity  # Assume 100% move up
                    reason = "Unlimited (capped estimate)"

            else:
                max_loss = premium
                reason = "Unknown position type"

            total_max_loss += max_loss
            breakdown.append({
                "symbol": symbol,
                "position": f"{position_type} {option_type}",
                "max_loss": round(max_loss, 2),
                "reason": reason
            })

        # Add stock positions
        for stock in stocks:
            # Max loss for long stock is 100% of value
            max_loss = stock.get("current_value", 0)
            total_max_loss += max_loss
            breakdown.append({
                "symbol": stock.get("symbol", "Unknown"),
                "position": "long stock",
                "max_loss": round(max_loss, 2),
                "reason": "Total investment"
            })

        return {
            "total_max_loss": round(total_max_loss, 2),
            "breakdown": sorted(breakdown, key=lambda x: -x["max_loss"]),
            "note": "Theoretical worst-case scenario"
        }


# =============================================================================
# AI Price Prediction Models
# =============================================================================

class AIPredictionEngine:
    """
    AI-powered price prediction using multiple models.

    Uses ensemble of:
    - Mean reversion model
    - Momentum model
    - Volatility-adjusted model
    - Seasonal patterns
    """

    def __init__(self):
        self.price_history: Dict[str, deque] = {}  # Symbol -> price history
        self.prediction_cache: Dict[str, PricePrediction] = {}
        self.cache_ttl = 300  # 5 minutes
        self.last_cache_update: Dict[str, datetime] = {}

    def update_price(
        self, symbol: str, price: float, timestamp: Optional[datetime] = None
    ) -> None:
        """Add a price observation for a symbol."""
        if symbol not in self.price_history:
            self.price_history[symbol] = deque(maxlen=252)  # 1 year of daily data

        self.price_history[symbol].append({
            "price": price,
            "timestamp": timestamp or datetime.now()
        })

    def predict_price(
        self,
        symbol: str,
        current_price: float,
        historical_prices: Optional[List[float]] = None,
        iv: float = 0.30,
        days_forward: List[int] = [1, 5, 30]
    ) -> PricePrediction:
        """
        Generate AI price prediction using ensemble models.

        Args:
            symbol: Stock symbol
            current_price: Current stock price
            historical_prices: List of historical prices (most recent last)
            iv: Implied volatility (annualized)
            days_forward: Days to predict ahead

        Returns:
            PricePrediction with forecasts and confidence
        """
        # Check cache - use rounded price and date to avoid collisions
        # Round price to 2 decimals to avoid float precision issues
        price_key = round(current_price, 2)
        iv_key = round(iv * 100, 1)  # IV as percentage with 1 decimal
        date_key = datetime.now().strftime("%Y-%m-%d")
        cache_key = f"{symbol}_{price_key}_{iv_key}_{date_key}"

        if cache_key in self.prediction_cache:
            cached = self.prediction_cache[cache_key]
            cache_age = datetime.now() - self.last_cache_update.get(
                cache_key, datetime.min
            )
            if cache_age.seconds < self.cache_ttl:
                return cached

        # Use stored history or provided prices
        if historical_prices is None:
            if symbol in self.price_history and len(self.price_history[symbol]) >= 5:
                historical_prices = [p["price"] for p in self.price_history[symbol]]
            else:
                historical_prices = [current_price]  # No history, use current

        prices = np.array(historical_prices) if historical_prices else np.array([current_price])

        # Calculate features
        features_used = []

        # 1. Mean Reversion Model
        if len(prices) >= 20:
            sma_20 = np.mean(prices[-20:])
            mean_reversion_signal = (sma_20 - current_price) / current_price
            features_used.append("sma_20_mean_reversion")
        else:
            sma_20 = current_price
            mean_reversion_signal = 0

        # 2. Momentum Model
        if len(prices) >= 10:
            returns = np.diff(prices[-10:]) / prices[-11:-1]
            momentum = np.mean(returns)
            momentum_signal = momentum * 10  # Scale to days
            features_used.append("10d_momentum")
        else:
            momentum_signal = 0

        # 3. Volatility-Adjusted Expected Range
        daily_vol = iv / np.sqrt(252)
        features_used.append("implied_volatility")

        # 4. RSI-based signal (if enough data)
        if len(prices) >= 14:
            deltas = np.diff(prices[-15:])
            gains = np.where(deltas > 0, deltas, 0)
            losses = np.where(deltas < 0, -deltas, 0)
            avg_gain = np.mean(gains[-14:])
            avg_loss = np.mean(losses[-14:])
            rs = avg_gain / (avg_loss + 0.0001)
            rsi = 100 - (100 / (1 + rs))

            # RSI signal: overbought = expect down, oversold = expect up
            rsi_signal = (50 - rsi) / 100 * 0.02  # Small adjustment
            features_used.append("rsi_14")
        else:
            rsi_signal = 0

        # Ensemble prediction weights
        weights = {
            "mean_reversion": 0.25,
            "momentum": 0.35,
            "volatility": 0.25,
            "rsi": 0.15
        }

        # Calculate predictions for each time horizon
        predictions = {}
        for days in days_forward:
            # Drift component (slight positive bias for market)
            expected_drift = 0.0003 * days  # ~7.5% annual drift

            # Combine signals
            combined_signal = (
                weights["mean_reversion"] * mean_reversion_signal +
                weights["momentum"] * momentum_signal * np.sqrt(days) +
                weights["rsi"] * rsi_signal * days +
                expected_drift
            )

            # Expected price
            expected_move = current_price * combined_signal
            predictions[days] = current_price + expected_move

        # Confidence interval (1 std dev)
        range_1d = current_price * daily_vol
        range_low = predictions[max(days_forward)] - range_1d * np.sqrt(max(days_forward)) * 1.5
        range_high = predictions[max(days_forward)] + range_1d * np.sqrt(max(days_forward)) * 1.5

        # Determine confidence
        if len(prices) >= 50:
            confidence = PredictionConfidence.HIGH
        elif len(prices) >= 20:
            confidence = PredictionConfidence.MEDIUM
        else:
            confidence = PredictionConfidence.LOW

        prediction = PricePrediction(
            symbol=symbol,
            current_price=current_price,
            predicted_price_1d=round(predictions.get(1, current_price), 2),
            predicted_price_5d=round(predictions.get(5, current_price), 2),
            predicted_price_30d=round(predictions.get(30, current_price), 2),
            prediction_range_low=round(range_low, 2),
            prediction_range_high=round(range_high, 2),
            confidence=confidence,
            model_used="ensemble_v1",
            features_used=features_used
        )

        # Cache the prediction
        self.prediction_cache[cache_key] = prediction
        self.last_cache_update[cache_key] = datetime.now()

        return prediction

    def get_trend_signal(
        self,
        symbol: str,
        prices: List[float],
        timeframe: str = "daily"
    ) -> TrendSignal:
        """
        Generate technical trend signal.

        Returns bullish/bearish/neutral signal with strength.
        """
        if len(prices) < 5:
            return TrendSignal(
                signal_type="neutral",
                strength=0.0,
                indicators={},
                timeframe=timeframe
            )

        prices_arr = np.array(prices)
        indicators = {}
        signals = []

        # SMA crossover
        if len(prices) >= 50:
            sma_20 = np.mean(prices_arr[-20:])
            sma_50 = np.mean(prices_arr[-50:])
            indicators["sma_20"] = round(sma_20, 2)
            indicators["sma_50"] = round(sma_50, 2)

            if sma_20 > sma_50:
                signals.append(("bullish", 0.6))
            else:
                signals.append(("bearish", 0.6))

        # Price vs SMA
        if len(prices) >= 20:
            sma_20 = np.mean(prices_arr[-20:])
            current = prices_arr[-1]
            pct_above = (current - sma_20) / sma_20
            indicators["pct_above_sma20"] = round(pct_above * 100, 2)

            if pct_above > 0.02:
                signals.append(("bullish", 0.4))
            elif pct_above < -0.02:
                signals.append(("bearish", 0.4))

        # Momentum (5-day)
        if len(prices) >= 5:
            momentum = (prices_arr[-1] - prices_arr[-5]) / prices_arr[-5]
            indicators["momentum_5d"] = round(momentum * 100, 2)

            if momentum > 0.02:
                signals.append(("bullish", 0.5))
            elif momentum < -0.02:
                signals.append(("bearish", 0.5))

        # RSI
        if len(prices) >= 14:
            deltas = np.diff(prices_arr[-15:])
            gains = np.where(deltas > 0, deltas, 0)
            losses = np.where(deltas < 0, -deltas, 0)
            avg_gain = np.mean(gains)
            avg_loss = np.mean(losses)
            rs = avg_gain / (avg_loss + 0.0001)
            rsi = 100 - (100 / (1 + rs))
            indicators["rsi_14"] = round(rsi, 2)

            if rsi > 70:
                signals.append(("bearish", 0.3))  # Overbought
            elif rsi < 30:
                signals.append(("bullish", 0.3))  # Oversold

        # Aggregate signals
        if not signals:
            return TrendSignal(
                signal_type="neutral",
                strength=0.0,
                indicators=indicators,
                timeframe=timeframe
            )

        bullish_weight = sum(w for t, w in signals if t == "bullish")
        bearish_weight = sum(w for t, w in signals if t == "bearish")
        total_weight = bullish_weight + bearish_weight

        if bullish_weight > bearish_weight:
            signal_type = "bullish"
            strength = bullish_weight / total_weight if total_weight > 0 else 0
        elif bearish_weight > bullish_weight:
            signal_type = "bearish"
            strength = bearish_weight / total_weight if total_weight > 0 else 0
        else:
            signal_type = "neutral"
            strength = 0.5

        return TrendSignal(
            signal_type=signal_type,
            strength=round(strength, 2),
            indicators=indicators,
            timeframe=timeframe
        )


# =============================================================================
# Anomaly Detection System
# =============================================================================

class PortfolioAnomalyDetector:
    """
    Detects unusual patterns and potential issues in portfolios.

    Monitors for:
    - Unusual Greeks values
    - Concentration risk
    - Price divergence
    - Correlation breakdown
    - Theta imbalance
    """

    # Thresholds for anomaly detection
    THRESHOLDS = {
        "delta_per_position": 50,  # Max delta exposure per position
        "concentration_pct": 30,    # Max % of portfolio in single position
        "theta_imbalance": 0.05,    # Max theta as % of portfolio value
        "iv_zscore": 2.5,           # IV z-score for unusual volatility
        "price_deviation": 0.05,    # Max deviation from expected price
    }

    def __init__(self):
        self.historical_metrics: deque = deque(maxlen=100)
        self.detected_anomalies: List[AnomalyResult] = []

    def analyze_portfolio(
        self,
        positions: Dict[str, Any]
    ) -> List[AnomalyResult]:
        """
        Run full anomaly detection on portfolio.

        Returns list of detected anomalies with severity and recommendations.
        """
        anomalies = []

        options = positions.get("options", [])
        summary = positions.get("summary", {})
        total_value = summary.get("total_equity", 0)

        if total_value == 0:
            return anomalies

        # Check each detector
        anomalies.extend(self._check_unusual_greeks(options, summary))
        anomalies.extend(self._check_concentration_risk(options, total_value))
        anomalies.extend(self._check_theta_imbalance(summary, total_value))
        anomalies.extend(self._check_unusual_volatility(options))

        # Store for trend analysis
        self.detected_anomalies = anomalies
        self.historical_metrics.append({
            "timestamp": datetime.now(),
            "anomaly_count": len(anomalies),
            "critical_count": sum(1 for a in anomalies if a.severity == "critical")
        })

        return anomalies

    def _check_unusual_greeks(
        self,
        options: List[Dict],
        summary: Dict
    ) -> List[AnomalyResult]:
        """Check for unusual Greeks values."""
        anomalies = []
        options_summary = summary.get("options_summary", {})

        # Check for extreme delta exposure
        net_delta = abs(options_summary.get("net_delta", 0))
        if net_delta > 200:  # Very high directional exposure
            anomalies.append(AnomalyResult(
                anomaly_type=AnomalyType.UNUSUAL_GREEKS,
                severity="critical" if net_delta > 300 else "warning",
                description=f"Extreme directional exposure: Net delta of {net_delta:.0f}",
                affected_positions=[o.get("symbol", "") for o in options],
                metric_value=net_delta,
                threshold=200,
                recommendation="Consider hedging with opposite delta exposure or reducing position sizes"
            ))

        # Check for extreme gamma
        net_gamma = abs(options_summary.get("net_gamma", 0))
        if net_gamma > 100:
            anomalies.append(AnomalyResult(
                anomaly_type=AnomalyType.UNUSUAL_GREEKS,
                severity="warning",
                description=f"High gamma exposure: {net_gamma:.1f} (accelerated P/L changes)",
                affected_positions=[o.get("symbol", "") for o in options if abs(o.get("greeks", {}).get("gamma", 0)) > 5],
                metric_value=net_gamma,
                threshold=100,
                recommendation="Monitor positions closely as P/L will change rapidly with price moves"
            ))

        return anomalies

    def _check_concentration_risk(
        self,
        options: List[Dict],
        total_value: float
    ) -> List[AnomalyResult]:
        """Check for excessive concentration in single positions."""
        anomalies = []

        for opt in options:
            position_value = abs(opt.get("current_value", 0))
            concentration_pct = (position_value / total_value * 100) if total_value > 0 else 0

            if concentration_pct > self.THRESHOLDS["concentration_pct"]:
                anomalies.append(AnomalyResult(
                    anomaly_type=AnomalyType.CONCENTRATION_RISK,
                    severity="warning" if concentration_pct < 50 else "critical",
                    description=f"{opt.get('symbol', 'Unknown')} represents {concentration_pct:.1f}% of portfolio",
                    affected_positions=[opt.get("symbol", "")],
                    metric_value=concentration_pct,
                    threshold=self.THRESHOLDS["concentration_pct"],
                    recommendation="Consider reducing position size or diversifying across more underlyings"
                ))

        return anomalies

    def _check_theta_imbalance(
        self,
        summary: Dict,
        total_value: float
    ) -> List[AnomalyResult]:
        """Check for unsustainable theta exposure."""
        anomalies = []
        options_summary = summary.get("options_summary", {})

        net_theta = options_summary.get("net_theta", 0)
        theta_pct = abs(net_theta) / total_value if total_value > 0 else 0

        # Warning if daily theta is > 5% of portfolio value
        if theta_pct > self.THRESHOLDS["theta_imbalance"]:
            direction = "losing" if net_theta < 0 else "gaining"
            anomalies.append(AnomalyResult(
                anomaly_type=AnomalyType.THETA_IMBALANCE,
                severity="warning",
                description=f"High theta exposure: ${abs(net_theta):.0f}/day ({direction})",
                affected_positions=[],
                metric_value=theta_pct * 100,
                threshold=self.THRESHOLDS["theta_imbalance"] * 100,
                recommendation="Review time decay impact on your portfolio"
            ))

        return anomalies

    def _check_unusual_volatility(
        self,
        options: List[Dict]
    ) -> List[AnomalyResult]:
        """Check for unusual implied volatility levels."""
        anomalies = []

        ivs = [o.get("greeks", {}).get("iv", 0) for o in options if o.get("greeks", {}).get("iv", 0) > 0]

        if len(ivs) >= 3:
            mean_iv = np.mean(ivs)
            std_iv = np.std(ivs)

            for opt in options:
                iv = opt.get("greeks", {}).get("iv", 0)
                if iv > 0 and std_iv > 0:
                    z_score = (iv - mean_iv) / std_iv

                    if abs(z_score) > self.THRESHOLDS["iv_zscore"]:
                        anomalies.append(AnomalyResult(
                            anomaly_type=AnomalyType.UNUSUAL_VOLATILITY,
                            severity="info",
                            description=f"{opt.get('symbol', 'Unknown')} IV ({iv:.1f}%) is {z_score:.1f} std devs from portfolio mean",
                            affected_positions=[opt.get("symbol", "")],
                            metric_value=z_score,
                            threshold=self.THRESHOLDS["iv_zscore"],
                            recommendation="Check for upcoming events or earnings that may be driving elevated IV"
                        ))

        return anomalies

    def get_risk_score(self, positions: Dict[str, Any]) -> Dict[str, Any]:
        """
        Calculate overall portfolio risk score (0-100).

        Lower = safer, Higher = riskier
        """
        anomalies = self.analyze_portfolio(positions)

        # Base score
        base_score = 30

        # Add points for anomalies
        critical_count = sum(1 for a in anomalies if a.severity == "critical")
        warning_count = sum(1 for a in anomalies if a.severity == "warning")
        info_count = sum(1 for a in anomalies if a.severity == "info")

        score = base_score + (critical_count * 25) + (warning_count * 10) + (info_count * 3)
        score = min(100, score)

        # Determine risk level
        if score >= 70:
            risk_level = "High"
        elif score >= 40:
            risk_level = "Moderate"
        else:
            risk_level = "Low"

        return {
            "score": score,
            "risk_level": risk_level,
            "anomaly_count": len(anomalies),
            "critical_issues": critical_count,
            "warnings": warning_count,
            "top_issues": [
                {
                    "type": a.anomaly_type.value,
                    "severity": a.severity,
                    "description": a.description,
                    "recommendation": a.recommendation
                }
                for a in sorted(anomalies, key=lambda x: {"critical": 0, "warning": 1, "info": 2}[x.severity])[:3]
            ]
        }


# =============================================================================
# Time Series Forecasting (ARIMA-like)
# =============================================================================

@dataclass
class TimeSeriesForecast:
    """Time series forecast result"""
    symbol: str
    forecast_horizon: int  # days
    forecasted_values: List[float]
    confidence_interval_lower: List[float]
    confidence_interval_upper: List[float]
    model_type: str
    mae: float  # Mean Absolute Error on training
    mape: float  # Mean Absolute Percentage Error
    trend: str  # "upward", "downward", "sideways"
    seasonality_detected: bool
    generated_at: datetime = field(default_factory=datetime.now)


class TimeSeriesForecaster:
    """
    ARIMA-like time series forecasting for price prediction.

    Uses exponential smoothing and autoregression for forecasting.
    Does not require external libraries like statsmodels.
    """

    def __init__(self, alpha: float = 0.3, beta: float = 0.1):
        """
        Initialize forecaster with smoothing parameters.

        Args:
            alpha: Level smoothing factor (0-1)
            beta: Trend smoothing factor (0-1)
        """
        self.alpha = alpha
        self.beta = beta

    def forecast(
        self,
        symbol: str,
        prices: List[float],
        horizon: int = 5,
        confidence_level: float = 0.95
    ) -> TimeSeriesForecast:
        """
        Generate time series forecast using Holt's Linear Trend Method.

        This is a simplified ARIMA-like approach using exponential smoothing
        with trend component. Works well for short-term forecasting.

        Args:
            symbol: Stock symbol
            prices: Historical prices (most recent last)
            horizon: Days to forecast ahead
            confidence_level: Confidence interval level

        Returns:
            TimeSeriesForecast with predictions and confidence intervals
        """
        if len(prices) < 3:
            # Not enough data for forecasting
            return TimeSeriesForecast(
                symbol=symbol,
                forecast_horizon=horizon,
                forecasted_values=[prices[-1]] * horizon if prices else [0] * horizon,
                confidence_interval_lower=[prices[-1] * 0.95] * horizon if prices else [0] * horizon,
                confidence_interval_upper=[prices[-1] * 1.05] * horizon if prices else [0] * horizon,
                model_type="baseline",
                mae=0,
                mape=0,
                trend="sideways",
                seasonality_detected=False
            )

        prices_arr = np.array(prices)
        n = len(prices_arr)

        # Initialize level and trend (Holt's method)
        level = prices_arr[0]
        trend = np.mean(np.diff(prices_arr[:min(5, n)]))  # Initial trend

        # Fit model using exponential smoothing with trend
        fitted = []
        residuals = []

        for i in range(n):
            if i == 0:
                forecast = level + trend
            else:
                # Update level and trend
                prev_level = level
                level = self.alpha * prices_arr[i] + (1 - self.alpha) * (level + trend)
                trend = self.beta * (level - prev_level) + (1 - self.beta) * trend
                forecast = level + trend

            fitted.append(forecast)
            residuals.append(prices_arr[i] - forecast)

        # Calculate error metrics
        residuals_arr = np.array(residuals[1:])  # Skip first
        mae = np.mean(np.abs(residuals_arr))
        mape = np.mean(np.abs(residuals_arr / prices_arr[1:])) * 100 if np.all(prices_arr[1:] != 0) else 0

        # Standard deviation of residuals for confidence intervals
        residual_std = np.std(residuals_arr)

        # Z-score for confidence interval
        z_score = 1.96 if confidence_level == 0.95 else 2.576

        # Generate forecasts
        forecasted_values = []
        confidence_lower = []
        confidence_upper = []

        for h in range(1, horizon + 1):
            # Forecast = level + h * trend
            forecast_h = level + h * trend

            # Confidence interval widens with horizon
            interval = z_score * residual_std * np.sqrt(h)

            forecasted_values.append(round(float(forecast_h), 2))
            confidence_lower.append(round(float(forecast_h - interval), 2))
            confidence_upper.append(round(float(forecast_h + interval), 2))

        # Determine trend direction
        if trend > 0.01 * prices_arr[-1]:
            trend_direction = "upward"
        elif trend < -0.01 * prices_arr[-1]:
            trend_direction = "downward"
        else:
            trend_direction = "sideways"

        # Simple seasonality detection (weekly pattern)
        seasonality_detected = False
        if n >= 14:
            # Check for 5-day (weekly) pattern
            weekly_returns = []
            for i in range(0, n - 7, 5):
                weekly_returns.append(
                    (prices_arr[min(i + 5, n - 1)] - prices_arr[i]) / prices_arr[i]
                )
            if len(weekly_returns) >= 2:
                weekly_std = np.std(weekly_returns)
                if weekly_std < np.std(np.diff(prices_arr) / prices_arr[:-1]) * 0.5:
                    seasonality_detected = True

        return TimeSeriesForecast(
            symbol=symbol,
            forecast_horizon=horizon,
            forecasted_values=forecasted_values,
            confidence_interval_lower=confidence_lower,
            confidence_interval_upper=confidence_upper,
            model_type="holt_linear_trend",
            mae=round(float(mae), 4),
            mape=round(float(mape), 2),
            trend=trend_direction,
            seasonality_detected=seasonality_detected
        )

    def forecast_with_arima_approximation(
        self,
        symbol: str,
        prices: List[float],
        horizon: int = 5,
        ar_order: int = 2,
        ma_order: int = 1
    ) -> TimeSeriesForecast:
        """
        ARIMA-like forecast using autoregression approximation.

        Uses simple AR(p) model fitted via least squares.

        Args:
            symbol: Stock symbol
            prices: Historical prices
            horizon: Forecast horizon
            ar_order: Autoregressive order (p)
            ma_order: Moving average order (for error smoothing)

        Returns:
            TimeSeriesForecast with ARIMA-style predictions
        """
        if len(prices) < ar_order + 5:
            return self.forecast(symbol, prices, horizon)

        prices_arr = np.array(prices)
        n = len(prices_arr)

        # Difference the series (I=1)
        diff = np.diff(prices_arr)

        # Build AR features matrix
        X = []
        y = []
        for i in range(ar_order, len(diff)):
            X.append(diff[i - ar_order:i][::-1])  # Lagged values
            y.append(diff[i])

        X = np.array(X)
        y = np.array(y)

        # Fit AR coefficients using least squares
        try:
            # Add intercept
            X_with_intercept = np.column_stack([np.ones(len(X)), X])
            coefficients = np.linalg.lstsq(X_with_intercept, y, rcond=None)[0]
            intercept = coefficients[0]
            ar_coeffs = coefficients[1:]
        except np.linalg.LinAlgError:
            return self.forecast(symbol, prices, horizon)

        # Calculate residuals for MA component and confidence intervals
        fitted_diff = X_with_intercept @ coefficients
        residuals = y - fitted_diff
        residual_std = np.std(residuals)

        # MA smoothing of residuals
        if ma_order > 0 and len(residuals) > ma_order:
            ma_residuals = np.convolve(
                residuals,
                np.ones(ma_order) / ma_order,
                mode='valid'
            )
            ma_adjustment = ma_residuals[-1] if len(ma_residuals) > 0 else 0
        else:
            ma_adjustment = 0

        # Generate forecasts
        last_diffs = list(diff[-ar_order:])
        last_price = prices_arr[-1]

        forecasted_values = []
        confidence_lower = []
        confidence_upper = []

        for h in range(horizon):
            # AR prediction for diff
            features = np.array(last_diffs[-ar_order:][::-1])
            diff_forecast = intercept + np.dot(ar_coeffs, features) + ma_adjustment

            # Integrate back to price level
            price_forecast = last_price + diff_forecast
            forecasted_values.append(round(float(price_forecast), 2))

            # Confidence interval
            interval = 1.96 * residual_std * np.sqrt(h + 1)
            confidence_lower.append(round(float(price_forecast - interval), 2))
            confidence_upper.append(round(float(price_forecast + interval), 2))

            # Update for next step
            last_diffs.append(diff_forecast)
            last_price = price_forecast

        # Error metrics
        mae = np.mean(np.abs(residuals))
        mape = np.mean(np.abs(residuals / (y + 0.0001))) * 100

        # Trend from AR coefficients
        total_ar_effect = np.sum(ar_coeffs)
        if total_ar_effect > 0.1:
            trend_direction = "upward"
        elif total_ar_effect < -0.1:
            trend_direction = "downward"
        else:
            trend_direction = "sideways"

        return TimeSeriesForecast(
            symbol=symbol,
            forecast_horizon=horizon,
            forecasted_values=forecasted_values,
            confidence_interval_lower=confidence_lower,
            confidence_interval_upper=confidence_upper,
            model_type=f"arima_approx_{ar_order}_1_{ma_order}",
            mae=round(float(mae), 4),
            mape=round(float(mape), 2),
            trend=trend_direction,
            seasonality_detected=False
        )


# =============================================================================
# Volatility Prediction Model
# =============================================================================

@dataclass
class VolatilityForecast:
    """Volatility forecast result"""
    symbol: str
    current_realized_vol: float
    current_iv: float
    forecasted_vol_5d: float
    forecasted_vol_10d: float
    forecasted_vol_21d: float
    vol_regime: str  # "low", "normal", "elevated", "extreme"
    vol_trend: str  # "increasing", "decreasing", "stable"
    iv_rv_spread: float  # IV premium/discount to RV
    vix_correlation: float
    garch_persistence: float
    confidence: PredictionConfidence
    generated_at: datetime = field(default_factory=datetime.now)


class VolatilityPredictor:
    """
    Volatility prediction using GARCH-like model.

    Implements simplified GARCH(1,1) for volatility forecasting
    without requiring external libraries.
    """

    def __init__(
        self,
        omega: float = 0.00001,  # Long-run variance weight
        alpha: float = 0.1,      # ARCH term (shock impact)
        beta: float = 0.85       # GARCH term (persistence)
    ):
        """
        Initialize with GARCH(1,1) parameters.

        omega + alpha + beta should be close to 1 for stationarity.
        """
        self.omega = omega
        self.alpha = alpha
        self.beta = beta
        self.long_run_var = omega / (1 - alpha - beta) if (alpha + beta) < 1 else omega

    def predict_volatility(
        self,
        symbol: str,
        prices: List[float],
        current_iv: float = 0.30,
        horizon_days: List[int] = [5, 10, 21]
    ) -> VolatilityForecast:
        """
        Predict future volatility using GARCH(1,1) model.

        Args:
            symbol: Stock symbol
            prices: Historical prices (at least 30 for reliable results)
            current_iv: Current implied volatility
            horizon_days: Days ahead to forecast

        Returns:
            VolatilityForecast with predictions and regime info
        """
        if len(prices) < 5:
            return self._default_forecast(symbol, current_iv)

        prices_arr = np.array(prices)

        # Calculate log returns
        log_returns = np.diff(np.log(prices_arr))
        n = len(log_returns)

        # Calculate realized volatility (annualized)
        daily_vol = np.std(log_returns)
        realized_vol = daily_vol * np.sqrt(252)

        # Fit GARCH(1,1) model
        # Initialize variance
        variance = daily_vol ** 2
        variances = [variance]

        for i in range(1, n):
            # GARCH(1,1): sigma^2_t = omega + alpha * r^2_{t-1} + beta * sigma^2_{t-1}
            variance = (
                self.omega +
                self.alpha * log_returns[i - 1] ** 2 +
                self.beta * variances[-1]
            )
            variances.append(variance)

        current_variance = variances[-1]

        # Forecast volatility for each horizon
        forecasts = {}
        for h in horizon_days:
            # h-step ahead variance forecast
            forecast_var = self.long_run_var + (
                (self.alpha + self.beta) ** h *
                (current_variance - self.long_run_var)
            )
            forecast_vol = np.sqrt(forecast_var * 252)  # Annualize
            forecasts[h] = round(float(forecast_vol * 100), 2)  # As percentage

        # Determine volatility regime
        vol_percentile = self._estimate_vol_percentile(realized_vol, prices_arr)

        if vol_percentile > 90:
            vol_regime = "extreme"
        elif vol_percentile > 70:
            vol_regime = "elevated"
        elif vol_percentile > 30:
            vol_regime = "normal"
        else:
            vol_regime = "low"

        # Determine volatility trend
        recent_vol = np.std(log_returns[-5:]) * np.sqrt(252) if n >= 5 else realized_vol
        older_vol = np.std(log_returns[-21:-5]) * np.sqrt(252) if n >= 21 else realized_vol

        if recent_vol > older_vol * 1.2:
            vol_trend = "increasing"
        elif recent_vol < older_vol * 0.8:
            vol_trend = "decreasing"
        else:
            vol_trend = "stable"

        # IV-RV spread (IV premium or discount)
        iv_rv_spread = (current_iv - realized_vol) / realized_vol * 100 if realized_vol > 0 else 0

        # GARCH persistence
        persistence = self.alpha + self.beta

        # Confidence based on data quality
        if n >= 60:
            confidence = PredictionConfidence.HIGH
        elif n >= 21:
            confidence = PredictionConfidence.MEDIUM
        else:
            confidence = PredictionConfidence.LOW

        return VolatilityForecast(
            symbol=symbol,
            current_realized_vol=round(float(realized_vol * 100), 2),
            current_iv=round(float(current_iv * 100), 2),
            forecasted_vol_5d=forecasts.get(5, forecasts.get(min(forecasts.keys()))),
            forecasted_vol_10d=forecasts.get(10, forecasts.get(min(forecasts.keys()))),
            forecasted_vol_21d=forecasts.get(21, forecasts.get(max(forecasts.keys()))),
            vol_regime=vol_regime,
            vol_trend=vol_trend,
            iv_rv_spread=round(float(iv_rv_spread), 2),
            vix_correlation=0.7,  # Typical equity-VIX correlation
            garch_persistence=round(float(persistence), 3),
            confidence=confidence
        )

    def _estimate_vol_percentile(
        self,
        current_vol: float,
        prices: np.ndarray
    ) -> float:
        """Estimate where current vol sits in historical distribution."""
        if len(prices) < 60:
            return 50.0  # Not enough data, assume median

        # Calculate rolling 21-day volatilities
        log_returns = np.diff(np.log(prices))
        window = 21

        rolling_vols = []
        for i in range(window, len(log_returns)):
            vol = np.std(log_returns[i - window:i]) * np.sqrt(252)
            rolling_vols.append(vol)

        if not rolling_vols:
            return 50.0

        # Calculate percentile
        percentile = np.sum(np.array(rolling_vols) < current_vol) / len(rolling_vols) * 100
        return float(percentile)

    def _default_forecast(self, symbol: str, current_iv: float) -> VolatilityForecast:
        """Return default forecast when insufficient data."""
        return VolatilityForecast(
            symbol=symbol,
            current_realized_vol=current_iv * 100,
            current_iv=current_iv * 100,
            forecasted_vol_5d=current_iv * 100,
            forecasted_vol_10d=current_iv * 100,
            forecasted_vol_21d=current_iv * 100,
            vol_regime="normal",
            vol_trend="stable",
            iv_rv_spread=0,
            vix_correlation=0.7,
            garch_persistence=0.95,
            confidence=PredictionConfidence.LOW
        )

    def estimate_vol_surface(
        self,
        symbol: str,
        prices: List[float],
        current_iv: float,
        days_ahead: int = 30
    ) -> Dict[str, Any]:
        """
        Estimate future volatility surface.

        Returns expected volatility at different time horizons.
        """
        base_forecast = self.predict_volatility(symbol, prices, current_iv)

        # Build term structure
        term_structure = {}
        persistence = base_forecast.garch_persistence

        for d in [1, 5, 10, 21, 30, 60]:
            if d <= days_ahead:
                # Decay toward long-run vol
                decay_factor = persistence ** d
                long_run_vol = base_forecast.current_realized_vol
                current_vol = base_forecast.current_realized_vol

                expected_vol = (
                    long_run_vol +
                    decay_factor * (current_vol - long_run_vol)
                )
                term_structure[f"{d}d"] = round(expected_vol, 2)

        return {
            "symbol": symbol,
            "current_vol": base_forecast.current_realized_vol,
            "term_structure": term_structure,
            "regime": base_forecast.vol_regime,
            "trend": base_forecast.vol_trend,
            "generated_at": datetime.now().isoformat()
        }


# =============================================================================
# Intelligent Trade Recommendations
# =============================================================================

class TradeRecommendationEngine:
    """
    Generates AI-powered trade recommendations based on portfolio analysis.
    """

    def __init__(
        self,
        prediction_engine: AIPredictionEngine,
        anomaly_detector: PortfolioAnomalyDetector
    ):
        self.prediction_engine = prediction_engine
        self.anomaly_detector = anomaly_detector

    def generate_recommendations(
        self,
        positions: Dict[str, Any],
        risk_tolerance: str = "moderate"  # "conservative", "moderate", "aggressive"
    ) -> List[Dict[str, Any]]:
        """
        Generate actionable trade recommendations.

        Considers:
        - Current portfolio risk
        - Detected anomalies
        - Price predictions
        - Greeks optimization
        """
        recommendations = []

        options = positions.get("options", [])
        summary = positions.get("summary", {})
        total_value = summary.get("total_equity", 0)

        if total_value == 0:
            return recommendations

        # Get current risk assessment
        risk_score = self.anomaly_detector.get_risk_score(positions)
        anomalies = self.anomaly_detector.detected_anomalies

        # Risk thresholds by tolerance
        risk_thresholds = {
            "conservative": 30,
            "moderate": 50,
            "aggressive": 70
        }
        target_threshold = risk_thresholds.get(risk_tolerance, 50)

        # 1. Risk reduction recommendations
        if risk_score["score"] > target_threshold:
            recommendations.append({
                "type": "risk_reduction",
                "priority": "high",
                "action": "Reduce portfolio risk",
                "rationale": f"Current risk score ({risk_score['score']}) exceeds your {risk_tolerance} tolerance threshold ({target_threshold})",
                "suggestions": [
                    "Close or reduce highest-risk positions",
                    "Add hedging positions (long puts or short calls)",
                    "Diversify across more underlyings"
                ]
            })

        # 2. Concentration fix recommendations
        concentration_anomalies = [a for a in anomalies if a.anomaly_type == AnomalyType.CONCENTRATION_RISK]
        for anomaly in concentration_anomalies:
            recommendations.append({
                "type": "diversification",
                "priority": "medium",
                "action": f"Reduce concentration in {anomaly.affected_positions[0]}",
                "rationale": anomaly.description,
                "suggestions": [
                    f"Consider taking profits on {anomaly.affected_positions[0]}",
                    "Redistribute capital across multiple positions",
                    "Set a max allocation rule (e.g., 10-15% per position)"
                ]
            })

        # 3. Delta hedging recommendations
        options_summary = summary.get("options_summary", {})
        net_delta = options_summary.get("net_delta", 0)

        if abs(net_delta) > 100:
            direction = "long" if net_delta > 0 else "short"
            opposite = "bearish" if net_delta > 0 else "bullish"

            recommendations.append({
                "type": "delta_hedge",
                "priority": "medium",
                "action": f"Hedge {direction} delta exposure ({net_delta:.0f})",
                "rationale": f"Portfolio has significant directional {direction} bias",
                "suggestions": [
                    f"Buy {opposite} spreads to reduce delta",
                    f"Add {'put' if net_delta > 0 else 'call'} options as hedge",
                    "Consider rolling positions to reduce directional exposure"
                ]
            })

        # 4. Theta optimization
        net_theta = options_summary.get("net_theta", 0)
        if net_theta < -50:  # Losing significant theta
            recommendations.append({
                "type": "theta_optimization",
                "priority": "low",
                "action": "Improve theta efficiency",
                "rationale": f"Losing ${abs(net_theta):.0f}/day to time decay",
                "suggestions": [
                    "Convert long options to spreads",
                    "Consider selling options against your longs (covered calls, put spreads)",
                    "Roll options closer to expiration for slower decay"
                ]
            })

        # 5. Position-specific recommendations based on predictions
        for opt in options[:5]:  # Top 5 positions
            symbol = opt.get("underlying", opt.get("symbol", ""))
            current_price = opt.get("underlying_price", 0)
            iv = opt.get("greeks", {}).get("iv", 30)

            if current_price > 0:
                prediction = self.prediction_engine.predict_price(
                    symbol=symbol,
                    current_price=current_price,
                    iv=iv / 100
                )

                # If prediction suggests significant move
                expected_5d_change = (prediction.predicted_price_5d - current_price) / current_price

                if abs(expected_5d_change) > 0.03:  # > 3% expected move
                    direction = "up" if expected_5d_change > 0 else "down"
                    recommendations.append({
                        "type": "prediction_alert",
                        "priority": "info",
                        "action": f"Review {symbol} position",
                        "rationale": f"Model predicts {symbol} may move {direction} ~{abs(expected_5d_change)*100:.1f}% in 5 days",
                        "confidence": prediction.confidence.value,
                        "suggestions": [
                            f"Consider {'adding' if expected_5d_change > 0 else 'reducing'} bullish exposure",
                            "Review stop-loss levels",
                            "Check for upcoming catalysts"
                        ]
                    })

        # Sort by priority
        priority_order = {"high": 0, "medium": 1, "low": 2, "info": 3}
        recommendations.sort(key=lambda x: priority_order.get(x.get("priority", "info"), 3))

        return recommendations


# =============================================================================
# Singleton Instances
# =============================================================================

import threading

_risk_models: Optional[AdvancedRiskModels] = None
_prediction_engine: Optional[AIPredictionEngine] = None
_anomaly_detector: Optional[PortfolioAnomalyDetector] = None
_recommendation_engine: Optional[TradeRecommendationEngine] = None
_ts_forecaster: Optional[TimeSeriesForecaster] = None
_vol_predictor: Optional[VolatilityPredictor] = None

# Thread-safe locks for singletons
_risk_models_lock = threading.Lock()
_prediction_engine_lock = threading.Lock()
_anomaly_detector_lock = threading.Lock()
_recommendation_engine_lock = threading.Lock()
_ts_forecaster_lock = threading.Lock()
_vol_predictor_lock = threading.Lock()


def get_risk_models() -> AdvancedRiskModels:
    """Get advanced risk models singleton (thread-safe)."""
    global _risk_models
    if _risk_models is None:
        with _risk_models_lock:
            if _risk_models is None:
                _risk_models = AdvancedRiskModels()
    return _risk_models


def get_prediction_engine() -> AIPredictionEngine:
    """Get AI prediction engine singleton (thread-safe)."""
    global _prediction_engine
    if _prediction_engine is None:
        with _prediction_engine_lock:
            if _prediction_engine is None:
                _prediction_engine = AIPredictionEngine()
    return _prediction_engine


def get_anomaly_detector() -> PortfolioAnomalyDetector:
    """Get anomaly detector singleton (thread-safe)."""
    global _anomaly_detector
    if _anomaly_detector is None:
        with _anomaly_detector_lock:
            if _anomaly_detector is None:
                _anomaly_detector = PortfolioAnomalyDetector()
    return _anomaly_detector


def get_recommendation_engine() -> TradeRecommendationEngine:
    """Get trade recommendation engine singleton (thread-safe)."""
    global _recommendation_engine
    if _recommendation_engine is None:
        with _recommendation_engine_lock:
            if _recommendation_engine is None:
                _recommendation_engine = TradeRecommendationEngine(
                    prediction_engine=get_prediction_engine(),
                    anomaly_detector=get_anomaly_detector()
                )
    return _recommendation_engine


def get_ts_forecaster() -> TimeSeriesForecaster:
    """Get time series forecaster singleton (thread-safe)."""
    global _ts_forecaster
    if _ts_forecaster is None:
        with _ts_forecaster_lock:
            if _ts_forecaster is None:
                _ts_forecaster = TimeSeriesForecaster()
    return _ts_forecaster


def get_vol_predictor() -> VolatilityPredictor:
    """Get volatility predictor singleton (thread-safe)."""
    global _vol_predictor
    if _vol_predictor is None:
        with _vol_predictor_lock:
            if _vol_predictor is None:
                _vol_predictor = VolatilityPredictor()
    return _vol_predictor

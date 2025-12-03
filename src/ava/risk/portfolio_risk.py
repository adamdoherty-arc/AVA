"""
Portfolio Risk Engine
=====================

Comprehensive portfolio-level risk management for options trading:
- Portfolio Greeks aggregation
- Value at Risk (VaR) calculations
- Stress testing scenarios
- Position sizing (Kelly Criterion, Fixed Fractional)
- Risk limits enforcement
- Correlation analysis

Author: AVA Trading Platform
Created: 2025-11-28
"""

from dataclasses import dataclass, field
from datetime import date, datetime
from typing import List, Dict, Optional, Tuple
from enum import Enum
import logging
import numpy as np
import pandas as pd
from scipy import stats

logger = logging.getLogger(__name__)


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class RiskLimits:
    """Risk limits configuration"""
    # Portfolio limits
    max_portfolio_delta: float = 500      # Max net delta
    max_portfolio_gamma: float = 100      # Max net gamma
    max_portfolio_vega: float = 1000      # Max net vega
    max_portfolio_theta: float = -500     # Max (most negative) theta

    # Position limits
    max_position_size_pct: float = 0.05   # 5% max per position
    max_sector_exposure_pct: float = 0.25  # 25% max per sector
    max_single_underlying_pct: float = 0.15  # 15% max per underlying

    # Loss limits
    max_daily_loss_pct: float = 0.02      # 2% daily loss limit
    max_weekly_loss_pct: float = 0.05     # 5% weekly loss limit
    max_drawdown_pct: float = 0.15        # 15% max drawdown

    # VaR limits
    max_var_95_pct: float = 0.03          # 3% 95% VaR limit
    max_var_99_pct: float = 0.05          # 5% 99% VaR limit

    # Concentration limits
    max_correlated_positions: int = 3      # Max positions with high correlation


@dataclass
class PortfolioGreeks:
    """Aggregated portfolio Greeks"""
    net_delta: float = 0.0
    net_gamma: float = 0.0
    net_theta: float = 0.0
    net_vega: float = 0.0

    # Higher order Greeks
    net_vanna: float = 0.0
    net_charm: float = 0.0

    # Dollar amounts
    delta_dollars: float = 0.0     # Dollar exposure per $1 move
    gamma_dollars: float = 0.0     # Dollar change in delta per $1 move
    theta_dollars: float = 0.0     # Daily theta decay in dollars
    vega_dollars: float = 0.0      # Dollar exposure per 1% IV change

    # By underlying
    delta_by_underlying: Dict[str, float] = field(default_factory=dict)
    theta_by_underlying: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict:
        return {
            'net_delta': round(self.net_delta, 2),
            'net_gamma': round(self.net_gamma, 4),
            'net_theta': round(self.net_theta, 2),
            'net_vega': round(self.net_vega, 2),
            'delta_dollars': round(self.delta_dollars, 2),
            'theta_dollars': round(self.theta_dollars, 2)
        }


@dataclass
class StressTestResult:
    """Result of a stress test scenario"""
    scenario_name: str
    description: str

    # Scenario parameters
    price_change_pct: float
    iv_change_pct: float
    time_days: int = 0

    # Results
    pnl_impact: float = 0.0
    pnl_impact_pct: float = 0.0

    # Greeks after scenario
    resulting_delta: float = 0.0
    resulting_gamma: float = 0.0

    # Risk metrics
    margin_impact: float = 0.0
    positions_at_risk: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict:
        return {
            'scenario': self.scenario_name,
            'description': self.description,
            'price_change': f"{self.price_change_pct:+.1%}",
            'iv_change': f"{self.iv_change_pct:+.1%}",
            'pnl_impact': f"${self.pnl_impact:,.2f}",
            'pnl_impact_pct': f"{self.pnl_impact_pct:+.2%}"
        }


@dataclass
class VaRResult:
    """Value at Risk calculation result"""
    var_95: float = 0.0           # 95% VaR (5% worst case)
    var_99: float = 0.0           # 99% VaR (1% worst case)
    expected_shortfall_95: float = 0.0  # CVaR / Expected Shortfall
    expected_shortfall_99: float = 0.0

    # As percentages
    var_95_pct: float = 0.0
    var_99_pct: float = 0.0

    # Method used
    method: str = "historical"    # historical, parametric, monte_carlo

    def to_dict(self) -> Dict:
        return {
            '95% VaR': f"${self.var_95:,.2f} ({self.var_95_pct:.2%})",
            '99% VaR': f"${self.var_99:,.2f} ({self.var_99_pct:.2%})",
            '95% CVaR': f"${self.expected_shortfall_95:,.2f}",
            'method': self.method
        }


@dataclass
class RiskViolation:
    """A risk limit violation"""
    limit_name: str
    current_value: float
    limit_value: float
    severity: str  # warning, critical
    message: str

    def to_dict(self) -> Dict:
        return {
            'limit': self.limit_name,
            'current': self.current_value,
            'limit': self.limit_value,
            'severity': self.severity,
            'message': self.message
        }


@dataclass
class RiskAnalysis:
    """Complete risk analysis result"""
    timestamp: datetime
    portfolio_value: float

    # Greeks
    greeks: PortfolioGreeks

    # VaR
    var: VaRResult

    # Stress tests
    stress_tests: List[StressTestResult] = field(default_factory=list)

    # Violations
    violations: List[RiskViolation] = field(default_factory=list)

    # Position analysis
    largest_positions: List[Dict] = field(default_factory=list)
    concentration_by_underlying: Dict[str, float] = field(default_factory=dict)

    # Recommendations
    recommendations: List[str] = field(default_factory=list)

    @property
    def is_within_limits(self) -> bool:
        return len([v for v in self.violations if v.severity == 'critical']) == 0

    def summary(self) -> str:
        return f"""
=== Portfolio Risk Analysis ===
Time: {self.timestamp.strftime('%Y-%m-%d %H:%M')}
Portfolio Value: ${self.portfolio_value:,.2f}

GREEKS
------
Net Delta: {self.greeks.net_delta:+.2f} (${self.greeks.delta_dollars:,.2f} per $1)
Net Gamma: {self.greeks.net_gamma:+.4f}
Net Theta: {self.greeks.net_theta:+.2f} (${self.greeks.theta_dollars:,.2f}/day)
Net Vega:  {self.greeks.net_vega:+.2f}

VALUE AT RISK
-------------
95% VaR: ${self.var.var_95:,.2f} ({self.var.var_95_pct:.2%})
99% VaR: ${self.var.var_99:,.2f} ({self.var.var_99_pct:.2%})

VIOLATIONS: {len(self.violations)}
{chr(10).join(['- ' + v.message for v in self.violations]) if self.violations else 'None'}

RECOMMENDATIONS
---------------
{chr(10).join(['- ' + r for r in self.recommendations]) if self.recommendations else 'Portfolio within acceptable risk parameters'}
"""


# =============================================================================
# PORTFOLIO RISK ENGINE
# =============================================================================

class PortfolioRiskEngine:
    """
    Comprehensive portfolio risk management engine.

    Usage:
        engine = PortfolioRiskEngine(limits=RiskLimits())
        analysis = engine.analyze_portfolio(positions, portfolio_value)
    """

    def __init__(self, limits: Optional[RiskLimits] = None):
        self.limits = limits or RiskLimits()

    def analyze_portfolio(
        self,
        positions: List[Dict],
        portfolio_value: float,
        market_data: Optional[Dict[str, float]] = None
    ) -> RiskAnalysis:
        """
        Perform comprehensive portfolio risk analysis.

        Args:
            positions: List of position dicts with keys:
                       symbol, delta, gamma, theta, vega, market_value, quantity
            portfolio_value: Total portfolio value
            market_data: Optional dict of symbol -> current_price

        Returns:
            RiskAnalysis with complete risk metrics
        """
        logger.info(f"Analyzing portfolio: {len(positions)} positions, ${portfolio_value:,.2f} value")

        # Calculate portfolio Greeks
        greeks = self._calculate_portfolio_greeks(positions)

        # Calculate VaR
        var = self._calculate_var(positions, portfolio_value)

        # Run stress tests
        stress_tests = self._run_stress_tests(positions, portfolio_value, market_data)

        # Check for violations
        violations = self._check_violations(greeks, var, portfolio_value)

        # Analyze concentration
        concentration = self._analyze_concentration(positions, portfolio_value)

        # Get largest positions
        largest = self._get_largest_positions(positions, portfolio_value)

        # Generate recommendations
        recommendations = self._generate_recommendations(greeks, var, violations, concentration)

        return RiskAnalysis(
            timestamp=datetime.now(),
            portfolio_value=portfolio_value,
            greeks=greeks,
            var=var,
            stress_tests=stress_tests,
            violations=violations,
            largest_positions=largest,
            concentration_by_underlying=concentration,
            recommendations=recommendations
        )

    def _calculate_portfolio_greeks(self, positions: List[Dict]) -> PortfolioGreeks:
        """Aggregate Greeks across all positions"""
        greeks = PortfolioGreeks()

        delta_by_underlying = {}
        theta_by_underlying = {}

        for pos in positions:
            qty = pos.get('quantity', 1)
            symbol = pos.get('symbol', 'UNKNOWN')

            # Aggregate Greeks
            greeks.net_delta += pos.get('delta', 0) * qty
            greeks.net_gamma += pos.get('gamma', 0) * qty
            greeks.net_theta += pos.get('theta', 0) * qty
            greeks.net_vega += pos.get('vega', 0) * qty

            # Higher order
            greeks.net_vanna += pos.get('vanna', 0) * qty
            greeks.net_charm += pos.get('charm', 0) * qty

            # By underlying
            underlying = pos.get('underlying', symbol)
            delta_by_underlying[underlying] = delta_by_underlying.get(underlying, 0) + pos.get('delta', 0) * qty
            theta_by_underlying[underlying] = theta_by_underlying.get(underlying, 0) + pos.get('theta', 0) * qty

        # Calculate dollar amounts (assuming $100 contract multiplier)
        greeks.delta_dollars = greeks.net_delta * 100
        greeks.gamma_dollars = greeks.net_gamma * 100
        greeks.theta_dollars = greeks.net_theta * 100
        greeks.vega_dollars = greeks.net_vega * 100

        greeks.delta_by_underlying = delta_by_underlying
        greeks.theta_by_underlying = theta_by_underlying

        return greeks

    def _calculate_var(
        self,
        positions: List[Dict],
        portfolio_value: float,
        method: str = 'parametric'
    ) -> VaRResult:
        """Calculate Value at Risk"""

        if not positions or portfolio_value <= 0:
            return VaRResult()

        # Parametric VaR (delta-normal)
        # Assumes normal distribution of returns

        # Estimate portfolio volatility from positions
        position_values = [abs(pos.get('market_value', 0)) for pos in positions]
        total_position_value = sum(position_values)

        if total_position_value == 0:
            return VaRResult()

        # Estimate daily volatility (simplified)
        # In production, use historical returns or implied vol
        avg_iv = np.mean([pos.get('iv', 0.30) for pos in positions])
        daily_vol = avg_iv / np.sqrt(252)

        # Portfolio weighted volatility
        portfolio_vol = daily_vol * (total_position_value / portfolio_value)

        # Calculate VaR
        var_95 = portfolio_value * portfolio_vol * stats.norm.ppf(0.95)
        var_99 = portfolio_value * portfolio_vol * stats.norm.ppf(0.99)

        # Expected Shortfall (CVaR)
        es_95 = portfolio_value * portfolio_vol * stats.norm.pdf(stats.norm.ppf(0.95)) / 0.05
        es_99 = portfolio_value * portfolio_vol * stats.norm.pdf(stats.norm.ppf(0.99)) / 0.01

        return VaRResult(
            var_95=var_95,
            var_99=var_99,
            expected_shortfall_95=es_95,
            expected_shortfall_99=es_99,
            var_95_pct=var_95 / portfolio_value,
            var_99_pct=var_99 / portfolio_value,
            method=method
        )

    def _run_stress_tests(
        self,
        positions: List[Dict],
        portfolio_value: float,
        market_data: Optional[Dict[str, float]] = None
    ) -> List[StressTestResult]:
        """Run standard stress test scenarios"""

        scenarios = [
            # Market crash scenarios
            ('Market Crash -10%', 'Sudden 10% market decline', -0.10, 0.50, 1),
            ('Market Crash -20%', 'Severe 20% market crash', -0.20, 1.00, 1),
            ('Flash Crash -5%', 'Intraday flash crash', -0.05, 0.30, 0),

            # Rally scenarios
            ('Rally +10%', 'Strong 10% rally', 0.10, -0.20, 7),
            ('Melt-up +20%', 'Parabolic 20% move up', 0.20, -0.30, 14),

            # Volatility scenarios
            ('Vol Spike +50%', 'IV expansion 50%', 0.0, 0.50, 1),
            ('Vol Crush -30%', 'IV contraction 30%', 0.0, -0.30, 1),

            # Time decay
            ('Theta Bleed 7d', '7 days of theta decay', 0.0, 0.0, 7),
            ('Theta Bleed 14d', '14 days of theta decay', 0.0, 0.0, 14),

            # Combined scenarios
            ('Selloff + Vol', 'Selloff with vol spike', -0.10, 0.40, 3),
            ('Grind Higher', 'Slow rally with vol decline', 0.05, -0.20, 14),
        ]

        results = []

        for name, desc, price_chg, iv_chg, days in scenarios:
            result = self._run_single_stress_test(
                positions, portfolio_value, name, desc,
                price_chg, iv_chg, days
            )
            results.append(result)

        return results

    def _run_single_stress_test(
        self,
        positions: List[Dict],
        portfolio_value: float,
        name: str,
        description: str,
        price_change: float,
        iv_change: float,
        time_days: int
    ) -> StressTestResult:
        """Run a single stress test scenario"""

        total_pnl = 0.0
        resulting_delta = 0.0
        positions_at_risk = []

        for pos in positions:
            qty = pos.get('quantity', 1)
            price = pos.get('underlying_price', 100)

            # Delta P&L
            delta = pos.get('delta', 0)
            delta_pnl = delta * price * price_change * qty * 100

            # Gamma P&L (second order)
            gamma = pos.get('gamma', 0)
            gamma_pnl = 0.5 * gamma * (price * price_change) ** 2 * qty * 100

            # Vega P&L
            vega = pos.get('vega', 0)
            vega_pnl = vega * iv_change * 100 * qty * 100  # IV in percentage points

            # Theta P&L
            theta = pos.get('theta', 0)
            theta_pnl = theta * time_days * qty * 100

            position_pnl = delta_pnl + gamma_pnl + vega_pnl + theta_pnl
            total_pnl += position_pnl

            # Track resulting delta
            resulting_delta += delta * qty + gamma * price * price_change * qty

            # Flag positions with significant losses
            if position_pnl < -pos.get('market_value', 0) * 0.5:
                positions_at_risk.append(pos.get('symbol', 'UNKNOWN'))

        return StressTestResult(
            scenario_name=name,
            description=description,
            price_change_pct=price_change,
            iv_change_pct=iv_change,
            time_days=time_days,
            pnl_impact=total_pnl,
            pnl_impact_pct=total_pnl / portfolio_value if portfolio_value > 0 else 0,
            resulting_delta=resulting_delta,
            positions_at_risk=positions_at_risk
        )

    def _check_violations(
        self,
        greeks: PortfolioGreeks,
        var: VaRResult,
        portfolio_value: float
    ) -> List[RiskViolation]:
        """Check for risk limit violations"""
        violations = []

        # Delta limits
        if abs(greeks.net_delta) > self.limits.max_portfolio_delta:
            violations.append(RiskViolation(
                limit_name='Portfolio Delta',
                current_value=greeks.net_delta,
                limit_value=self.limits.max_portfolio_delta,
                severity='critical' if abs(greeks.net_delta) > self.limits.max_portfolio_delta * 1.5 else 'warning',
                message=f"Portfolio delta {greeks.net_delta:.0f} exceeds limit {self.limits.max_portfolio_delta:.0f}"
            ))

        # Gamma limits
        if abs(greeks.net_gamma) > self.limits.max_portfolio_gamma:
            violations.append(RiskViolation(
                limit_name='Portfolio Gamma',
                current_value=greeks.net_gamma,
                limit_value=self.limits.max_portfolio_gamma,
                severity='warning',
                message=f"Portfolio gamma {greeks.net_gamma:.2f} exceeds limit"
            ))

        # Vega limits
        if abs(greeks.net_vega) > self.limits.max_portfolio_vega:
            violations.append(RiskViolation(
                limit_name='Portfolio Vega',
                current_value=greeks.net_vega,
                limit_value=self.limits.max_portfolio_vega,
                severity='warning',
                message=f"Portfolio vega {greeks.net_vega:.0f} exceeds limit"
            ))

        # VaR limits
        if var.var_95_pct > self.limits.max_var_95_pct:
            violations.append(RiskViolation(
                limit_name='95% VaR',
                current_value=var.var_95_pct,
                limit_value=self.limits.max_var_95_pct,
                severity='critical',
                message=f"95% VaR {var.var_95_pct:.2%} exceeds limit {self.limits.max_var_95_pct:.2%}"
            ))

        if var.var_99_pct > self.limits.max_var_99_pct:
            violations.append(RiskViolation(
                limit_name='99% VaR',
                current_value=var.var_99_pct,
                limit_value=self.limits.max_var_99_pct,
                severity='critical',
                message=f"99% VaR {var.var_99_pct:.2%} exceeds limit"
            ))

        return violations

    def _analyze_concentration(
        self,
        positions: List[Dict],
        portfolio_value: float
    ) -> Dict[str, float]:
        """Analyze portfolio concentration by underlying"""
        concentration = {}

        for pos in positions:
            underlying = pos.get('underlying', pos.get('symbol', 'UNKNOWN'))
            value = abs(pos.get('market_value', 0))
            concentration[underlying] = concentration.get(underlying, 0) + value

        # Convert to percentages
        for underlying in concentration:
            concentration[underlying] /= portfolio_value

        return dict(sorted(concentration.items(), key=lambda x: x[1], reverse=True))

    def _get_largest_positions(
        self,
        positions: List[Dict],
        portfolio_value: float
    ) -> List[Dict]:
        """Get largest positions by market value"""
        sorted_positions = sorted(
            positions,
            key=lambda x: abs(x.get('market_value', 0)),
            reverse=True
        )

        return [
            {
                'symbol': pos.get('symbol'),
                'value': pos.get('market_value', 0),
                'pct': abs(pos.get('market_value', 0)) / portfolio_value,
                'delta': pos.get('delta', 0),
                'theta': pos.get('theta', 0)
            }
            for pos in sorted_positions[:10]
        ]

    def _generate_recommendations(
        self,
        greeks: PortfolioGreeks,
        var: VaRResult,
        violations: List[RiskViolation],
        concentration: Dict[str, float]
    ) -> List[str]:
        """Generate risk management recommendations"""
        recommendations = []

        # Delta recommendations
        if greeks.net_delta > 200:
            recommendations.append(
                f"Consider hedging long delta: Net delta is +{greeks.net_delta:.0f}. "
                "Add bear call spreads or long puts."
            )
        elif greeks.net_delta < -200:
            recommendations.append(
                f"Consider hedging short delta: Net delta is {greeks.net_delta:.0f}. "
                "Add bull put spreads or long calls."
            )

        # Theta recommendations
        if greeks.net_theta < -50:
            recommendations.append(
                f"High theta decay: ${abs(greeks.theta_dollars):.0f}/day. "
                "Consider adding positive theta positions."
            )

        # Vega recommendations
        if greeks.net_vega > 500:
            recommendations.append(
                f"High vega exposure: {greeks.net_vega:.0f}. "
                "Portfolio will lose on IV decline."
            )
        elif greeks.net_vega < -500:
            recommendations.append(
                f"Short vega exposure: {greeks.net_vega:.0f}. "
                "Portfolio will lose on IV spike."
            )

        # Concentration recommendations
        for underlying, pct in concentration.items():
            if pct > 0.20:
                recommendations.append(
                    f"High concentration in {underlying}: {pct:.1%}. "
                    "Consider diversifying."
                )

        # VaR recommendations
        if var.var_95_pct > 0.02:
            recommendations.append(
                f"Elevated VaR: {var.var_95_pct:.2%} daily. "
                "Consider reducing position sizes."
            )

        # Violation-based recommendations
        for violation in violations:
            if violation.severity == 'critical':
                recommendations.append(
                    f"CRITICAL: {violation.message}. Immediate action required."
                )

        return recommendations

    # =========================================================================
    # POSITION SIZING METHODS
    # =========================================================================

    def kelly_criterion(
        self,
        win_rate: float,
        avg_win: float,
        avg_loss: float,
        kelly_fraction: float = 0.5
    ) -> float:
        """
        Calculate optimal position size using Kelly Criterion.

        Args:
            win_rate: Historical win rate (0-1)
            avg_win: Average winning trade
            avg_loss: Average losing trade (positive number)
            kelly_fraction: Fraction of Kelly to use (0.5 = half Kelly)

        Returns:
            Optimal position size as fraction of portfolio
        """
        if avg_loss == 0 or win_rate == 0:
            return 0

        # Kelly formula: f = (bp - q) / b
        # where b = odds (avg_win / avg_loss), p = win_rate, q = 1-p
        b = avg_win / avg_loss
        f = (b * win_rate - (1 - win_rate)) / b

        # Apply Kelly fraction and cap
        f = max(0, f * kelly_fraction)
        f = min(f, self.limits.max_position_size_pct)

        return f

    def fixed_fractional(
        self,
        risk_per_trade: float,
        max_loss: float,
        portfolio_value: float
    ) -> int:
        """
        Calculate position size using fixed fractional method.

        Args:
            risk_per_trade: Max risk per trade as fraction (e.g., 0.02 for 2%)
            max_loss: Max loss per contract
            portfolio_value: Total portfolio value

        Returns:
            Number of contracts
        """
        if max_loss <= 0:
            return 0

        risk_amount = portfolio_value * risk_per_trade
        contracts = int(risk_amount / max_loss)

        return max(1, contracts)

    def optimal_f(
        self,
        trade_results: List[float],
        portfolio_value: float
    ) -> float:
        """
        Calculate optimal f (fraction to risk) from trade history.

        Args:
            trade_results: List of P&L from past trades
            portfolio_value: Current portfolio value

        Returns:
            Optimal fraction to risk per trade
        """
        if not trade_results or len(trade_results) < 10:
            return 0.02  # Default 2%

        # Find largest loss
        max_loss = abs(min(trade_results))
        if max_loss == 0:
            return 0.02

        # Test different f values
        best_f = 0.01
        best_terminal_wealth = 0

        for f in np.arange(0.01, 0.30, 0.01):
            terminal = portfolio_value
            for pnl in trade_results:
                # Simulate growth/loss
                if pnl < 0:
                    terminal *= (1 - f * abs(pnl) / max_loss)
                else:
                    terminal *= (1 + f * pnl / max_loss)

            if terminal > best_terminal_wealth:
                best_terminal_wealth = terminal
                best_f = f

        return min(best_f, self.limits.max_position_size_pct)


# =============================================================================
# TESTING
# =============================================================================

if __name__ == "__main__":
    print("\n=== Testing Portfolio Risk Engine ===\n")

    # Create engine
    limits = RiskLimits(
        max_portfolio_delta=300,
        max_var_95_pct=0.03
    )
    engine = PortfolioRiskEngine(limits)

    # Sample positions
    positions = [
        {
            'symbol': 'SPY',
            'underlying': 'SPY',
            'quantity': 10,
            'delta': -0.30,
            'gamma': 0.02,
            'theta': 0.15,
            'vega': 0.25,
            'market_value': 5000,
            'underlying_price': 560,
            'iv': 0.15
        },
        {
            'symbol': 'QQQ',
            'underlying': 'QQQ',
            'quantity': 5,
            'delta': 0.40,
            'gamma': 0.03,
            'theta': -0.20,
            'vega': 0.30,
            'market_value': 3000,
            'underlying_price': 480,
            'iv': 0.18
        },
        {
            'symbol': 'NVDA',
            'underlying': 'NVDA',
            'quantity': 2,
            'delta': -0.25,
            'gamma': 0.01,
            'theta': 0.10,
            'vega': 0.15,
            'market_value': 2000,
            'underlying_price': 140,
            'iv': 0.45
        }
    ]

    portfolio_value = 100000

    # Run analysis
    analysis = engine.analyze_portfolio(positions, portfolio_value)

    print(analysis.summary())

    # Test stress tests
    print("\nSTRESS TEST RESULTS")
    print("-" * 50)
    for test in analysis.stress_tests[:5]:
        print(f"{test.scenario_name}: {test.pnl_impact:+,.0f} ({test.pnl_impact_pct:+.2%})")

    # Test position sizing
    print("\nPOSITION SIZING")
    print("-" * 50)
    kelly = engine.kelly_criterion(0.65, 200, 100, 0.5)
    print(f"Kelly Criterion (65% win rate): {kelly:.2%} of portfolio")

    contracts = engine.fixed_fractional(0.02, 500, portfolio_value)
    print(f"Fixed Fractional (2% risk, $500 max loss): {contracts} contracts")

    print("\n✅ Portfolio risk engine ready!")

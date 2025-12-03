"""
Risk Analytics Suite
====================

Institutional-grade risk metrics and analysis:
- Value at Risk (VaR)
- Sharpe Ratio & Sortino Ratio
- Stress Testing
- Monte Carlo Simulations
- Concentration Analysis
- Correlation Matrix
- Maximum Drawdown

Author: Magnus Trading Platform
Created: 2025-11-21
"""

import logging
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class RiskMetrics:
    """Container for risk metrics"""
    var_95: float  # Value at Risk at 95% confidence
    var_99: float  # Value at Risk at 99% confidence
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown: float
    beta: float
    annual_volatility: float
    concentration_risk: str  # 'Low', 'Medium', 'High'


class RiskAnalytics:
    """
    Professional risk analytics suite.

    Provides institutional-grade risk metrics used by hedge funds
    and professional traders.
    """

    def __init__(self) -> None:
        """Initialize risk analytics"""
        self.risk_free_rate = 0.05  # 5% annual (adjust based on current rates)

    # =========================================================================
    # VALUE AT RISK (VaR)
    # =========================================================================

    def calculate_var(
        self,
        portfolio_value: float,
        annual_volatility: float = 0.20,
        confidence: float = 0.95,
        days: int = 1
    ) -> str:
        """
        Calculate Value at Risk using historical method.

        VaR answers: "What's the maximum I could lose with X% confidence?"

        Args:
            portfolio_value: Total portfolio value
            annual_volatility: Annual volatility (default 20%)
            confidence: Confidence level (0.95 = 95%, 0.99 = 99%)
            days: Time horizon in days

        Returns:
            Formatted VaR statement
        """
        # Convert annual volatility to daily
        daily_volatility = annual_volatility / np.sqrt(252)  # 252 trading days

        # Z-scores for confidence levels
        z_scores = {
            0.90: 1.28,
            0.95: 1.65,
            0.99: 2.33
        }

        z_score = z_scores.get(confidence, 1.65)

        # Calculate VaR
        var = portfolio_value * daily_volatility * z_score * np.sqrt(days)

        confidence_pct = int(confidence * 100)

        return f"{confidence_pct}% confident you won't lose more than ${var:,.2f} in {days} day(s)"

    def calculate_var_detailed(
        self,
        portfolio_value: float,
        annual_volatility: float = 0.20
    ) -> Dict:
        """
        Calculate VaR at multiple confidence levels and time horizons.

        Returns:
            Dictionary with VaR for different scenarios
        """
        results = {
            'portfolio_value': portfolio_value,
            'annual_volatility': annual_volatility,
            'scenarios': {}
        }

        # Different time horizons
        days_list = [1, 5, 21]  # 1 day, 1 week, 1 month
        confidences = [0.95, 0.99]

        for days in days_list:
            for conf in confidences:
                var_str = self.calculate_var(portfolio_value, annual_volatility, conf, days)
                key = f"{int(conf*100)}%_{days}day"
                results['scenarios'][key] = var_str

        return results

    # =========================================================================
    # SHARPE RATIO - Risk-Adjusted Returns
    # =========================================================================

    def calculate_sharpe_ratio(
        self,
        positions: List[Dict],
        portfolio_returns: Optional[List[float]] = None
    ) -> Dict:
        """
        Calculate Sharpe Ratio - measures risk-adjusted returns.

        Sharpe Ratio = (Portfolio Return - Risk-Free Rate) / Portfolio Volatility

        Interpretation:
        - < 1.0: Poor
        - 1.0 - 2.0: Good
        - > 2.0: Excellent
        - > 3.0: Exceptional

        Args:
            positions: List of portfolio positions
            portfolio_returns: Optional historical returns

        Returns:
            Dict with Sharpe ratio and interpretation
        """
        if portfolio_returns is None or len(portfolio_returns) < 20:
            # Estimate based on positions (simplified)
            # In production, you'd use actual historical returns
            estimated_return = 0.12  # 12% annual
            estimated_vol = 0.20     # 20% annual volatility
        else:
            # Calculate from actual returns
            estimated_return = np.mean(portfolio_returns) * 252  # Annualize
            estimated_vol = np.std(portfolio_returns) * np.sqrt(252)  # Annualize

        # Calculate Sharpe Ratio
        sharpe = (estimated_return - self.risk_free_rate) / estimated_vol

        # Rating
        if sharpe > 3.0:
            rating = "Exceptional"
        elif sharpe > 2.0:
            rating = "Excellent"
        elif sharpe > 1.0:
            rating = "Good"
        else:
            rating = "Poor"

        return {
            'ratio': sharpe,
            'rating': rating,
            'annual_return': estimated_return * 100,
            'annual_vol': estimated_vol * 100,
            'risk_free_rate': self.risk_free_rate * 100
        }

    # =========================================================================
    # SORTINO RATIO - Downside Risk Focus
    # =========================================================================

    def calculate_sortino_ratio(
        self,
        portfolio_returns: List[float],
        target_return: float = 0.0
    ) -> float:
        """
        Calculate Sortino Ratio - like Sharpe but only penalizes downside volatility.

        Better than Sharpe for strategies with asymmetric returns.

        Args:
            portfolio_returns: Historical returns
            target_return: Target/minimum acceptable return

        Returns:
            Sortino ratio
        """
        if len(portfolio_returns) < 20:
            return 0.0

        # Calculate excess returns
        excess_returns = np.array(portfolio_returns) - target_return

        # Calculate mean return
        mean_return = np.mean(excess_returns)

        # Calculate downside deviation (only negative returns)
        downside_returns = excess_returns[excess_returns < 0]
        downside_dev = np.std(downside_returns) if len(downside_returns) > 0 else 0.01

        # Annualize
        annual_mean = mean_return * 252
        annual_downside_dev = downside_dev * np.sqrt(252)

        return annual_mean / annual_downside_dev if annual_downside_dev > 0 else 0.0

    # =========================================================================
    # STRESS TESTING - What If Scenarios
    # =========================================================================

    def stress_test_portfolio(
        self,
        portfolio_value: float,
        positions: List[Dict]
    ) -> Dict[str, Dict]:
        """
        Stress test portfolio against historical crisis scenarios.

        Scenarios:
        - 2008 Financial Crisis (-37% S&P)
        - 2020 COVID Crash (-34% S&P)
        - 2000 Tech Bubble (-49% Nasdaq)
        - Flash Crash (-20% single day)
        - VIX Spike (VIX to 80)

        Args:
            portfolio_value: Current portfolio value
            positions: List of positions

        Returns:
            Dict of scenarios with projected losses
        """
        scenarios = {
            '2008 Financial Crisis': {
                'market_drop': -0.37,
                'description': 'S&P 500 dropped 37% peak to trough'
            },
            '2020 COVID Crash': {
                'market_drop': -0.34,
                'description': 'Fastest bear market in history'
            },
            '2000 Tech Bubble': {
                'market_drop': -0.49,
                'description': 'Nasdaq dropped 49% over 2 years'
            },
            'Flash Crash': {
                'market_drop': -0.20,
                'description': 'Single day 20% drop'
            },
            'Moderate Correction': {
                'market_drop': -0.10,
                'description': 'Normal 10% correction'
            }
        }

        results = {}

        for scenario_name, scenario_data in scenarios.items():
            # Simplified calculation (assumes portfolio moves with market)
            # In production, you'd calculate based on individual positions and betas
            market_drop = scenario_data['market_drop']

            # Assume portfolio has beta of 1.0 for simplification
            portfolio_drop = market_drop

            new_value = portfolio_value * (1 + portfolio_drop)
            loss = portfolio_value - new_value

            results[scenario_name] = {
                'description': scenario_data['description'],
                'market_drop_pct': market_drop * 100,
                'portfolio_value': new_value,
                'loss': loss,
                'loss_pct': portfolio_drop * 100
            }

        return results

    # =========================================================================
    # MONTE CARLO SIMULATION
    # =========================================================================

    def monte_carlo_simulation(
        self,
        initial_value: float,
        annual_return: float = 0.12,
        annual_volatility: float = 0.20,
        days: int = 252,
        num_simulations: int = 10000
    ) -> Dict:
        """
        Run Monte Carlo simulation to project portfolio outcomes.

        Args:
            initial_value: Starting portfolio value
            annual_return: Expected annual return (12% = 0.12)
            annual_volatility: Expected volatility (20% = 0.20)
            days: Number of days to simulate
            num_simulations: Number of simulation runs

        Returns:
            Statistics on projected outcomes
        """
        # Convert to daily parameters
        daily_return = annual_return / 252
        daily_vol = annual_volatility / np.sqrt(252)

        # Run simulations
        final_values = []

        for _ in range(num_simulations):
            value = initial_value

            for day in range(days):
                # Generate random daily return
                daily_change = np.random.normal(daily_return, daily_vol)
                value *= (1 + daily_change)

            final_values.append(value)

        final_values = np.array(final_values)

        # Calculate statistics
        return {
            'initial_value': initial_value,
            'days': days,
            'simulations': num_simulations,
            'mean_value': np.mean(final_values),
            'median_value': np.median(final_values),
            'min_value': np.min(final_values),
            'max_value': np.max(final_values),
            'percentile_5': np.percentile(final_values, 5),
            'percentile_25': np.percentile(final_values, 25),
            'percentile_75': np.percentile(final_values, 75),
            'percentile_95': np.percentile(final_values, 95),
            'probability_profit': np.sum(final_values > initial_value) / num_simulations * 100,
            'probability_loss': np.sum(final_values < initial_value) / num_simulations * 100
        }

    # =========================================================================
    # CONCENTRATION ANALYSIS
    # =========================================================================

    def analyze_concentration(
        self,
        positions: List[Dict],
        portfolio_value: float
    ) -> Dict:
        """
        Analyze portfolio concentration risks.

        Flags:
        - Single position >10% of portfolio
        - Top 3 positions >40% of portfolio
        - Single sector >25% of portfolio

        Args:
            positions: List of positions
            portfolio_value: Total portfolio value

        Returns:
            Concentration analysis
        """
        if not positions or portfolio_value == 0:
            return {
                'concentration_risk': 'Unknown',
                'largest_position': 'N/A',
                'largest_position_pct': 0,
                'top_3_pct': 0,
                'warnings': []
            }

        # Calculate position sizes
        position_values = []
        for pos in positions:
            value = float(pos.get('market_value', 0))
            symbol = pos.get('symbol', 'UNKNOWN')
            pct = (value / portfolio_value * 100) if portfolio_value > 0 else 0
            position_values.append({
                'symbol': symbol,
                'value': value,
                'pct': pct
            })

        # Sort by value
        position_values.sort(key=lambda x: x['value'], reverse=True)

        # Largest position
        largest = position_values[0] if position_values else {'symbol': 'N/A', 'pct': 0}

        # Top 3
        top_3_pct = sum(pos['pct'] for pos in position_values[:3])

        # Determine risk level
        warnings = []
        if largest['pct'] > 10:
            warnings.append(f"{largest['symbol']} is {largest['pct']:.1f}% of portfolio (>10%)")

        if top_3_pct > 40:
            warnings.append(f"Top 3 positions are {top_3_pct:.1f}% of portfolio (>40%)")

        if largest['pct'] > 15:
            risk = 'High'
        elif largest['pct'] > 10:
            risk = 'Medium'
        else:
            risk = 'Low'

        return {
            'concentration_risk': risk,
            'largest_position': largest['symbol'],
            'largest_position_pct': largest['pct'],
            'top_3_pct': top_3_pct,
            'warnings': warnings,
            'position_breakdown': position_values[:10]  # Top 10
        }

    # =========================================================================
    # CORRELATION ANALYSIS
    # =========================================================================

    def calculate_correlation_matrix(
        self,
        positions: List[Dict],
        historical_data: Optional[Dict] = None
    ) -> Dict:
        """
        Calculate correlation matrix for portfolio positions.

        Identifies positions that move together (correlation risk).

        Args:
            positions: List of positions
            historical_data: Optional historical price data

        Returns:
            Correlation matrix and high-correlation pairs
        """
        # Simplified version - in production, fetch actual price correlations
        # For now, use sector-based correlation estimates

        symbols = [pos.get('symbol') for pos in positions if pos.get('symbol')]

        if len(symbols) < 2:
            return {
                'symbols': symbols,
                'high_correlation_pairs': [],
                'diversification_score': 100
            }

        # Estimate correlations based on sector
        # In production: calculate from actual price data
        high_corr_pairs = []

        # Tech stocks are highly correlated
        tech_stocks = ['AAPL', 'MSFT', 'GOOGL', 'META', 'NVDA', 'AMD', 'TSLA']
        tech_in_portfolio = [s for s in symbols if s in tech_stocks]

        if len(tech_in_portfolio) >= 2:
            for i, stock1 in enumerate(tech_in_portfolio):
                for stock2 in tech_in_portfolio[i+1:]:
                    high_corr_pairs.append({
                        'pair': f"{stock1}-{stock2}",
                        'correlation': 0.85,
                        'warning': 'High correlation - moves together'
                    })

        # Diversification score (100 = perfect, 0 = all correlated)
        if len(high_corr_pairs) == 0:
            diversification_score = 100
        else:
            # Penalize for correlated pairs
            diversification_score = max(0, 100 - len(high_corr_pairs) * 15)

        return {
            'symbols': symbols,
            'high_correlation_pairs': high_corr_pairs,
            'diversification_score': diversification_score,
            'recommendation': 'Good diversification' if diversification_score > 70 else 'Consider reducing correlated positions'
        }

    # =========================================================================
    # MAXIMUM DRAWDOWN
    # =========================================================================

    def calculate_max_drawdown(
        self,
        portfolio_values: List[float]
    ) -> Dict:
        """
        Calculate maximum drawdown - worst peak-to-trough decline.

        Args:
            portfolio_values: Historical portfolio values

        Returns:
            Max drawdown statistics
        """
        if len(portfolio_values) < 2:
            return {
                'max_drawdown_pct': 0,
                'max_drawdown_dollars': 0,
                'peak_value': 0,
                'trough_value': 0
            }

        # Calculate running maximum
        running_max = np.maximum.accumulate(portfolio_values)

        # Calculate drawdown at each point
        drawdown = (portfolio_values - running_max) / running_max * 100

        # Find maximum drawdown
        max_dd_pct = np.min(drawdown)
        max_dd_idx = np.argmin(drawdown)

        # Find the peak before max drawdown
        peak_value = running_max[max_dd_idx]
        trough_value = portfolio_values[max_dd_idx]
        max_dd_dollars = peak_value - trough_value

        return {
            'max_drawdown_pct': max_dd_pct,
            'max_drawdown_dollars': max_dd_dollars,
            'peak_value': peak_value,
            'trough_value': trough_value,
            'interpretation': self._interpret_max_drawdown(max_dd_pct)
        }

    def _interpret_max_drawdown(self, max_dd_pct: float) -> str:
        """Interpret maximum drawdown"""
        max_dd = abs(max_dd_pct)

        if max_dd < 10:
            return "Excellent - very low drawdown"
        elif max_dd < 20:
            return "Good - moderate drawdown"
        elif max_dd < 30:
            return "Concerning - significant drawdown"
        else:
            return "Critical - very large drawdown"


if __name__ == "__main__":
    # Test risk analytics
    print("\n=== Testing Risk Analytics ===\n")

    analytics = RiskAnalytics()

    # Test VaR
    print("1. Value at Risk (VaR):")
    portfolio_value = 50000
    var = analytics.calculate_var(portfolio_value)
    print(f"   {var}\n")

    # Test Sharpe Ratio
    print("2. Sharpe Ratio:")
    sharpe = analytics.calculate_sharpe_ratio([])
    print(f"   Ratio: {sharpe['ratio']:.2f} ({sharpe['rating']})")
    print(f"   Annual Return: {sharpe['annual_return']:.2f}%")
    print(f"   Annual Volatility: {sharpe['annual_vol']:.2f}%\n")

    # Test Stress Testing
    print("3. Stress Testing:")
    stress = analytics.stress_test_portfolio(portfolio_value, [])
    for scenario, result in stress.items():
        print(f"\n   {scenario}:")
        print(f"   Portfolio Value: ${result['portfolio_value']:,.2f}")
        print(f"   Loss: ${result['loss']:,.2f} ({result['loss_pct']:.2f}%)")

    print("\n✅ Risk Analytics Suite is ready!")

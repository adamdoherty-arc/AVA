"""
Risk Manager Agent
==================

Evaluates risk and position sizing.
"""

from datetime import datetime
from typing import Any, Optional, List, Dict

from src.ava.strategies.base import StrategySetup, MarketContext
from ..orchestrator import AgentAnalysis, Conviction


class RiskManager:
    """
    Risk management agent that evaluates:
    - Position sizing recommendations
    - Portfolio concentration
    - Correlation risks
    - Maximum loss scenarios
    - Risk limit compliance
    """

    def __init__(self, llm_client: Optional[Any] = None):
        self.llm_client = llm_client
        self.name = "Risk Manager"

        # Risk limits
        self.max_position_pct = 0.05  # 5% max per position
        self.max_daily_risk_pct = 0.02  # 2% daily risk
        self.max_portfolio_delta = 300
        self.max_correlated_positions = 3

    async def analyze(
        self,
        setup: StrategySetup,
        context: MarketContext,
        portfolio_value: float = 100000,
        current_positions: List[Dict] = None
    ) -> AgentAnalysis:
        """Perform risk analysis"""

        current_positions = current_positions or []

        bullish_factors = []
        bearish_factors = []
        neutral_factors = []
        risks = []
        score = 0

        # Position size analysis
        max_loss = setup.max_loss
        if max_loss != float('inf') and max_loss > 0:
            loss_pct = max_loss / portfolio_value
            if loss_pct <= 0.01:
                bullish_factors.append(f"Small position risk: {loss_pct:.2%} of portfolio")
                score += 15
            elif loss_pct <= 0.02:
                bullish_factors.append(f"Acceptable position risk: {loss_pct:.2%}")
                score += 10
            elif loss_pct <= 0.05:
                neutral_factors.append(f"Moderate position risk: {loss_pct:.2%}")
            else:
                bearish_factors.append(f"Large position risk: {loss_pct:.2%}")
                risks.append(f"Position exceeds {self.max_position_pct:.0%} max allocation")
                score -= 20
        else:
            bearish_factors.append("UNDEFINED MAX LOSS - cannot properly size")
            risks.append("Undefined risk makes position sizing impossible")
            score -= 30

        # Concentration analysis
        symbol = setup.symbol
        existing_exposure = sum(
            abs(pos.get('market_value', 0))
            for pos in current_positions
            if pos.get('symbol') == symbol or pos.get('underlying') == symbol
        )

        if existing_exposure > 0:
            total_exposure = existing_exposure + max_loss
            exposure_pct = total_exposure / portfolio_value
            if exposure_pct > 0.15:
                bearish_factors.append(f"High concentration in {symbol}: {exposure_pct:.1%}")
                risks.append("Excessive single-name concentration")
                score -= 20
            elif exposure_pct > 0.10:
                neutral_factors.append(f"Moderate concentration in {symbol}: {exposure_pct:.1%}")
                score -= 5
            else:
                bullish_factors.append(f"Acceptable concentration: {exposure_pct:.1%}")
        else:
            bullish_factors.append(f"New position - no existing {symbol} exposure")
            score += 5

        # Portfolio delta impact
        current_delta = sum(pos.get('delta', 0) * pos.get('quantity', 1) for pos in current_positions)
        new_delta = current_delta + setup.net_delta

        if abs(new_delta) > self.max_portfolio_delta:
            bearish_factors.append(f"Would exceed delta limit: {new_delta:.0f}")
            risks.append("Portfolio delta would exceed risk limits")
            score -= 15
        elif abs(new_delta) > self.max_portfolio_delta * 0.8:
            neutral_factors.append(f"Approaching delta limit: {new_delta:.0f}")
            score -= 5
        else:
            bullish_factors.append(f"Portfolio delta within limits: {new_delta:.0f}")
            score += 5

        # Correlation risk
        correlated_count = sum(
            1 for pos in current_positions
            if self._is_correlated(pos.get('symbol', ''), symbol)
        )

        if correlated_count >= self.max_correlated_positions:
            bearish_factors.append(f"Already {correlated_count} correlated positions")
            risks.append("High correlation concentration")
            score -= 10
        elif correlated_count > 0:
            neutral_factors.append(f"{correlated_count} existing correlated positions")

        # Position count
        num_positions = len([p for p in current_positions if not p.get('is_closed', False)])
        if num_positions >= 10:
            neutral_factors.append(f"Portfolio has {num_positions} open positions")
            risks.append("Many positions to manage")
        else:
            bullish_factors.append(f"Manageable position count: {num_positions}")

        # Risk/reward from risk perspective
        if setup.max_profit > 0 and max_loss > 0 and max_loss != float('inf'):
            rr = max_loss / setup.max_profit
            if rr <= 2:
                bullish_factors.append(f"Favorable risk profile: {rr:.1f}:1")
                score += 10
            elif rr <= 3:
                neutral_factors.append(f"Acceptable risk profile: {rr:.1f}:1")
            else:
                bearish_factors.append(f"Unfavorable risk profile: {rr:.1f}:1")
                score -= 10

        # Calculate recommended position size
        if max_loss > 0 and max_loss != float('inf'):
            risk_budget = portfolio_value * self.max_daily_risk_pct
            recommended_contracts = max(1, int(risk_budget / max_loss))
        else:
            recommended_contracts = 1

        # Determine conviction
        if score >= 20:
            conviction = Conviction.HIGH
        elif score >= 5:
            conviction = Conviction.MODERATE
        elif score >= -10:
            conviction = Conviction.LOW
        else:
            conviction = Conviction.VERY_LOW

        # Generate summary
        if score >= 15:
            summary = f"Risk analysis APPROVES this trade. Position fits well within risk parameters."
        elif score >= 0:
            summary = f"Risk analysis gives CONDITIONAL APPROVAL. Some risk factors to monitor."
        elif score >= -15:
            summary = f"Risk analysis has CONCERNS. Significant risks identified."
        else:
            summary = f"Risk analysis REJECTS this trade. Risk limits would be violated."

        return AgentAnalysis(
            agent_name=self.name,
            agent_type="risk",
            timestamp=datetime.now(),
            summary=summary,
            score=score,
            conviction=conviction,
            bullish_factors=bullish_factors,
            bearish_factors=bearish_factors,
            neutral_factors=neutral_factors,
            risks=risks,
            confidence=0.9,  # High confidence - objective risk metrics
            metrics={
                'max_loss_pct': max_loss / portfolio_value if max_loss != float('inf') else None,
                'recommended_contracts': recommended_contracts,
                'portfolio_delta_after': new_delta,
                'concentration': existing_exposure / portfolio_value if existing_exposure > 0 else 0
            }
        )

    def _is_correlated(self, symbol1: str, symbol2: str) -> bool:
        """Check if two symbols are correlated"""
        # Simplified correlation check
        tech = ['AAPL', 'MSFT', 'GOOGL', 'META', 'NVDA', 'AMD', 'INTC', 'QQQ']
        indices = ['SPY', 'QQQ', 'IWM', 'DIA']
        financials = ['JPM', 'BAC', 'GS', 'MS', 'XLF']

        s1, s2 = symbol1.upper(), symbol2.upper()

        # Same symbol
        if s1 == s2:
            return True

        # Both indices (highly correlated)
        if s1 in indices and s2 in indices:
            return True

        # Both in same sector
        if s1 in tech and s2 in tech:
            return True
        if s1 in financials and s2 in financials:
            return True

        return False

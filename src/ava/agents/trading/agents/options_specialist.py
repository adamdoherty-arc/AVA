"""
Options Specialist Agent
========================

Analyzes options-specific factors for trading decisions.
"""

from datetime import datetime
from typing import Any, Optional

from src.ava.strategies.base import StrategySetup, MarketContext
from ..orchestrator import AgentAnalysis, Conviction


class OptionsSpecialist:
    """
    Options analysis agent that evaluates:
    - IV rank and percentile
    - Greeks analysis
    - Strategy-specific considerations
    - Probability of profit
    - Risk/reward characteristics
    """

    def __init__(self, llm_client: Optional[Any] = None):
        self.llm_client = llm_client
        self.name = "Options Specialist"

    async def analyze(
        self,
        setup: StrategySetup,
        context: MarketContext
    ) -> AgentAnalysis:
        """Perform options-specific analysis"""

        bullish_factors = []
        bearish_factors = []
        neutral_factors = []
        risks = []
        score = 0

        # IV Rank Analysis
        iv_rank = context.iv_rank
        is_credit_strategy = setup.net_theta > 0  # Selling premium

        if is_credit_strategy:
            # Credit strategies want high IV
            if iv_rank >= 70:
                bullish_factors.append(f"Excellent IV rank for selling premium: {iv_rank:.0f}")
                score += 25
            elif iv_rank >= 50:
                bullish_factors.append(f"Good IV rank: {iv_rank:.0f}")
                score += 15
            elif iv_rank >= 30:
                neutral_factors.append(f"Moderate IV rank: {iv_rank:.0f}")
                score += 5
            else:
                bearish_factors.append(f"Low IV rank for credit strategy: {iv_rank:.0f}")
                risks.append("Low IV means less premium to collect")
                score -= 15
        else:
            # Debit strategies want low IV
            if iv_rank <= 30:
                bullish_factors.append(f"Low IV favorable for buying options: {iv_rank:.0f}")
                score += 20
            elif iv_rank <= 50:
                bullish_factors.append(f"Moderate IV: {iv_rank:.0f}")
                score += 10
            else:
                bearish_factors.append(f"High IV makes options expensive: {iv_rank:.0f}")
                risks.append("IV crush risk on debit position")
                score -= 10

        # Probability of Profit
        pop = setup.probability_of_profit
        if pop >= 0.80:
            bullish_factors.append(f"Very high POP: {pop:.1%}")
            score += 20
        elif pop >= 0.70:
            bullish_factors.append(f"High POP: {pop:.1%}")
            score += 15
        elif pop >= 0.60:
            bullish_factors.append(f"Good POP: {pop:.1%}")
            score += 10
        elif pop >= 0.50:
            neutral_factors.append(f"Moderate POP: {pop:.1%}")
        else:
            bearish_factors.append(f"Low POP: {pop:.1%}")
            risks.append("Less than 50% probability of profit")
            score -= 15

        # Risk/Reward Analysis
        rr = setup.risk_reward_ratio
        if rr <= 1.0:
            bullish_factors.append(f"Excellent risk/reward: {rr:.2f}:1")
            score += 15
        elif rr <= 2.0:
            bullish_factors.append(f"Good risk/reward: {rr:.2f}:1")
            score += 10
        elif rr <= 3.0:
            neutral_factors.append(f"Acceptable risk/reward: {rr:.2f}:1")
        else:
            bearish_factors.append(f"Poor risk/reward: {rr:.2f}:1")
            risks.append("Risk exceeds potential reward significantly")
            score -= 10

        # Greeks Analysis
        # Delta
        net_delta = setup.net_delta
        if abs(net_delta) < 0.10:
            bullish_factors.append(f"Delta neutral: {net_delta:.3f}")
            score += 5
        elif abs(net_delta) < 0.25:
            neutral_factors.append(f"Moderate delta exposure: {net_delta:.3f}")
        else:
            bearish_factors.append(f"High delta exposure: {net_delta:.3f}")
            risks.append("Significant directional risk")
            score -= 5

        # Theta
        net_theta = setup.net_theta
        if net_theta > 0:
            bullish_factors.append(f"Positive theta: ${net_theta:.2f}/day")
            score += 10
        elif net_theta < -0.5:
            bearish_factors.append(f"Significant theta decay: ${net_theta:.2f}/day")
            risks.append("Time decay working against position")
            score -= 10

        # DTE Analysis
        dte = setup.days_to_expiration
        if 30 <= dte <= 45:
            bullish_factors.append(f"Optimal DTE: {dte} days")
            score += 10
        elif 21 <= dte <= 60:
            neutral_factors.append(f"Acceptable DTE: {dte} days")
            score += 5
        elif dte < 14:
            bearish_factors.append(f"Short DTE: {dte} days")
            risks.append("Gamma risk increases near expiration")
            score -= 10
        elif dte > 60:
            neutral_factors.append(f"Long DTE: {dte} days")

        # Expected Value
        ev = setup.expected_value
        if ev > 0:
            bullish_factors.append(f"Positive expected value: ${ev:.2f}")
            score += 10
        else:
            bearish_factors.append(f"Negative expected value: ${ev:.2f}")
            score -= 10

        # Max profit/loss analysis
        if setup.max_loss != float('inf'):
            bullish_factors.append(f"Defined risk: ${setup.max_loss:.2f} max loss")
        else:
            bearish_factors.append("UNDEFINED RISK - unlimited loss potential")
            risks.append("Position has unlimited loss potential")
            score -= 25

        # Determine conviction
        if score >= 40:
            conviction = Conviction.VERY_HIGH
        elif score >= 25:
            conviction = Conviction.HIGH
        elif score >= 10:
            conviction = Conviction.MODERATE
        elif score >= 0:
            conviction = Conviction.LOW
        else:
            conviction = Conviction.VERY_LOW

        # Generate summary
        if score >= 30:
            summary = f"Options analysis STRONGLY FAVORS this {setup.strategy_name}. Excellent setup characteristics."
        elif score >= 15:
            summary = f"Options analysis SUPPORTS this {setup.strategy_name}. Good risk/reward profile."
        elif score >= 0:
            summary = f"Options analysis is NEUTRAL on this {setup.strategy_name}. Mixed characteristics."
        else:
            summary = f"Options analysis DISCOURAGES this {setup.strategy_name}. Unfavorable characteristics."

        return AgentAnalysis(
            agent_name=self.name,
            agent_type="options",
            timestamp=datetime.now(),
            summary=summary,
            score=score,
            conviction=conviction,
            bullish_factors=bullish_factors,
            bearish_factors=bearish_factors,
            neutral_factors=neutral_factors,
            risks=risks,
            confidence=0.85,  # High confidence - direct analysis of setup
            metrics={
                'iv_rank': iv_rank,
                'pop': pop,
                'risk_reward': rr,
                'net_delta': net_delta,
                'net_theta': net_theta,
                'dte': dte,
                'expected_value': ev
            }
        )

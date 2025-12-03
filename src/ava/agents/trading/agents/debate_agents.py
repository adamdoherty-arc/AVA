"""
Debate Agents - Bull and Bear Researchers
==========================================

Agents that argue opposing sides of a trade to provide balanced analysis.
"""

from datetime import datetime
from typing import Any, Optional, List

from src.ava.strategies.base import StrategySetup, MarketContext
from ..orchestrator import AgentAnalysis, Conviction


class BullResearcher:
    """
    Bull researcher that builds the case FOR the trade.
    Actively looks for bullish factors and counterarguments to bearish concerns.
    """

    def __init__(self, llm_client: Optional[Any] = None):
        self.llm_client = llm_client
        self.name = "Bull Researcher"

    async def build_case(
        self,
        setup: StrategySetup,
        context: MarketContext,
        prior_analyses: List[AgentAnalysis]
    ) -> AgentAnalysis:
        """Build the bullish case for the trade"""

        bullish_factors = []
        bearish_factors = []  # Acknowledge but counter
        risks = []
        score = 0

        # Aggregate bullish factors from all analyses
        all_bullish = []
        all_bearish = []

        for analysis in prior_analyses:
            all_bullish.extend(analysis.bullish_factors)
            all_bearish.extend(analysis.bearish_factors)

        # Build strongest bullish arguments
        if setup.probability_of_profit >= 0.70:
            bullish_factors.append(f"HIGH PROBABILITY trade: {setup.probability_of_profit:.1%} POP gives statistical edge")
            score += 25

        if setup.net_theta > 0:
            bullish_factors.append(f"TIME IS ON OUR SIDE: Theta of ${setup.net_theta:.2f}/day means profit just from holding")
            score += 20

        if context.iv_rank >= 50 and setup.net_theta > 0:
            bullish_factors.append(f"PREMIUM RICH environment: IV rank {context.iv_rank:.0f} means we're selling expensive options")
            score += 20

        if setup.max_loss != float('inf'):
            bullish_factors.append(f"DEFINED RISK: Max loss capped at ${setup.max_loss:.0f} - we know our worst case")
            score += 15

        if setup.expected_value > 0:
            bullish_factors.append(f"POSITIVE EXPECTED VALUE: ${setup.expected_value:.2f} mathematical edge")
            score += 15

        if context.days_to_earnings is None or context.days_to_earnings > 30:
            bullish_factors.append("NO EARNINGS RISK: Clear runway without binary events")
            score += 10

        # Counter bearish arguments
        for bearish in all_bearish[:3]:  # Top 3 concerns
            counter = self._counter_argument(bearish, setup, context)
            if counter:
                bullish_factors.append(f"Counter to '{bearish[:30]}...': {counter}")
                score += 5

        # Add any unique bullish factors from analyses
        for bf in all_bullish:
            if bf not in bullish_factors and len(bullish_factors) < 10:
                bullish_factors.append(bf)
                score += 5

        # Build thesis
        thesis = self._build_thesis(setup, context, bullish_factors)

        # Acknowledge risks but frame positively
        risks.append("While risks exist, the setup's edge outweighs concerns")

        conviction = Conviction.HIGH if score >= 50 else Conviction.MODERATE

        return AgentAnalysis(
            agent_name=self.name,
            agent_type="debate_bull",
            timestamp=datetime.now(),
            summary=thesis,
            score=min(score, 80),  # Cap at reasonable level
            conviction=conviction,
            bullish_factors=bullish_factors,
            bearish_factors=bearish_factors,
            neutral_factors=[],
            risks=risks,
            confidence=0.75
        )

    def _counter_argument(self, bearish: str, setup: StrategySetup, context: MarketContext) -> Optional[str]:
        """Generate counter-argument to bearish point"""
        bearish_lower = bearish.lower()

        if 'iv' in bearish_lower and 'low' in bearish_lower:
            return "Low IV also means lower option prices, reducing capital at risk"

        if 'risk' in bearish_lower:
            if setup.max_loss != float('inf'):
                return f"Risk is defined and limited to ${setup.max_loss:.0f}"

        if 'delta' in bearish_lower:
            return "Delta exposure is temporary and can be managed with adjustments"

        if 'earnings' in bearish_lower:
            return "Position can be closed before earnings if needed"

        return None

    def _build_thesis(self, setup: StrategySetup, context: MarketContext, factors: List[str]) -> str:
        """Build bullish thesis statement"""
        top_reasons = factors[:3] if factors else ["Statistical edge present"]

        thesis = f"""BULLISH THESIS for {setup.symbol} {setup.strategy_name}:

This trade offers a compelling risk/reward opportunity. With a {setup.probability_of_profit:.0%} probability of profit and ${setup.max_profit:.0f} max gain potential, the math is in our favor.

Key reasons to take this trade:
1. {top_reasons[0] if len(top_reasons) > 0 else 'N/A'}
2. {top_reasons[1] if len(top_reasons) > 1 else 'N/A'}
3. {top_reasons[2] if len(top_reasons) > 2 else 'N/A'}

RECOMMENDATION: PROCEED WITH TRADE"""

        return thesis


class BearResearcher:
    """
    Bear researcher that builds the case AGAINST the trade.
    Actively looks for risks and weaknesses.
    """

    def __init__(self, llm_client: Optional[Any] = None):
        self.llm_client = llm_client
        self.name = "Bear Researcher"

    async def build_case(
        self,
        setup: StrategySetup,
        context: MarketContext,
        prior_analyses: List[AgentAnalysis]
    ) -> AgentAnalysis:
        """Build the bearish case against the trade"""

        bullish_factors = []  # Acknowledge but counter
        bearish_factors = []
        risks = []
        score = 0

        # Aggregate concerns from all analyses
        all_risks = []
        all_bearish = []

        for analysis in prior_analyses:
            all_risks.extend(analysis.risks)
            all_bearish.extend(analysis.bearish_factors)

        # Build strongest bearish arguments
        if setup.probability_of_profit < 0.60:
            bearish_factors.append(f"COIN FLIP ODDS: Only {setup.probability_of_profit:.1%} POP - barely better than random")
            score += 25
            risks.append("Low probability trade")

        if setup.risk_reward_ratio > 2.5:
            bearish_factors.append(f"POOR RISK/REWARD: Risking {setup.risk_reward_ratio:.1f}x potential profit")
            score += 20
            risks.append("Unfavorable risk/reward")

        if setup.max_loss == float('inf'):
            bearish_factors.append("UNLIMITED RISK: This position can lose MORE than your account")
            score += 30
            risks.append("Undefined/unlimited risk")

        if context.iv_rank < 30 and setup.net_theta > 0:
            bearish_factors.append(f"SELLING CHEAP OPTIONS: IV rank only {context.iv_rank:.0f} - premiums are low")
            score += 15
            risks.append("Low IV environment for selling premium")

        if context.days_to_earnings and context.days_to_earnings < 14:
            bearish_factors.append(f"EARNINGS MINEFIELD: {context.days_to_earnings} days to earnings creates binary risk")
            score += 25
            risks.append("Binary earnings event")

        if setup.days_to_expiration < 14:
            bearish_factors.append(f"GAMMA RISK: Only {setup.days_to_expiration} DTE - gamma can whipsaw position")
            score += 15
            risks.append("Near-term gamma exposure")

        if setup.net_theta < 0:
            bearish_factors.append(f"TIME DECAY ENEMY: Losing ${abs(setup.net_theta):.2f}/day to theta")
            score += 15
            risks.append("Negative theta")

        if abs(setup.net_delta) > 0.30:
            bearish_factors.append(f"DIRECTIONAL BET: High delta ({setup.net_delta:.2f}) means significant price exposure")
            score += 10
            risks.append("Directional risk")

        # Add any unique bearish factors
        for bf in all_bearish:
            if bf not in bearish_factors and len(bearish_factors) < 10:
                bearish_factors.append(bf)
                score += 5

        for risk in all_risks:
            if risk not in risks:
                risks.append(risk)

        # Build thesis
        thesis = self._build_thesis(setup, context, bearish_factors, risks)

        conviction = Conviction.HIGH if score >= 50 else Conviction.MODERATE

        return AgentAnalysis(
            agent_name=self.name,
            agent_type="debate_bear",
            timestamp=datetime.now(),
            summary=thesis,
            score=min(score, 80),
            conviction=conviction,
            bullish_factors=bullish_factors,
            bearish_factors=bearish_factors,
            neutral_factors=[],
            risks=risks,
            confidence=0.75
        )

    def _build_thesis(
        self,
        setup: StrategySetup,
        context: MarketContext,
        factors: List[str],
        risks: List[str]
    ) -> str:
        """Build bearish thesis statement"""
        top_concerns = factors[:3] if factors else ["Insufficient edge"]
        top_risks = risks[:3] if risks else ["Market risk"]

        thesis = f"""BEARISH THESIS against {setup.symbol} {setup.strategy_name}:

This trade has significant flaws that outweigh potential benefits. The risk/reward does not justify capital deployment.

Key concerns:
1. {top_concerns[0] if len(top_concerns) > 0 else 'N/A'}
2. {top_concerns[1] if len(top_concerns) > 1 else 'N/A'}
3. {top_concerns[2] if len(top_concerns) > 2 else 'N/A'}

Critical risks:
- {top_risks[0] if len(top_risks) > 0 else 'N/A'}
- {top_risks[1] if len(top_risks) > 1 else 'N/A'}

RECOMMENDATION: PASS ON THIS TRADE"""

        return thesis

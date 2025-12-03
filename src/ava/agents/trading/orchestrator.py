"""
Trading Orchestrator
====================

Multi-agent trading analysis system that coordinates specialized agents
to analyze trading opportunities and make informed decisions.

Architecture (inspired by TradingAgents):
┌─────────────────────────────────────────────────┐
│           TRADING ORCHESTRATOR                  │
├─────────────────────────────────────────────────┤
│  PARALLEL ANALYSIS PHASE                        │
│  ├── Technical Analyst                          │
│  ├── Fundamental Analyst                        │
│  ├── Options Specialist                         │
│  └── Sentiment Analyst                          │
├─────────────────────────────────────────────────┤
│  DEBATE PHASE                                   │
│  ├── Bull Researcher (bullish thesis)           │
│  └── Bear Researcher (bearish thesis)           │
├─────────────────────────────────────────────────┤
│  RISK & DECISION                                │
│  ├── Risk Manager (limits, sizing)              │
│  └── Trading Decision Maker (final call)        │
└─────────────────────────────────────────────────┘

Author: AVA Trading Platform
Created: 2025-11-28
"""

import asyncio
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Dict, Optional, Any
from enum import Enum
import logging
import json

from src.ava.strategies.base import StrategySetup, MarketContext

logger = logging.getLogger(__name__)


# =============================================================================
# DATA CLASSES
# =============================================================================

class Conviction(Enum):
    """Conviction level for analysis"""
    VERY_HIGH = "very_high"
    HIGH = "high"
    MODERATE = "moderate"
    LOW = "low"
    VERY_LOW = "very_low"


class TradeAction(Enum):
    """Recommended trade action"""
    STRONG_BUY = "strong_buy"
    BUY = "buy"
    HOLD = "hold"
    AVOID = "avoid"
    STRONG_AVOID = "strong_avoid"


@dataclass
class AgentAnalysis:
    """Analysis result from a single agent"""
    agent_name: str
    agent_type: str
    timestamp: datetime

    # Core analysis
    summary: str
    score: float  # -100 to +100
    conviction: Conviction

    # Supporting data
    bullish_factors: List[str] = field(default_factory=list)
    bearish_factors: List[str] = field(default_factory=list)
    neutral_factors: List[str] = field(default_factory=list)

    # Specific metrics (varies by agent)
    metrics: Dict[str, Any] = field(default_factory=dict)

    # Risks identified
    risks: List[str] = field(default_factory=list)

    # Confidence in analysis
    confidence: float = 0.7  # 0-1

    def to_dict(self) -> Dict:
        return {
            'agent': self.agent_name,
            'type': self.agent_type,
            'summary': self.summary,
            'score': self.score,
            'conviction': self.conviction.value,
            'bullish': self.bullish_factors,
            'bearish': self.bearish_factors,
            'risks': self.risks,
            'confidence': self.confidence
        }


@dataclass
class DebateResult:
    """Result of bull vs bear debate"""
    bull_thesis: str
    bear_thesis: str

    bull_arguments: List[str] = field(default_factory=list)
    bear_arguments: List[str] = field(default_factory=list)

    bull_score: float = 0.0  # Strength of bull case
    bear_score: float = 0.0  # Strength of bear case

    winner: str = "neutral"  # bull, bear, neutral
    key_differentiator: str = ""

    def to_dict(self) -> Dict:
        return {
            'bull_thesis': self.bull_thesis,
            'bear_thesis': self.bear_thesis,
            'bull_score': self.bull_score,
            'bear_score': self.bear_score,
            'winner': self.winner,
            'key_point': self.key_differentiator
        }


@dataclass
class TradingDecision:
    """Final trading decision from orchestrator"""
    timestamp: datetime
    symbol: str
    strategy_name: str

    # Decision
    action: TradeAction
    conviction: Conviction
    confidence: float

    # Reasoning
    summary: str
    key_reasons: List[str] = field(default_factory=list)

    # Position sizing recommendation
    recommended_size: int = 0
    max_size: int = 0
    risk_adjusted_size: int = 0

    # Risk parameters
    suggested_stop_loss: float = 0.0
    suggested_profit_target: float = 0.0
    max_days_to_hold: int = 0

    # Agent contributions
    agent_analyses: List[AgentAnalysis] = field(default_factory=list)
    debate_result: Optional[DebateResult] = None

    # Scores
    composite_score: float = 0.0  # -100 to +100
    risk_score: float = 0.0       # 0-100 (higher = riskier)

    # Warnings
    warnings: List[str] = field(default_factory=list)

    def should_trade(self) -> bool:
        """Whether to proceed with the trade"""
        return self.action in [TradeAction.STRONG_BUY, TradeAction.BUY]

    def to_dict(self) -> Dict:
        return {
            'symbol': self.symbol,
            'strategy': self.strategy_name,
            'action': self.action.value,
            'conviction': self.conviction.value,
            'confidence': self.confidence,
            'summary': self.summary,
            'reasons': self.key_reasons,
            'recommended_size': self.recommended_size,
            'composite_score': self.composite_score,
            'warnings': self.warnings
        }

    def summary_report(self) -> str:
        """Generate human-readable summary"""
        return f"""
=== Trading Decision: {self.symbol} - {self.strategy_name} ===
Time: {self.timestamp.strftime('%Y-%m-%d %H:%M')}

DECISION: {self.action.value.upper()}
Conviction: {self.conviction.value}
Confidence: {self.confidence:.0%}
Composite Score: {self.composite_score:+.1f}/100

SUMMARY
{self.summary}

KEY REASONS
{chr(10).join(['• ' + r for r in self.key_reasons])}

POSITION SIZING
• Recommended: {self.recommended_size} contracts
• Max Allowed: {self.max_size} contracts
• Risk-Adjusted: {self.risk_adjusted_size} contracts

RISK PARAMETERS
• Stop Loss: {self.suggested_stop_loss:.0%} of premium
• Profit Target: {self.suggested_profit_target:.0%} of max profit
• Max Hold: {self.max_days_to_hold} days

WARNINGS
{chr(10).join(['⚠️ ' + w for w in self.warnings]) if self.warnings else 'None'}

AGENT SCORES
{chr(10).join([f'• {a.agent_name}: {a.score:+.1f} ({a.conviction.value})' for a in self.agent_analyses])}
"""


# =============================================================================
# TRADING ORCHESTRATOR
# =============================================================================

class TradingOrchestrator:
    """
    Orchestrates multiple AI agents to analyze trading opportunities.

    Usage:
        orchestrator = TradingOrchestrator(llm_client=your_llm)
        decision = await orchestrator.analyze_opportunity(setup, context)
    """

    def __init__(
        self,
        llm_client: Optional[Any] = None,
        use_debate: bool = True,
        parallel_analysis: bool = True,
        min_confidence_threshold: float = 0.6
    ):
        self.llm_client = llm_client
        self.use_debate = use_debate
        self.parallel_analysis = parallel_analysis
        self.min_confidence_threshold = min_confidence_threshold

        # Initialize agents
        from .agents import (
            TechnicalAnalyst,
            FundamentalAnalyst,
            OptionsSpecialist,
            SentimentAnalyst,
            RiskManager,
            BullResearcher,
            BearResearcher,
            TradingDecisionMaker
        )

        self.technical_analyst = TechnicalAnalyst(llm_client)
        self.fundamental_analyst = FundamentalAnalyst(llm_client)
        self.options_specialist = OptionsSpecialist(llm_client)
        self.sentiment_analyst = SentimentAnalyst(llm_client)
        self.risk_manager = RiskManager(llm_client)
        self.bull_researcher = BullResearcher(llm_client)
        self.bear_researcher = BearResearcher(llm_client)
        self.decision_maker = TradingDecisionMaker(llm_client)

        logger.info("Trading Orchestrator initialized with all agents")

    async def analyze_opportunity(
        self,
        setup: StrategySetup,
        context: MarketContext,
        portfolio_value: float = 100000,
        current_positions: List[Dict] = None
    ) -> TradingDecision:
        """
        Comprehensive analysis of a trading opportunity.

        Args:
            setup: Strategy setup to analyze
            context: Current market context
            portfolio_value: Total portfolio value
            current_positions: List of current positions

        Returns:
            TradingDecision with recommendation
        """
        logger.info(f"Analyzing opportunity: {setup.symbol} - {setup.strategy_name}")
        current_positions = current_positions or []

        # Phase 1: Parallel Analysis
        analyses = await self._run_parallel_analysis(setup, context)

        # Phase 2: Bull vs Bear Debate
        debate_result = None
        if self.use_debate:
            debate_result = await self._run_debate(setup, context, analyses)

        # Phase 3: Risk Assessment
        risk_analysis = await self.risk_manager.analyze(
            setup, context, portfolio_value, current_positions
        )
        analyses.append(risk_analysis)

        # Phase 4: Final Decision
        decision = await self.decision_maker.decide(
            setup, context, analyses, debate_result,
            portfolio_value, current_positions
        )

        logger.info(f"Decision: {decision.action.value} with {decision.confidence:.0%} confidence")
        return decision

    async def _run_parallel_analysis(
        self,
        setup: StrategySetup,
        context: MarketContext
    ) -> List[AgentAnalysis]:
        """Run all analysis agents in parallel"""

        if self.parallel_analysis:
            # Run all agents concurrently
            results = await asyncio.gather(
                self.technical_analyst.analyze(setup, context),
                self.fundamental_analyst.analyze(setup, context),
                self.options_specialist.analyze(setup, context),
                self.sentiment_analyst.analyze(setup, context),
                return_exceptions=True
            )

            analyses = []
            for result in results:
                if isinstance(result, AgentAnalysis):
                    analyses.append(result)
                elif isinstance(result, Exception):
                    logger.error(f"Agent error: {result}")

            return analyses
        else:
            # Run sequentially
            return [
                await self.technical_analyst.analyze(setup, context),
                await self.fundamental_analyst.analyze(setup, context),
                await self.options_specialist.analyze(setup, context),
                await self.sentiment_analyst.analyze(setup, context)
            ]

    async def _run_debate(
        self,
        setup: StrategySetup,
        context: MarketContext,
        analyses: List[AgentAnalysis]
    ) -> DebateResult:
        """Run bull vs bear debate"""

        # Get initial arguments from both sides
        bull_analysis = await self.bull_researcher.build_case(setup, context, analyses)
        bear_analysis = await self.bear_researcher.build_case(setup, context, analyses)

        # Score the arguments
        bull_score = self._score_arguments(bull_analysis)
        bear_score = self._score_arguments(bear_analysis)

        # Determine winner
        if bull_score > bear_score + 10:
            winner = "bull"
        elif bear_score > bull_score + 10:
            winner = "bear"
        else:
            winner = "neutral"

        return DebateResult(
            bull_thesis=bull_analysis.summary,
            bear_thesis=bear_analysis.summary,
            bull_arguments=bull_analysis.bullish_factors,
            bear_arguments=bear_analysis.bearish_factors,
            bull_score=bull_score,
            bear_score=bear_score,
            winner=winner,
            key_differentiator=self._find_key_differentiator(bull_analysis, bear_analysis)
        )

    def _score_arguments(self, analysis: AgentAnalysis) -> float:
        """Score the strength of arguments"""
        base_score = analysis.score
        confidence_multiplier = analysis.confidence

        # Adjust for number of supporting factors
        factor_bonus = len(analysis.bullish_factors) * 2 - len(analysis.risks) * 3

        return (base_score + factor_bonus) * confidence_multiplier

    def _find_key_differentiator(
        self,
        bull: AgentAnalysis,
        bear: AgentAnalysis
    ) -> str:
        """Find the key point of disagreement"""
        # Simplified - in production, use LLM to synthesize
        if bull.score > bear.score:
            return f"Bull case stronger: {bull.bullish_factors[0] if bull.bullish_factors else 'N/A'}"
        else:
            return f"Bear case stronger: {bear.bearish_factors[0] if bear.bearish_factors else 'N/A'}"

    def quick_analysis(
        self,
        setup: StrategySetup,
        context: MarketContext
    ) -> TradingDecision:
        """
        Quick synchronous analysis without full agent debate.
        Useful for rapid screening.
        """
        # Simple scoring based on setup characteristics
        score = 0
        reasons = []
        warnings = []

        # POP scoring
        if setup.probability_of_profit >= 0.70:
            score += 25
            reasons.append(f"High probability of profit: {setup.probability_of_profit:.1%}")
        elif setup.probability_of_profit >= 0.60:
            score += 15
        else:
            score -= 10
            warnings.append(f"Low POP: {setup.probability_of_profit:.1%}")

        # Risk/reward scoring
        if setup.risk_reward_ratio <= 1.5:
            score += 20
            reasons.append(f"Excellent risk/reward: {setup.risk_reward_ratio:.2f}:1")
        elif setup.risk_reward_ratio <= 2.5:
            score += 10
        else:
            score -= 15
            warnings.append(f"Poor risk/reward: {setup.risk_reward_ratio:.2f}:1")

        # IV rank scoring for credit strategies
        if context.iv_rank >= 60:
            score += 20
            reasons.append(f"High IV rank: {context.iv_rank:.0f}")
        elif context.iv_rank >= 40:
            score += 10
        elif context.iv_rank < 30:
            score -= 10
            warnings.append(f"Low IV rank: {context.iv_rank:.0f}")

        # Theta scoring
        if setup.net_theta > 0:
            score += 15
            reasons.append(f"Positive theta: ${setup.net_theta:.2f}/day")
        else:
            score -= 5

        # DTE scoring
        if 30 <= setup.days_to_expiration <= 45:
            score += 10
            reasons.append(f"Optimal DTE: {setup.days_to_expiration}")
        elif setup.days_to_expiration < 14:
            warnings.append(f"Short DTE: {setup.days_to_expiration} days")

        # Earnings check
        if context.days_to_earnings and context.days_to_earnings < 14:
            score -= 20
            warnings.append(f"Earnings in {context.days_to_earnings} days")

        # Determine action
        if score >= 60:
            action = TradeAction.STRONG_BUY
            conviction = Conviction.HIGH
        elif score >= 40:
            action = TradeAction.BUY
            conviction = Conviction.MODERATE
        elif score >= 20:
            action = TradeAction.HOLD
            conviction = Conviction.LOW
        elif score >= 0:
            action = TradeAction.AVOID
            conviction = Conviction.LOW
        else:
            action = TradeAction.STRONG_AVOID
            conviction = Conviction.HIGH

        return TradingDecision(
            timestamp=datetime.now(),
            symbol=setup.symbol,
            strategy_name=setup.strategy_name,
            action=action,
            conviction=conviction,
            confidence=min(0.5 + abs(score) / 200, 0.95),
            summary=f"Quick analysis score: {score}/100",
            key_reasons=reasons,
            composite_score=score,
            warnings=warnings
        )


# =============================================================================
# TESTING
# =============================================================================

if __name__ == "__main__":
    from datetime import date, timedelta

    print("\n=== Testing Trading Orchestrator ===\n")

    # Create mock setup
    setup = StrategySetup(
        symbol='SPY',
        strategy_name='Iron Condor',
        legs=[],
        max_profit=250,
        max_loss=750,
        probability_of_profit=0.72,
        net_theta=12.5,
        net_delta=-0.05,
        underlying_price=560,
        iv_rank=55
    )

    # Create mock context
    context = MarketContext(
        symbol='SPY',
        current_price=560,
        previous_close=558,
        iv_rank=55,
        iv_percentile=60,
        implied_volatility=0.15,
        vix=16,
        days_to_earnings=45
    )

    # Test quick analysis
    orchestrator = TradingOrchestrator(use_debate=False)
    decision = orchestrator.quick_analysis(setup, context)

    print(decision.summary_report())

    print("\n✅ Trading Orchestrator ready!")

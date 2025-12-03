"""
Trading Decision Maker Agent
============================

Final decision maker that synthesizes all agent analyses.
"""

from datetime import datetime
from typing import Any, Optional, List, Dict

from src.ava.strategies.base import StrategySetup, MarketContext
from ..orchestrator import (
    AgentAnalysis,
    TradingDecision,
    DebateResult,
    Conviction,
    TradeAction
)


class TradingDecisionMaker:
    """
    Final decision maker that:
    - Synthesizes all agent analyses
    - Weighs different perspectives
    - Considers risk constraints
    - Produces actionable trading decision
    """

    def __init__(self, llm_client: Optional[Any] = None):
        self.llm_client = llm_client
        self.name = "Trading Decision Maker"

        # Agent weights for scoring
        self.weights = {
            'options': 0.30,      # Options specialist most important
            'risk': 0.25,         # Risk manager second
            'technical': 0.20,    # Technical analysis
            'fundamental': 0.15,  # Fundamental analysis
            'sentiment': 0.10     # Sentiment least weight
        }

    async def decide(
        self,
        setup: StrategySetup,
        context: MarketContext,
        analyses: List[AgentAnalysis],
        debate_result: Optional[DebateResult],
        portfolio_value: float = 100000,
        current_positions: List[Dict] = None
    ) -> TradingDecision:
        """Make final trading decision"""

        current_positions = current_positions or []

        # Calculate weighted composite score
        composite_score = self._calculate_composite_score(analyses)

        # Adjust for debate result
        if debate_result:
            if debate_result.winner == "bull":
                composite_score += 10
            elif debate_result.winner == "bear":
                composite_score -= 10

        # Determine action
        action, conviction = self._determine_action(composite_score, analyses)

        # Calculate confidence
        confidence = self._calculate_confidence(analyses, composite_score)

        # Determine position sizing
        recommended_size, max_size, risk_adjusted = self._calculate_position_size(
            setup, portfolio_value, analyses
        )

        # Compile key reasons
        key_reasons = self._compile_key_reasons(analyses, action)

        # Compile warnings
        warnings = self._compile_warnings(analyses)

        # Calculate risk parameters
        stop_loss, profit_target, max_days = self._calculate_risk_params(setup, context)

        # Generate summary
        summary = self._generate_summary(
            setup, action, conviction, composite_score, key_reasons
        )

        return TradingDecision(
            timestamp=datetime.now(),
            symbol=setup.symbol,
            strategy_name=setup.strategy_name,
            action=action,
            conviction=conviction,
            confidence=confidence,
            summary=summary,
            key_reasons=key_reasons,
            recommended_size=recommended_size,
            max_size=max_size,
            risk_adjusted_size=risk_adjusted,
            suggested_stop_loss=stop_loss,
            suggested_profit_target=profit_target,
            max_days_to_hold=max_days,
            agent_analyses=analyses,
            debate_result=debate_result,
            composite_score=composite_score,
            risk_score=self._calculate_risk_score(analyses),
            warnings=warnings
        )

    def _calculate_composite_score(self, analyses: List[AgentAnalysis]) -> float:
        """Calculate weighted composite score from all analyses"""
        total_weight = 0
        weighted_score = 0

        for analysis in analyses:
            agent_type = analysis.agent_type
            weight = self.weights.get(agent_type, 0.1)

            # Adjust weight by confidence
            adjusted_weight = weight * analysis.confidence

            weighted_score += analysis.score * adjusted_weight
            total_weight += adjusted_weight

        if total_weight == 0:
            return 0

        return weighted_score / total_weight

    def _determine_action(
        self,
        composite_score: float,
        analyses: List[AgentAnalysis]
    ) -> tuple[TradeAction, Conviction]:
        """Determine action and conviction from score"""

        # Check for any critical blocks
        risk_analysis = next((a for a in analyses if a.agent_type == 'risk'), None)
        if risk_analysis and risk_analysis.score < -20:
            return TradeAction.STRONG_AVOID, Conviction.HIGH

        # Score-based decision
        if composite_score >= 40:
            return TradeAction.STRONG_BUY, Conviction.VERY_HIGH
        elif composite_score >= 25:
            return TradeAction.STRONG_BUY, Conviction.HIGH
        elif composite_score >= 15:
            return TradeAction.BUY, Conviction.HIGH
        elif composite_score >= 5:
            return TradeAction.BUY, Conviction.MODERATE
        elif composite_score >= -5:
            return TradeAction.HOLD, Conviction.LOW
        elif composite_score >= -15:
            return TradeAction.AVOID, Conviction.MODERATE
        elif composite_score >= -25:
            return TradeAction.AVOID, Conviction.HIGH
        else:
            return TradeAction.STRONG_AVOID, Conviction.VERY_HIGH

    def _calculate_confidence(
        self,
        analyses: List[AgentAnalysis],
        composite_score: float
    ) -> float:
        """Calculate overall confidence in decision"""

        # Base confidence from score magnitude
        base_confidence = min(0.5 + abs(composite_score) / 100, 0.95)

        # Average confidence from agents
        agent_confidence = sum(a.confidence for a in analyses) / len(analyses) if analyses else 0.5

        # Agreement factor - do agents agree?
        scores = [a.score for a in analyses]
        if scores:
            score_std = max(1, abs(max(scores) - min(scores)))
            agreement_factor = 1 - (score_std / 100)
        else:
            agreement_factor = 0.5

        # Combined confidence
        confidence = (base_confidence * 0.4 + agent_confidence * 0.3 + agreement_factor * 0.3)

        return min(max(confidence, 0.3), 0.95)

    def _calculate_position_size(
        self,
        setup: StrategySetup,
        portfolio_value: float,
        analyses: List[AgentAnalysis]
    ) -> tuple[int, int, int]:
        """Calculate position sizing recommendations"""

        # Get risk analysis recommendation if available
        risk_analysis = next((a for a in analyses if a.agent_type == 'risk'), None)

        if risk_analysis and 'recommended_contracts' in risk_analysis.metrics:
            base_size = risk_analysis.metrics['recommended_contracts']
        else:
            # Calculate based on 2% risk
            if setup.max_loss > 0 and setup.max_loss != float('inf'):
                base_size = max(1, int(portfolio_value * 0.02 / setup.max_loss))
            else:
                base_size = 1

        # Max size (5% of portfolio)
        if setup.max_loss > 0 and setup.max_loss != float('inf'):
            max_size = max(1, int(portfolio_value * 0.05 / setup.max_loss))
        else:
            max_size = 2

        # Risk-adjusted size based on composite score
        composite = self._calculate_composite_score(analyses)
        if composite >= 30:
            risk_adjusted = base_size  # Full size
        elif composite >= 15:
            risk_adjusted = max(1, int(base_size * 0.75))  # 75%
        elif composite >= 0:
            risk_adjusted = max(1, int(base_size * 0.5))   # 50%
        else:
            risk_adjusted = 1  # Minimum

        return base_size, max_size, risk_adjusted

    def _compile_key_reasons(
        self,
        analyses: List[AgentAnalysis],
        action: TradeAction
    ) -> List[str]:
        """Compile key reasons for the decision"""

        reasons = []

        if action in [TradeAction.STRONG_BUY, TradeAction.BUY]:
            # Gather top bullish factors
            for analysis in analyses:
                for factor in analysis.bullish_factors[:2]:
                    if factor not in reasons and len(reasons) < 5:
                        reasons.append(factor)
        else:
            # Gather top concerns
            for analysis in analyses:
                for factor in analysis.bearish_factors[:2]:
                    if factor not in reasons and len(reasons) < 5:
                        reasons.append(factor)
                for risk in analysis.risks[:1]:
                    if risk not in reasons and len(reasons) < 5:
                        reasons.append(f"Risk: {risk}")

        return reasons[:5]

    def _compile_warnings(self, analyses: List[AgentAnalysis]) -> List[str]:
        """Compile warnings from all analyses"""
        warnings = []

        for analysis in analyses:
            for risk in analysis.risks:
                if risk not in warnings:
                    warnings.append(risk)

        return warnings[:5]

    def _calculate_risk_params(
        self,
        setup: StrategySetup,
        context: MarketContext
    ) -> tuple[float, float, int]:
        """Calculate risk management parameters"""

        # Stop loss as percentage of premium/max profit
        if setup.net_theta > 0:  # Credit strategy
            stop_loss = 2.0  # 200% of credit received
        else:  # Debit strategy
            stop_loss = 0.50  # 50% of debit paid

        # Profit target
        profit_target = 0.50  # Take profits at 50%

        # Max days to hold
        if setup.days_to_expiration <= 14:
            max_days = setup.days_to_expiration - 2
        elif setup.days_to_expiration <= 30:
            max_days = min(21, setup.days_to_expiration - 7)
        else:
            max_days = min(45, setup.days_to_expiration - 14)

        max_days = max(1, max_days)

        return stop_loss, profit_target, max_days

    def _calculate_risk_score(self, analyses: List[AgentAnalysis]) -> float:
        """Calculate overall risk score (0-100, higher = riskier)"""

        risk_count = sum(len(a.risks) for a in analyses)
        bearish_count = sum(len(a.bearish_factors) for a in analyses)

        # Base risk from counts
        base_risk = min(risk_count * 10 + bearish_count * 5, 100)

        # Adjust for negative scores
        negative_scores = [a.score for a in analyses if a.score < 0]
        score_penalty = abs(sum(negative_scores)) if negative_scores else 0

        return min(base_risk + score_penalty, 100)

    def _generate_summary(
        self,
        setup: StrategySetup,
        action: TradeAction,
        conviction: Conviction,
        score: float,
        reasons: List[str]
    ) -> str:
        """Generate decision summary"""

        action_text = {
            TradeAction.STRONG_BUY: "STRONGLY RECOMMENDED",
            TradeAction.BUY: "RECOMMENDED",
            TradeAction.HOLD: "NEUTRAL - Consider alternatives",
            TradeAction.AVOID: "NOT RECOMMENDED",
            TradeAction.STRONG_AVOID: "STRONGLY ADVISED AGAINST"
        }

        summary = f"""Multi-agent analysis complete for {setup.symbol} {setup.strategy_name}.

VERDICT: {action_text[action]}
Composite Score: {score:+.1f}/100
Conviction: {conviction.value.upper()}

The trading committee of AI analysts has evaluated this opportunity across technical, fundamental, options-specific, sentiment, and risk dimensions.

"""

        if action in [TradeAction.STRONG_BUY, TradeAction.BUY]:
            summary += f"The trade shows favorable characteristics with {setup.probability_of_profit:.0%} probability of profit"
            if setup.net_theta > 0:
                summary += f" and ${setup.net_theta:.2f}/day positive theta decay."
            else:
                summary += "."
        else:
            summary += "Identified risks outweigh potential benefits at this time."

        return summary

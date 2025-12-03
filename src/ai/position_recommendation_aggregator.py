"""
Position Recommendation Aggregator
Combines quantitative and LLM analysis into final recommendations
"""

import logging
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional
from datetime import datetime

from src.ai.position_data_aggregator import EnrichedPosition
from src.ai.position_quantitative_analyzer import QuantitativeAnalysis
from src.ai.position_llm_analyzer import PositionLLMAnalyzer
from src.ai.position_quantitative_analyzer import PositionQuantitativeAnalyzer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class FinalRecommendation:
    """Final synthesized recommendation"""

    # Core recommendation
    action: str  # 'hold', 'close_now', 'roll_out', 'roll_strike', 'add_hedge', 'cut_loss'
    confidence: int  # 0-100
    rationale: str
    key_factors: List[str]

    # Risk assessment
    risk_level: str  # 'low', 'medium', 'high'
    urgency: str  # 'low', 'medium', 'high'

    # Action details
    action_details: Dict
    expected_outcome: str

    # Source signals
    quant_signal: str
    llm_signal: Optional[str]

    # Metadata
    model_used: Optional[str]
    cost: float
    timestamp: datetime

    # Position reference
    position_id: str
    symbol: str
    position_type: str

    def to_dict(self) -> dict:
        """Convert to dictionary"""
        data = asdict(self)
        data['timestamp'] = self.timestamp.isoformat()
        return data


class PositionRecommendationAggregator:
    """
    Aggregates quantitative and LLM recommendations

    Strategy:
    - If both agree → high confidence
    - If disagree → favor quantitative for risk management
    - If LLM confidence < 60% → use quantitative only
    - If critical position (big loss) → always use LLM
    """

    def __init__(self) -> None:
        self.quant_analyzer = PositionQuantitativeAnalyzer()
        self.llm_analyzer = PositionLLMAnalyzer()

    async def get_recommendation(
        self,
        position: EnrichedPosition,
        use_llm: bool = True,
        market_context: Optional[Dict] = None
    ) -> FinalRecommendation:
        """
        Get final recommendation for a position

        Args:
            position: Enriched position data
            use_llm: Whether to use LLM (if False, quant-only)
            market_context: Optional market context

        Returns:
            FinalRecommendation object
        """
        try:
            # Always run quantitative analysis
            quant_analysis = self.quant_analyzer.analyze(position)

            quant_rec = {
                'action': quant_analysis.recommended_action,
                'confidence': quant_analysis.confidence,
                'reasoning': quant_analysis.reasoning,
                'risk_level': quant_analysis.risk_level
            }

            # Run LLM analysis if enabled
            llm_rec = None
            cost = 0.0
            model_used = None

            if use_llm:
                try:
                    llm_rec = await self.llm_analyzer.analyze_position(
                        position,
                        quant_analysis=asdict(quant_analysis),
                        market_context=market_context or {}
                    )
                    cost = llm_rec.get('cost', 0.0)
                    model_used = llm_rec.get('model_used')
                except Exception as e:
                    logger.error(f"LLM analysis failed: {e}")
                    llm_rec = None

            # Synthesize recommendations
            final_rec = self._synthesize_recommendations(
                position,
                quant_rec,
                llm_rec,
                quant_analysis
            )

            # Add metadata
            return FinalRecommendation(
                action=final_rec['action'],
                confidence=final_rec['confidence'],
                rationale=final_rec['rationale'],
                key_factors=final_rec['key_factors'],
                risk_level=final_rec['risk_level'],
                urgency=final_rec['urgency'],
                action_details=final_rec['action_details'],
                expected_outcome=final_rec['expected_outcome'],
                quant_signal=quant_rec['action'],
                llm_signal=llm_rec['recommendation'] if llm_rec else None,
                model_used=model_used,
                cost=cost,
                timestamp=datetime.now(),
                position_id=position.position_id,
                symbol=position.symbol,
                position_type=position.position_type
            )

        except Exception as e:
            logger.error(f"Error generating recommendation: {e}")
            return self._get_fallback_recommendation(position)

    def _synthesize_recommendations(
        self,
        position: EnrichedPosition,
        quant_rec: Dict,
        llm_rec: Optional[Dict],
        quant_analysis: QuantitativeAnalysis
    ) -> Dict:
        """
        Combine quantitative and LLM recommendations

        Decision Logic:
        1. If both agree → use higher confidence
        2. If both disagree:
           - For losing positions → favor LLM (more context-aware)
           - For winning positions → favor quant (less emotional)
           - If LLM confidence < 60% → use quant
        3. If only quant available → use quant
        """

        # If no LLM, return quant recommendation
        if not llm_rec:
            return self._format_quant_recommendation(quant_rec, quant_analysis, position)

        # If both agree, merge with high confidence
        if quant_rec['action'] == llm_rec['recommendation']:
            confidence = min(95, int((quant_rec['confidence'] + llm_rec['confidence']) / 2))

            return {
                'action': quant_rec['action'],
                'confidence': confidence,
                'rationale': llm_rec['rationale'],
                'key_factors': llm_rec.get('key_factors', [quant_rec['reasoning']]),
                'risk_level': self._max_risk_level(quant_rec['risk_level'], llm_rec['risk_level']),
                'urgency': llm_rec.get('urgency', 'medium'),
                'action_details': llm_rec.get('action_details', {}),
                'expected_outcome': llm_rec.get('expected_outcome', 'Following consensus recommendation')
            }

        # If they disagree, apply conflict resolution
        return self._resolve_conflict(position, quant_rec, llm_rec, quant_analysis)

    def _resolve_conflict(
        self,
        position: EnrichedPosition,
        quant_rec: Dict,
        llm_rec: Dict,
        quant_analysis: QuantitativeAnalysis
    ) -> Dict:
        """
        Resolve conflicts between quant and LLM recommendations

        Priority Rules:
        1. Big losses (< -$500) → Trust LLM (better context awareness)
        2. LLM low confidence (< 60%) → Trust quant
        3. Assignment risk (DTE < 7, ITM) → Trust quant (rules-based safety)
        4. High profit (> 50%) → Trust quant (systematic profit-taking)
        5. Otherwise → Weighted blend (60% LLM, 40% quant)
        """

        llm_confidence = llm_rec.get('confidence', 50)

        # Rule 1: Big loss → trust LLM
        if position.pnl_dollar < -500:
            logger.info(f"Conflict resolved: Big loss, favoring LLM")
            return {
                'action': llm_rec['recommendation'],
                'confidence': int(llm_confidence * 0.9),
                'rationale': f"LLM analysis preferred for losing position. {llm_rec['rationale']}",
                'key_factors': llm_rec.get('key_factors', []),
                'risk_level': llm_rec['risk_level'],
                'urgency': llm_rec.get('urgency', 'high'),
                'action_details': llm_rec.get('action_details', {}),
                'expected_outcome': llm_rec.get('expected_outcome', '')
            }

        # Rule 2: LLM low confidence → trust quant
        if llm_confidence < 60:
            logger.info(f"Conflict resolved: Low LLM confidence, favoring quant")
            return self._format_quant_recommendation(quant_rec, quant_analysis, position)

        # Rule 3: Assignment risk → trust quant
        if position.dte < 7 and position.moneyness == 'ITM':
            logger.info(f"Conflict resolved: Assignment risk, favoring quant")
            return self._format_quant_recommendation(quant_rec, quant_analysis, position)

        # Rule 4: High profit → trust quant
        if position.pnl_percent > 50:
            logger.info(f"Conflict resolved: High profit, favoring quant")
            return self._format_quant_recommendation(quant_rec, quant_analysis, position)

        # Rule 5: Default → weighted blend (favor LLM)
        logger.info(f"Conflict resolved: Weighted blend (60% LLM)")

        # Use LLM action but moderate confidence
        blended_confidence = int(llm_confidence * 0.6 + quant_rec['confidence'] * 0.4)

        return {
            'action': llm_rec['recommendation'],
            'confidence': blended_confidence,
            'rationale': f"{llm_rec['rationale']} (Note: Quant analysis suggests {quant_rec['action']})",
            'key_factors': llm_rec.get('key_factors', []),
            'risk_level': self._max_risk_level(quant_rec['risk_level'], llm_rec['risk_level']),
            'urgency': llm_rec.get('urgency', 'medium'),
            'action_details': llm_rec.get('action_details', {}),
            'expected_outcome': llm_rec.get('expected_outcome', '')
        }

    def _format_quant_recommendation(
        self,
        quant_rec: Dict,
        quant_analysis: QuantitativeAnalysis,
        position: EnrichedPosition
    ) -> Dict:
        """Format quantitative recommendation"""

        # Build key factors from quant analysis
        key_factors = [
            f"Position P/L: {position.pnl_percent:+.1f}%",
            f"DTE: {position.dte} days",
            f"Moneyness: {position.moneyness}"
        ]

        # Add risk-specific factors
        if quant_analysis.gamma_risk_level == 'high':
            key_factors.append("High gamma risk detected")

        if position.dte < 7:
            key_factors.append("Approaching expiration")

        # Determine urgency
        urgency = self._determine_urgency(position, quant_rec['action'])

        return {
            'action': quant_rec['action'],
            'confidence': quant_rec['confidence'],
            'rationale': quant_rec['reasoning'],
            'key_factors': key_factors,
            'risk_level': quant_rec['risk_level'],
            'urgency': urgency,
            'action_details': {
                'target_exit_price': quant_analysis.optimal_exit_price,
                'stop_loss_price': quant_analysis.stop_loss_price
            },
            'expected_outcome': self._get_expected_outcome(quant_rec['action'], position)
        }

    def _determine_urgency(self, position: EnrichedPosition, action: str) -> str:
        """Determine action urgency"""

        # Critical urgency
        if position.dte <= 3:
            return 'high'

        if position.pnl_percent < -100:
            return 'high'

        if action in ['cut_loss', 'add_hedge']:
            return 'high'

        # Medium urgency
        if position.dte <= 7:
            return 'medium'

        if action in ['roll_out', 'close_now']:
            return 'medium'

        # Low urgency
        return 'low'

    def _get_expected_outcome(self, action: str, position: EnrichedPosition) -> str:
        """Generate expected outcome text"""

        outcomes = {
            'hold': f"Position will continue to decay via theta. Target {position.dte} days for full profit.",
            'close_now': f"Lock in current P/L of {position.pnl_percent:+.1f}%.",
            'roll_out': f"Extend duration to collect additional premium and avoid assignment.",
            'roll_strike': f"Adjust strike to improve probability of success.",
            'add_hedge': f"Reduce risk exposure while maintaining position.",
            'cut_loss': f"Exit to prevent further losses beyond current {position.pnl_percent:.1f}%."
        }

        return outcomes.get(action, "Execute recommended action")

    def _max_risk_level(self, level1: str, level2: str) -> str:
        """Return higher of two risk levels"""
        levels = ['low', 'medium', 'high']
        idx1 = levels.index(level1) if level1 in levels else 1
        idx2 = levels.index(level2) if level2 in levels else 1
        return levels[max(idx1, idx2)]

    def _get_fallback_recommendation(self, position: EnrichedPosition) -> FinalRecommendation:
        """Generate safe fallback recommendation"""

        return FinalRecommendation(
            action='hold',
            confidence=50,
            rationale='Analysis error - defaulting to hold. Review position manually.',
            key_factors=['Analysis system error'],
            risk_level='medium',
            urgency='low',
            action_details={},
            expected_outcome='Manual review required',
            quant_signal='hold',
            llm_signal=None,
            model_used=None,
            cost=0.0,
            timestamp=datetime.now(),
            position_id=position.position_id,
            symbol=position.symbol,
            position_type=position.position_type
        )


# ============================================================================
# Batch Processing
# ============================================================================

async def analyze_portfolio(
    positions: List[EnrichedPosition],
    use_llm: bool = True
) -> List[FinalRecommendation]:
    """
    Analyze entire portfolio

    Args:
        positions: List of positions
        use_llm: Use LLM analysis

    Returns:
        List of recommendations
    """
    aggregator = PositionRecommendationAggregator()

    recommendations = []
    for position in positions:
        try:
            rec = await aggregator.get_recommendation(position, use_llm=use_llm)
            recommendations.append(rec)
        except Exception as e:
            logger.error(f"Error analyzing {position.symbol}: {e}")
            recommendations.append(aggregator._get_fallback_recommendation(position))

    return recommendations


# ============================================================================
# Testing
# ============================================================================

async def test_aggregator():
    """Test recommendation aggregator"""
    from datetime import date, timedelta
    import asyncio

    print("\n" + "="*80)
    print("RECOMMENDATION AGGREGATOR TEST")
    print("="*80)

    # Create test position
    test_position = EnrichedPosition(
        symbol='AAPL',
        position_type='CSP',
        strike=150.0,
        expiration=date.today() + timedelta(days=5),  # Near expiration
        dte=5,
        quantity=1,
        premium_collected=250.0,
        current_value=350.0,
        pnl_dollar=-100.0,
        pnl_percent=-40.0,
        stock_price=148.5,  # Close to strike
        stock_price_ah=148.3,
        stock_change_percent=-1.5,
        delta=-0.45,
        gamma=0.08,
        theta=3.5,
        vega=0.12,
        implied_volatility=35.0,
        moneyness='ATM',
        distance_to_strike=-1.0,
        probability_itm=45.0,
        iv_rank=45.0,
        iv_percentile=50.0,
        stock_rsi=48.0,
        stock_trend='neutral',
        support_level=145.0,
        resistance_level=155.0,
        news_sentiment=-0.1,
        news_count_24h=5,
        analyzed_at=datetime.now(),
        position_id='AAPL_150_2025-12-20_put'
    )

    aggregator = PositionRecommendationAggregator()

    # Test with LLM
    print("\n" + "="*80)
    print("TEST 1: With LLM Analysis")
    print("="*80)

    rec = await aggregator.get_recommendation(test_position, use_llm=True)

    print(f"\nPosition: {rec.symbol} ${test_position.strike} {rec.position_type}")
    print(f"Status: ${test_position.pnl_dollar:.2f} ({test_position.pnl_percent:+.1f}%)")
    print(f"DTE: {test_position.dte} | Moneyness: {test_position.moneyness}")

    print("\n" + "-"*80)
    print(f"ACTION: {rec.action.upper()}")
    print(f"Confidence: {rec.confidence}%")
    print(f"Risk: {rec.risk_level.upper()} | Urgency: {rec.urgency.upper()}")
    print(f"\nRationale: {rec.rationale}")
    print("\nKey Factors:")
    for factor in rec.key_factors:
        print(f"  - {factor}")

    print(f"\nSignals: Quant={rec.quant_signal} | LLM={rec.llm_signal}")
    print(f"Model: {rec.model_used} | Cost: ${rec.cost:.4f}")

    # Test without LLM (quant-only)
    print("\n" + "="*80)
    print("TEST 2: Quantitative Only")
    print("="*80)

    rec2 = await aggregator.get_recommendation(test_position, use_llm=False)

    print(f"\nACTION: {rec2.action.upper()}")
    print(f"Confidence: {rec2.confidence}%")
    print(f"Rationale: {rec2.rationale}")


if __name__ == "__main__":
    import asyncio
    asyncio.run(test_aggregator())

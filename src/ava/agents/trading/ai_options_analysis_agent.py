"""
AI-Powered Options Analysis Agent
=================================

Uses Claude to analyze options opportunities and provide
comprehensive strategy recommendations with Greeks analysis.

Author: AVA Trading Platform
Created: 2025-11-28
"""

import logging
from datetime import datetime
from typing import Dict, Any, List, Optional
from pydantic import Field

from src.ava.agents.base.llm_agent import (
    LLMAgent,
    AgentOutputBase,
    AgentConfidence,
    AgentExecutionContext
)

logger = logging.getLogger(__name__)


# =============================================================================
# OUTPUT MODELS
# =============================================================================

class GreeksAnalysis(AgentOutputBase):
    """Greeks breakdown for an option"""
    delta: float = 0.0
    gamma: float = 0.0
    theta: float = 0.0
    vega: float = 0.0
    rho: float = 0.0
    delta_interpretation: str = ""
    theta_interpretation: str = ""
    vega_interpretation: str = ""


class OptionLegAnalysis(AgentOutputBase):
    """Analysis of a single option leg"""
    strike: float = 0.0
    expiration: str = ""
    option_type: str = ""  # call, put
    direction: str = ""  # long, short
    premium: float = 0.0
    bid: float = 0.0
    ask: float = 0.0
    spread_percent: float = 0.0
    greeks: GreeksAnalysis = Field(default_factory=GreeksAnalysis)
    open_interest: int = 0
    volume: int = 0
    implied_volatility: float = 0.0


class RiskRewardProfile(AgentOutputBase):
    """Risk/reward analysis"""
    max_profit: float = 0.0
    max_loss: float = 0.0
    breakeven: float = 0.0
    breakeven_upper: Optional[float] = None
    probability_of_profit: float = 0.0
    risk_reward_ratio: float = 0.0
    return_on_risk: float = 0.0
    annualized_return: float = 0.0


class OptionsAnalysisOutput(AgentOutputBase):
    """Complete options analysis output"""
    symbol: str = ""
    underlying_price: float = 0.0
    strategy: str = ""  # csp, cc, iron_condor, etc.
    strategy_name: str = ""

    # Overall assessment
    opportunity_score: int = Field(default=50, ge=0, le=100)
    recommendation: str = ""  # strong_buy, buy, hold, avoid

    # IV Analysis
    iv_rank: float = 0.0
    iv_percentile: float = 0.0
    iv_environment: str = ""  # high, normal, low
    iv_signal: str = ""  # favor selling, favor buying, neutral

    # Strategy legs
    legs: List[OptionLegAnalysis] = Field(default_factory=list)

    # Risk/Reward
    risk_reward: RiskRewardProfile = Field(default_factory=RiskRewardProfile)

    # Greeks summary
    net_delta: float = 0.0
    net_theta: float = 0.0
    net_vega: float = 0.0
    theta_per_day: float = 0.0
    delta_exposure: str = ""  # bullish, bearish, neutral

    # Timing analysis
    days_to_expiration: int = 0
    optimal_dte_range: str = ""
    timing_assessment: str = ""

    # Liquidity
    liquidity_score: int = Field(default=50, ge=0, le=100)
    spread_quality: str = ""  # tight, acceptable, wide
    volume_assessment: str = ""

    # Entry/Exit guidance
    suggested_entry: float = 0.0
    suggested_exit_profit: float = 0.0
    suggested_exit_loss: float = 0.0
    adjustment_triggers: List[str] = Field(default_factory=list)

    # Warnings
    warnings: List[str] = Field(default_factory=list)
    catalysts_to_watch: List[str] = Field(default_factory=list)


# =============================================================================
# OPTIONS ANALYSIS AGENT
# =============================================================================

class AIOptionsAnalysisAgent(LLMAgent[OptionsAnalysisOutput]):
    """
    AI-powered options analysis agent.

    Provides comprehensive analysis of options opportunities
    including Greeks, risk/reward, and strategy optimization.

    Usage:
        agent = AIOptionsAnalysisAgent()
        result = await agent.execute({
            "symbol": "AAPL",
            "underlying_price": 185.50,
            "strategy": "csp",
            "option_chain": {...},
            "iv_data": {...}
        })
    """

    name = "ai_options_analysis"
    description = "Comprehensive options analysis with Greeks and strategy scoring"
    output_model = OptionsAnalysisOutput
    temperature = 0.3

    system_prompt = """You are an expert options analyst specializing in:

1. GREEKS ANALYSIS:
   - Delta: Directional exposure and probability approximation
   - Gamma: Rate of delta change (acceleration)
   - Theta: Time decay (your friend when selling)
   - Vega: Volatility sensitivity
   - Understanding Greek interactions

2. STRATEGY MECHANICS:
   - Cash-Secured Puts (CSP): Strike selection, delta targeting
   - Covered Calls (CC): Strike selection, call-away probability
   - Iron Condors: Wing placement, probability of profit
   - Spreads: Risk definition, max profit/loss calculation
   - Wheel Strategy: Entry timing, assignment management

3. IV ANALYSIS:
   - IV Rank interpretation (0-100 scale)
   - IV Percentile meaning
   - IV vs HV comparison
   - Volatility skew implications
   - Mean reversion expectations

4. RISK/REWARD ASSESSMENT:
   - Max profit/loss calculations
   - Breakeven analysis
   - Probability of profit estimation
   - Risk/reward ratios
   - Annualized return calculations

5. LIQUIDITY ANALYSIS:
   - Bid-ask spread evaluation
   - Open interest assessment
   - Volume patterns
   - Execution considerations

Your role is to:
1. Score the opportunity (0-100)
2. Analyze all relevant Greeks
3. Calculate risk/reward metrics
4. Assess IV environment suitability
5. Provide specific entry/exit guidance
6. Highlight any warnings or concerns

Be precise with numbers and calculations."""

    def build_prompt(self, input_data: Dict[str, Any]) -> str:
        """Build options analysis prompt"""
        symbol = input_data.get('symbol', 'UNKNOWN')
        price = input_data.get('underlying_price', 0)
        strategy = input_data.get('strategy', 'csp')
        option_chain = input_data.get('option_chain', {})
        iv_data = input_data.get('iv_data', {})
        target_delta = input_data.get('target_delta', 0.30)
        target_dte = input_data.get('target_dte', 30)

        strategy_names = {
            'csp': 'Cash-Secured Put',
            'cc': 'Covered Call',
            'ic': 'Iron Condor',
            'strangle': 'Short Strangle',
            'straddle': 'Short Straddle',
            'put_spread': 'Put Credit Spread',
            'call_spread': 'Call Credit Spread',
            'calendar': 'Calendar Spread',
            'diagonal': 'Diagonal Spread'
        }

        prompt = f"""## Options Analysis Request: {symbol}

### Underlying Data
- **Symbol**: {symbol}
- **Current Price**: ${price:.2f}
- **Strategy**: {strategy_names.get(strategy, strategy.upper())}
- **Target Delta**: {target_delta:.2f}
- **Target DTE**: {target_dte} days

### Implied Volatility Data
- **IV Rank**: {iv_data.get('iv_rank', 50):.1f}%
- **IV Percentile**: {iv_data.get('iv_percentile', 50):.1f}%
- **Current IV**: {iv_data.get('current_iv', 0.30):.1%}
- **Historical Vol (20d)**: {iv_data.get('hv_20', 0.25):.1%}
- **Historical Vol (60d)**: {iv_data.get('hv_60', 0.25):.1%}
- **IV Premium**: {iv_data.get('current_iv', 0.30) - iv_data.get('hv_20', 0.25):.1%}

### Option Chain Data
"""
        # Add relevant option chain data
        puts = option_chain.get('puts', [])
        calls = option_chain.get('calls', [])

        if puts:
            prompt += "\n**Puts (for CSP/Put Spreads):**\n"
            for put in puts[:7]:
                prompt += f"- Strike ${put.get('strike', 0):.0f}: "
                prompt += f"Bid ${put.get('bid', 0):.2f} / Ask ${put.get('ask', 0):.2f}, "
                prompt += f"Δ {put.get('delta', 0):.2f}, "
                prompt += f"θ ${put.get('theta', 0):.3f}, "
                prompt += f"IV {put.get('iv', 0):.1%}, "
                prompt += f"OI {put.get('open_interest', 0):,}\n"

        if calls:
            prompt += "\n**Calls (for CC/Call Spreads):**\n"
            for call in calls[:7]:
                prompt += f"- Strike ${call.get('strike', 0):.0f}: "
                prompt += f"Bid ${call.get('bid', 0):.2f} / Ask ${call.get('ask', 0):.2f}, "
                prompt += f"Δ {call.get('delta', 0):.2f}, "
                prompt += f"θ ${call.get('theta', 0):.3f}, "
                prompt += f"IV {call.get('iv', 0):.1%}, "
                prompt += f"OI {call.get('open_interest', 0):,}\n"

        # Earnings and events
        earnings_date = input_data.get('earnings_date')
        if earnings_date:
            prompt += f"\n**⚠️ Earnings Date**: {earnings_date}\n"

        prompt += f"""

### Analysis Tasks

1. **Opportunity Score (0-100)**:
   - 80-100: Excellent opportunity
   - 60-79: Good opportunity
   - 40-59: Average opportunity
   - 20-39: Below average
   - 0-19: Poor opportunity

2. **Optimal Strike Selection**:
   - For {strategy_names.get(strategy, strategy)}, which strike is best?
   - Consider delta target of ~{target_delta:.2f}
   - Balance premium vs probability

3. **Greeks Analysis**:
   - Net delta exposure interpretation
   - Daily theta capture (income)
   - Vega exposure (volatility risk)

4. **IV Environment Assessment**:
   - Is IV elevated (favor selling) or depressed (favor buying)?
   - Is there unusual skew?
   - IV crush potential after events?

5. **Risk/Reward Calculation**:
   - Max profit and max loss
   - Breakeven price
   - Probability of profit estimate
   - Annualized return on risk

6. **Liquidity Assessment**:
   - Bid-ask spread quality
   - Open interest adequacy
   - Expected fill quality

7. **Entry/Exit Guidance**:
   - Suggested entry price (mid, bid, etc.)
   - Profit target (50%, 75%, etc.)
   - Stop loss level
   - When to adjust

8. **Warnings**: Any red flags or concerns

Be specific with strike recommendations and numerical targets.
"""
        return prompt

    def parse_response(self, response: str, input_data: Dict[str, Any]) -> OptionsAnalysisOutput:
        """Parse options analysis response"""
        import json
        import re

        try:
            json_match = re.search(r'\{[\s\S]*\}', response)
            if json_match:
                data = json.loads(json_match.group())
                data['agent_name'] = self.name
                data['symbol'] = input_data.get('symbol', '')
                data['underlying_price'] = input_data.get('underlying_price', 0)
                data['strategy'] = input_data.get('strategy', 'csp')
                return OptionsAnalysisOutput(**data)
        except Exception as e:
            logger.warning(f"Failed to parse JSON: {e}")

        return self._parse_text_response(response, input_data)

    def _parse_text_response(
        self,
        response: str,
        input_data: Dict[str, Any]
    ) -> OptionsAnalysisOutput:
        """Parse unstructured response"""
        import re

        response_lower = response.lower()
        iv_data = input_data.get('iv_data', {})

        # Extract score
        score = 50
        score_match = re.search(r'score[:\s]*(\d+)', response_lower)
        if score_match:
            score = min(100, max(0, int(score_match.group(1))))
        elif 'excellent' in response_lower:
            score = 85
        elif 'good' in response_lower:
            score = 70
        elif 'average' in response_lower:
            score = 55
        elif 'poor' in response_lower:
            score = 25

        # Recommendation
        if score >= 70:
            recommendation = 'buy' if score >= 80 else 'buy'
        elif score >= 50:
            recommendation = 'hold'
        else:
            recommendation = 'avoid'

        # IV environment
        iv_rank = iv_data.get('iv_rank', 50)
        if iv_rank > 60:
            iv_environment = 'high'
            iv_signal = 'favor selling'
        elif iv_rank < 30:
            iv_environment = 'low'
            iv_signal = 'favor buying'
        else:
            iv_environment = 'normal'
            iv_signal = 'neutral'

        return OptionsAnalysisOutput(
            agent_name=self.name,
            symbol=input_data.get('symbol', ''),
            underlying_price=input_data.get('underlying_price', 0),
            strategy=input_data.get('strategy', 'csp'),
            opportunity_score=score,
            recommendation=recommendation,
            iv_rank=iv_rank,
            iv_percentile=iv_data.get('iv_percentile', 50),
            iv_environment=iv_environment,
            iv_signal=iv_signal,
            days_to_expiration=input_data.get('target_dte', 30),
            reasoning=response,
            confidence=AgentConfidence.MEDIUM
        )


# =============================================================================
# QUICK OPTIONS SCORER
# =============================================================================

class QuickOptionsScorer:
    """Rule-based quick options scoring"""

    @staticmethod
    def score(
        iv_rank: float = 50,
        delta: float = 0.30,
        dte: int = 30,
        bid_ask_spread_pct: float = 5.0,
        open_interest: int = 100,
        premium_vs_risk: float = 0.02,
        has_earnings: bool = False
    ) -> Dict:
        """Quick options opportunity assessment"""
        score = 50
        factors = []
        warnings = []

        # IV environment
        if iv_rank > 60:
            factors.append({"factor": "IV", "impact": "+15", "note": "High IV - premium selling favored"})
            score += 15
        elif iv_rank > 40:
            factors.append({"factor": "IV", "impact": "+5", "note": "Moderate IV"})
            score += 5
        else:
            factors.append({"factor": "IV", "impact": "-10", "note": "Low IV - limited premium"})
            score -= 10

        # Delta selection
        if 0.25 <= delta <= 0.35:
            factors.append({"factor": "Delta", "impact": "+10", "note": "Optimal delta range"})
            score += 10
        elif delta < 0.20:
            factors.append({"factor": "Delta", "impact": "-5", "note": "Very low delta - limited premium"})
            score -= 5
        elif delta > 0.40:
            factors.append({"factor": "Delta", "impact": "-5", "note": "High delta - higher assignment risk"})
            score -= 5

        # DTE
        if 30 <= dte <= 45:
            factors.append({"factor": "DTE", "impact": "+10", "note": "Optimal DTE for theta decay"})
            score += 10
        elif 21 <= dte <= 60:
            factors.append({"factor": "DTE", "impact": "+5", "note": "Acceptable DTE range"})
            score += 5
        else:
            factors.append({"factor": "DTE", "impact": "-5", "note": "Suboptimal DTE"})
            score -= 5

        # Liquidity
        if bid_ask_spread_pct < 3:
            factors.append({"factor": "Spread", "impact": "+10", "note": "Tight bid-ask spread"})
            score += 10
        elif bid_ask_spread_pct < 5:
            factors.append({"factor": "Spread", "impact": "+5", "note": "Acceptable spread"})
            score += 5
        else:
            factors.append({"factor": "Spread", "impact": "-10", "note": "Wide bid-ask spread"})
            warnings.append("Wide bid-ask spread may impact fills")
            score -= 10

        # Open interest
        if open_interest > 500:
            factors.append({"factor": "OI", "impact": "+5", "note": "Good open interest"})
            score += 5
        elif open_interest < 100:
            factors.append({"factor": "OI", "impact": "-5", "note": "Low open interest"})
            warnings.append("Low open interest - limited liquidity")
            score -= 5

        # Premium quality
        if premium_vs_risk > 0.03:
            factors.append({"factor": "Premium", "impact": "+10", "note": "Excellent premium/risk ratio"})
            score += 10
        elif premium_vs_risk > 0.02:
            factors.append({"factor": "Premium", "impact": "+5", "note": "Good premium/risk ratio"})
            score += 5
        elif premium_vs_risk < 0.01:
            factors.append({"factor": "Premium", "impact": "-10", "note": "Poor premium/risk ratio"})
            score -= 10

        # Earnings warning
        if has_earnings:
            warnings.append("⚠️ Earnings within expiration - elevated risk")
            score -= 15

        score = min(100, max(0, score))

        # Recommendation
        if score >= 75:
            recommendation = 'strong_buy'
        elif score >= 60:
            recommendation = 'buy'
        elif score >= 45:
            recommendation = 'hold'
        else:
            recommendation = 'avoid'

        return {
            'opportunity_score': score,
            'recommendation': recommendation,
            'factors': factors,
            'warnings': warnings
        }


# =============================================================================
# TESTING
# =============================================================================

if __name__ == "__main__":
    import asyncio

    print("\n=== Testing AI Options Analysis Agent ===\n")

    async def test_agent():
        # Test quick scorer
        print("1. Testing Quick Options Scorer...")
        scorer = QuickOptionsScorer()

        test_cases = [
            {"iv_rank": 75, "delta": 0.30, "dte": 35, "bid_ask_spread_pct": 2, "open_interest": 1000, "premium_vs_risk": 0.035},
            {"iv_rank": 25, "delta": 0.15, "dte": 7, "bid_ask_spread_pct": 8, "open_interest": 50, "premium_vs_risk": 0.01},
            {"iv_rank": 50, "delta": 0.30, "dte": 30, "bid_ask_spread_pct": 4, "open_interest": 300, "premium_vs_risk": 0.025},
        ]

        for case in test_cases:
            result = scorer.score(**case)
            print(f"\n   IV={case['iv_rank']}, Δ={case['delta']}, DTE={case['dte']}:")
            print(f"      Score: {result['opportunity_score']}")
            print(f"      Recommendation: {result['recommendation']}")
            if result['warnings']:
                print(f"      Warnings: {result['warnings']}")

        # Test agent structure
        print("\n2. Testing Agent Structure...")
        agent = AIOptionsAnalysisAgent()
        print(f"   Name: {agent.name}")
        print(f"   Output model: {agent.output_model.__name__}")

        print("\n✅ AI Options Analysis Agent ready!")

    asyncio.run(test_agent())

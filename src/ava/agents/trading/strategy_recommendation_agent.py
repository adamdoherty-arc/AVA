"""
AI-Powered Strategy Recommendation Agent
=========================================

Uses Claude to analyze market conditions and recommend
optimal options strategies with detailed reasoning.

Author: AVA Trading Platform
Created: 2025-11-28
"""

import logging
from datetime import datetime, date
from typing import Dict, Any, List, Optional
from pydantic import Field

from src.ava.agents.base.llm_agent import (
    LLMAgent,
    AgentOutputBase,
    AgentConfidence,
    AgentExecutionContext
)
from src.ava.core.config import get_config

logger = logging.getLogger(__name__)


# =============================================================================
# OUTPUT MODELS
# =============================================================================

class StrategyRecommendation(AgentOutputBase):
    """Single strategy recommendation"""
    strategy_name: str = ""
    strategy_type: str = ""  # income, directional, volatility, neutral
    fit_score: int = Field(default=0, ge=0, le=100)

    # Parameters
    suggested_delta: float = 0.30
    suggested_dte_min: int = 21
    suggested_dte_max: int = 45
    suggested_iv_rank_min: float = 0
    suggested_iv_rank_max: float = 100

    # Analysis
    why_recommended: str = ""
    market_conditions_fit: str = ""
    expected_outcome: str = ""
    key_risks: List[str] = Field(default_factory=list)

    # Position sizing
    suggested_allocation_pct: float = Field(default=2.0, ge=0.5, le=10)


class StrategyRecommendationOutput(AgentOutputBase):
    """Full strategy recommendation output"""
    symbol: str = ""
    underlying_price: float = 0
    iv_rank: float = 50
    market_regime: str = "neutral"

    # Top recommendations
    recommendations: List[StrategyRecommendation] = Field(default_factory=list)

    # Market analysis
    technical_outlook: str = ""
    fundamental_outlook: str = ""
    volatility_outlook: str = ""

    # Warnings
    warnings: List[str] = Field(default_factory=list)
    avoid_strategies: List[str] = Field(default_factory=list)


# =============================================================================
# STRATEGY RECOMMENDATION AGENT
# =============================================================================

class StrategyRecommendationAgent(LLMAgent[StrategyRecommendationOutput]):
    """
    AI-powered strategy recommendation agent.

    Analyzes market conditions and recommends optimal options strategies.

    Usage:
        agent = StrategyRecommendationAgent()
        result = await agent.execute({
            "symbol": "AAPL",
            "underlying_price": 230.50,
            "iv_rank": 45,
            "account_size": 100000,
            "risk_tolerance": "moderate"
        })
    """

    name = "strategy_recommendation"
    description = "Analyzes market conditions and recommends optimal options strategies"
    output_model = StrategyRecommendationOutput
    temperature = 0.4

    system_prompt = """You are an expert options strategist with deep knowledge of:

1. OPTIONS STRATEGIES:
   - Income strategies: Cash-secured puts, covered calls, wheel strategy
   - Neutral strategies: Iron condors, iron butterflies, strangles
   - Directional strategies: Vertical spreads (bull put, bear call, bull call, bear put)
   - Volatility strategies: Long/short straddles, strangles, calendar spreads
   - Advanced: Diagonal spreads, ratio spreads, 0DTE strategies

2. MARKET CONDITIONS:
   - IV Rank interpretation (high IV = sell premium, low IV = buy premium)
   - Trend analysis (bullish/bearish/neutral)
   - Volatility regimes (low vol, normal, high vol, crisis)
   - Earnings and event risk

3. RISK MANAGEMENT:
   - Position sizing based on account size
   - Delta exposure management
   - Probability of profit optimization
   - Max loss containment

Your role is to recommend the TOP 3 most suitable strategies for the current conditions.
Always provide detailed reasoning for each recommendation.
Be specific about entry parameters (delta, DTE, strikes).
Warn about any strategies that should be avoided."""

    def build_prompt(self, input_data: Dict[str, Any]) -> str:
        """Build analysis prompt"""
        config = get_config()

        symbol = input_data.get('symbol', 'UNKNOWN')
        price = input_data.get('underlying_price', 0)
        iv_rank = input_data.get('iv_rank', 50)
        iv_percentile = input_data.get('iv_percentile', iv_rank)
        hv_20 = input_data.get('hv_20', 0.25)
        vix = input_data.get('vix', 15)
        trend = input_data.get('trend', 'neutral')
        days_to_earnings = input_data.get('days_to_earnings', None)
        sector = input_data.get('sector', 'Unknown')
        account_size = input_data.get('account_size', 100000)
        risk_tolerance = input_data.get('risk_tolerance', 'moderate')

        # Technical indicators if provided
        rsi = input_data.get('rsi', None)
        sma_20 = input_data.get('sma_20', None)
        sma_50 = input_data.get('sma_50', None)

        prompt = f"""## Market Analysis Request

### Symbol Information
- **Symbol**: {symbol}
- **Current Price**: ${price:.2f}
- **Sector**: {sector}

### Volatility Analysis
- **IV Rank**: {iv_rank:.1f}% ({"HIGH - favor selling premium" if iv_rank > 50 else "LOW - favor buying premium"})
- **IV Percentile**: {iv_percentile:.1f}%
- **Historical Volatility (20d)**: {hv_20:.1%}
- **VIX**: {vix:.1f}

### Market Conditions
- **Trend**: {trend.upper()}
- **Days to Earnings**: {days_to_earnings if days_to_earnings else "No upcoming earnings"}
"""

        if rsi:
            prompt += f"- **RSI**: {rsi:.1f}\n"
        if sma_20 and sma_50:
            prompt += f"- **SMA20**: ${sma_20:.2f}, **SMA50**: ${sma_50:.2f}\n"

        prompt += f"""
### Account Parameters
- **Account Size**: ${account_size:,.0f}
- **Risk Tolerance**: {risk_tolerance.upper()}
- **Max Position Size**: {config.risk.max_position_size_pct}% of account

### Available Strategies to Consider
1. **Income/Theta Strategies** (best when IV Rank > 40):
   - Cash-Secured Put (CSP)
   - Covered Call
   - Wheel Strategy (CSP + CC rotation)
   - Iron Condor
   - Short Strangle

2. **Directional Strategies** (for trending markets):
   - Bull Put Spread (bullish)
   - Bear Call Spread (bearish)
   - Bull Call Spread (bullish, debit)
   - Bear Put Spread (bearish, debit)

3. **Volatility Strategies** (for expected vol changes):
   - Long Straddle (expect big move, low IV)
   - Long Strangle (cheaper volatility bet)
   - Calendar Spread (time decay play)
   - Diagonal Spread (directional + time)

4. **Advanced/0DTE** (for experienced traders only):
   - 0DTE Iron Condor (SPY/QQQ only)
   - 0DTE Credit Spreads
   - Gamma Scalping

### Your Task
Recommend the **TOP 3** most suitable strategies for this situation.

For each strategy, provide:
1. Strategy name and type
2. Fit score (0-100)
3. Specific entry parameters:
   - Target delta for short strikes
   - DTE range
   - IV rank requirements
4. Why this strategy fits current conditions
5. Expected outcome
6. Key risks to monitor
7. Suggested allocation (% of account)

Also provide:
- Overall market regime assessment
- Technical outlook summary
- Strategies to AVOID and why
- Any warnings (earnings, ex-dividend, etc.)
"""

        return prompt

    def parse_response(self, response: str, input_data: Dict[str, Any]) -> StrategyRecommendationOutput:
        """Parse LLM response with fallback logic"""
        import json
        import re

        try:
            # Try to extract JSON
            json_match = re.search(r'\{[\s\S]*\}', response)
            if json_match:
                data = json.loads(json_match.group())
                data['agent_name'] = self.name
                data['symbol'] = input_data.get('symbol', '')
                data['underlying_price'] = input_data.get('underlying_price', 0)
                data['iv_rank'] = input_data.get('iv_rank', 50)
                return StrategyRecommendationOutput(**data)

        except Exception as e:
            logger.warning(f"Failed to parse JSON response: {e}")

        # Fallback: Extract recommendations from text
        return self._parse_text_response(response, input_data)

    def _parse_text_response(
        self,
        response: str,
        input_data: Dict[str, Any]
    ) -> StrategyRecommendationOutput:
        """Parse unstructured text response"""
        recommendations = []

        # Strategy patterns to look for
        strategy_patterns = [
            ("cash-secured put", "income", 0.30),
            ("covered call", "income", 0.30),
            ("wheel", "income", 0.30),
            ("iron condor", "neutral", 0.16),
            ("bull put spread", "directional", 0.30),
            ("bear call spread", "directional", 0.30),
            ("bull call spread", "directional", 0.40),
            ("bear put spread", "directional", 0.40),
            ("long straddle", "volatility", 0.50),
            ("long strangle", "volatility", 0.30),
            ("calendar spread", "volatility", 0.50),
            ("0dte", "advanced", 0.10),
        ]

        response_lower = response.lower()
        for pattern, strategy_type, default_delta in strategy_patterns:
            if pattern in response_lower:
                recommendations.append(StrategyRecommendation(
                    agent_name=self.name,
                    strategy_name=pattern.title(),
                    strategy_type=strategy_type,
                    fit_score=70,
                    suggested_delta=default_delta,
                    why_recommended="Identified from analysis",
                    confidence=AgentConfidence.MEDIUM
                ))

        # Determine market regime
        iv_rank = input_data.get('iv_rank', 50)
        trend = input_data.get('trend', 'neutral')

        if iv_rank > 60:
            market_regime = "high_volatility"
        elif iv_rank < 30:
            market_regime = "low_volatility"
        else:
            market_regime = trend

        return StrategyRecommendationOutput(
            agent_name=self.name,
            symbol=input_data.get('symbol', ''),
            underlying_price=input_data.get('underlying_price', 0),
            iv_rank=iv_rank,
            market_regime=market_regime,
            recommendations=recommendations[:3],
            reasoning=response,
            confidence=AgentConfidence.MEDIUM
        )


# =============================================================================
# QUICK STRATEGY SELECTOR
# =============================================================================

class QuickStrategySelector:
    """
    Rule-based quick strategy selector for when LLM is unavailable.
    Uses simple heuristics based on IV rank and trend.
    """

    @staticmethod
    def select(
        iv_rank: float,
        trend: str = "neutral",
        days_to_earnings: Optional[int] = None,
        risk_tolerance: str = "moderate"
    ) -> List[Dict]:
        """
        Select strategies based on simple rules.

        Returns list of strategy suggestions.
        """
        strategies = []

        # Earnings warning
        if days_to_earnings and days_to_earnings <= 14:
            return [{
                "strategy": "Wait/Reduce",
                "reason": f"Earnings in {days_to_earnings} days - avoid new positions"
            }]

        # High IV environment (> 50)
        if iv_rank > 50:
            if trend == "bullish":
                strategies.append({
                    "strategy": "Bull Put Spread",
                    "delta": 0.30,
                    "reason": "High IV + bullish = sell put premium"
                })
                strategies.append({
                    "strategy": "Cash-Secured Put",
                    "delta": 0.30,
                    "reason": "High IV + bullish = wheel entry"
                })
            elif trend == "bearish":
                strategies.append({
                    "strategy": "Bear Call Spread",
                    "delta": 0.30,
                    "reason": "High IV + bearish = sell call premium"
                })
            else:  # neutral
                strategies.append({
                    "strategy": "Iron Condor",
                    "delta": 0.16,
                    "reason": "High IV + neutral = collect premium both sides"
                })
                strategies.append({
                    "strategy": "Short Strangle",
                    "delta": 0.16,
                    "reason": "High IV + neutral = undefined risk premium"
                })

        # Low IV environment (< 30)
        elif iv_rank < 30:
            if trend == "bullish":
                strategies.append({
                    "strategy": "Bull Call Spread",
                    "delta": 0.40,
                    "reason": "Low IV + bullish = buy cheap calls"
                })
            elif trend == "bearish":
                strategies.append({
                    "strategy": "Bear Put Spread",
                    "delta": 0.40,
                    "reason": "Low IV + bearish = buy cheap puts"
                })
            else:  # neutral
                strategies.append({
                    "strategy": "Calendar Spread",
                    "delta": 0.50,
                    "reason": "Low IV + neutral = bet on IV increase"
                })
                strategies.append({
                    "strategy": "Long Straddle",
                    "delta": 0.50,
                    "reason": "Low IV = cheap options for big move"
                })

        # Medium IV (30-50)
        else:
            if trend == "bullish":
                strategies.append({
                    "strategy": "Bull Put Spread",
                    "delta": 0.25,
                    "reason": "Moderate IV + bullish = balanced approach"
                })
            elif trend == "bearish":
                strategies.append({
                    "strategy": "Bear Call Spread",
                    "delta": 0.25,
                    "reason": "Moderate IV + bearish = balanced approach"
                })
            else:
                strategies.append({
                    "strategy": "Iron Condor",
                    "delta": 0.20,
                    "reason": "Moderate IV + neutral = standard IC"
                })

        return strategies[:3]


# =============================================================================
# TESTING
# =============================================================================

if __name__ == "__main__":
    import asyncio

    print("\n=== Testing Strategy Recommendation Agent ===\n")

    async def test_agent():
        # Test quick selector first (no API needed)
        print("1. Testing Quick Strategy Selector...")
        quick = QuickStrategySelector()

        test_cases = [
            {"iv_rank": 65, "trend": "bullish"},
            {"iv_rank": 25, "trend": "neutral"},
            {"iv_rank": 45, "trend": "bearish"},
            {"iv_rank": 50, "trend": "neutral", "days_to_earnings": 7},
        ]

        for case in test_cases:
            strategies = quick.select(**case)
            print(f"\n   IV={case['iv_rank']}, Trend={case['trend']}:")
            for s in strategies:
                print(f"      - {s['strategy']}: {s['reason']}")

        # Test agent structure
        print("\n2. Testing Agent Structure...")
        agent = StrategyRecommendationAgent()
        print(f"   Name: {agent.name}")
        print(f"   Output model: {agent.output_model.__name__}")

        # Test prompt building
        print("\n3. Testing Prompt Building...")
        prompt = agent.build_prompt({
            "symbol": "AAPL",
            "underlying_price": 230.50,
            "iv_rank": 45,
            "trend": "bullish",
            "account_size": 100000
        })
        print(f"   Prompt length: {len(prompt)} chars")
        print(f"   First 200 chars: {prompt[:200]}...")

        print("\n✅ Strategy recommendation agent ready!")

    asyncio.run(test_agent())

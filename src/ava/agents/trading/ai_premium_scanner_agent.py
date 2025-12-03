"""
AI-Powered Premium Scanner Agent
================================

Uses Claude to analyze and score premium-selling opportunities
across the options market.

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

class PremiumOpportunity(AgentOutputBase):
    """Individual premium opportunity"""
    symbol: str = ""
    underlying_price: float = 0.0

    # Option details
    strategy: str = ""  # csp, cc, ic, strangle
    strike: float = 0.0
    strike_2: Optional[float] = None  # For spreads
    expiration: str = ""
    days_to_expiration: int = 0

    # Premium metrics
    premium: float = 0.0
    premium_per_day: float = 0.0
    annualized_return: float = 0.0
    return_on_risk: float = 0.0

    # Greeks
    delta: float = 0.0
    theta: float = 0.0
    iv: float = 0.0
    iv_rank: float = 0.0

    # Quality metrics
    opportunity_score: int = Field(default=50, ge=0, le=100)
    liquidity_score: int = Field(default=50, ge=0, le=100)

    # Risk factors
    distance_to_strike_pct: float = 0.0
    probability_of_profit: float = 0.0
    max_loss: float = 0.0

    # Warnings
    warnings: List[str] = Field(default_factory=list)
    has_earnings: bool = False


class PremiumScanOutput(AgentOutputBase):
    """Complete premium scan output"""
    scan_timestamp: str = ""
    scan_criteria: Dict[str, Any] = Field(default_factory=dict)

    # Results summary
    total_scanned: int = 0
    opportunities_found: int = 0

    # Market context
    market_iv_environment: str = ""  # elevated, normal, depressed
    vix_level: float = 0.0
    sector_highlights: List[str] = Field(default_factory=list)

    # Opportunities by category
    top_csp_opportunities: List[PremiumOpportunity] = Field(default_factory=list)
    top_cc_opportunities: List[PremiumOpportunity] = Field(default_factory=list)
    top_ic_opportunities: List[PremiumOpportunity] = Field(default_factory=list)

    # Overall recommendations
    best_overall: List[PremiumOpportunity] = Field(default_factory=list)

    # Risk warnings
    market_warnings: List[str] = Field(default_factory=list)
    earnings_this_week: List[str] = Field(default_factory=list)


# =============================================================================
# PREMIUM SCANNER AGENT
# =============================================================================

class AIPremiumScannerAgent(LLMAgent[PremiumScanOutput]):
    """
    AI-powered premium scanner agent.

    Scans the market for optimal premium-selling opportunities
    and ranks them based on multiple factors.

    Usage:
        agent = AIPremiumScannerAgent()
        result = await agent.execute({
            "watchlist": ["AAPL", "MSFT", "NVDA"],
            "min_premium": 1.0,
            "max_delta": 0.35,
            "min_iv_rank": 30,
            "target_dte": 30
        })
    """

    name = "ai_premium_scanner"
    description = "Scans market for optimal premium-selling opportunities"
    output_model = PremiumScanOutput
    temperature = 0.3

    system_prompt = """You are an expert options premium analyst specializing in:

1. PREMIUM SELLING STRATEGIES:
   - Cash-Secured Puts (CSP): Income generation, wheel entry
   - Covered Calls (CC): Income on existing positions
   - Iron Condors (IC): Neutral premium collection
   - Strangles: Undefined risk premium strategies

2. OPPORTUNITY SCORING:
   - Premium yield (annualized return)
   - IV rank and percentile
   - Probability of profit
   - Risk/reward ratio
   - Liquidity quality

3. MARKET CONTEXT:
   - VIX environment assessment
   - Sector IV analysis
   - Earnings calendar awareness
   - Market regime (trending vs range-bound)

4. RISK FACTORS:
   - Assignment probability
   - Max loss scenarios
   - Earnings event exposure
   - Liquidity concerns

Your role is to:
1. Analyze submitted opportunities
2. Score each opportunity (0-100)
3. Rank the best opportunities
4. Highlight any concerns or warnings
5. Consider market context in recommendations

Focus on quality over quantity - better to have fewer excellent opportunities
than many mediocre ones."""

    def build_prompt(self, input_data: Dict[str, Any]) -> str:
        """Build premium scan prompt"""
        opportunities = input_data.get('opportunities', [])
        market_data = input_data.get('market_data', {})
        criteria = input_data.get('criteria', {})

        prompt = f"""## Premium Scan Analysis Request

### Scan Criteria
- **Minimum Premium**: ${criteria.get('min_premium', 1.0):.2f}
- **Maximum Delta**: {criteria.get('max_delta', 0.35):.2f}
- **Minimum IV Rank**: {criteria.get('min_iv_rank', 30)}%
- **Target DTE**: {criteria.get('min_dte', 21)}-{criteria.get('max_dte', 45)} days

### Market Context
- **VIX**: {market_data.get('vix', 15):.1f}
- **Market Trend**: {market_data.get('trend', 'neutral')}
- **Overall IV Environment**: {market_data.get('iv_environment', 'normal')}

### Opportunities to Analyze

"""
        for i, opp in enumerate(opportunities[:20], 1):
            prompt += f"""
**{i}. {opp.get('symbol', 'N/A')}** - ${opp.get('underlying_price', 0):.2f}
   - Strategy: {opp.get('strategy', 'csp').upper()}
   - Strike: ${opp.get('strike', 0):.0f} ({opp.get('distance_pct', 0):.1f}% OTM)
   - DTE: {opp.get('dte', 0)} days
   - Premium: ${opp.get('premium', 0):.2f} (${opp.get('premium_per_day', 0):.3f}/day)
   - Delta: {opp.get('delta', 0):.2f}
   - IV Rank: {opp.get('iv_rank', 0):.0f}%
   - Annualized: {opp.get('annualized_return', 0):.1%}
   - Bid-Ask: ${opp.get('bid', 0):.2f} / ${opp.get('ask', 0):.2f}
   - Open Interest: {opp.get('open_interest', 0):,}
   - Earnings: {'⚠️ YES' if opp.get('has_earnings', False) else 'No'}
"""

        prompt += """

### Analysis Tasks

1. **Score Each Opportunity (0-100)**:
   - Premium quality (annualized return)
   - IV environment favorability
   - Delta/probability assessment
   - Liquidity quality
   - Risk factors

2. **Identify Top Opportunities**:
   - Best CSP opportunities
   - Best CC opportunities
   - Best overall (any strategy)

3. **Risk Assessment**:
   - Flag any earnings concerns
   - Note liquidity issues
   - Identify elevated risk situations

4. **Market Context Impact**:
   - How does current VIX affect these trades?
   - Any sector-specific considerations?

5. **Final Recommendations**:
   - Which 3-5 opportunities are best?
   - What makes them stand out?
   - Any opportunities to avoid?

Rank opportunities and provide clear reasoning.
"""
        return prompt

    def parse_response(self, response: str, input_data: Dict[str, Any]) -> PremiumScanOutput:
        """Parse premium scan response"""
        import json
        import re

        try:
            json_match = re.search(r'\{[\s\S]*\}', response)
            if json_match:
                data = json.loads(json_match.group())
                data['agent_name'] = self.name
                return PremiumScanOutput(**data)
        except Exception as e:
            logger.warning(f"Failed to parse JSON: {e}")

        return self._parse_text_response(response, input_data)

    def _parse_text_response(
        self,
        response: str,
        input_data: Dict[str, Any]
    ) -> PremiumScanOutput:
        """Parse unstructured response"""
        opportunities = input_data.get('opportunities', [])
        market_data = input_data.get('market_data', {})

        # Score opportunities based on response mentions
        scored_opps = []
        for opp in opportunities:
            symbol = opp.get('symbol', '')
            score = self._calculate_opportunity_score(opp)

            # Boost if mentioned positively in response
            if symbol.lower() in response.lower():
                if any(word in response.lower() for word in ['excellent', 'great', 'strong', 'top']):
                    score = min(100, score + 10)

            scored_opps.append(PremiumOpportunity(
                symbol=symbol,
                underlying_price=opp.get('underlying_price', 0),
                strategy=opp.get('strategy', 'csp'),
                strike=opp.get('strike', 0),
                expiration=opp.get('expiration', ''),
                days_to_expiration=opp.get('dte', 0),
                premium=opp.get('premium', 0),
                premium_per_day=opp.get('premium_per_day', 0),
                annualized_return=opp.get('annualized_return', 0),
                delta=opp.get('delta', 0),
                theta=opp.get('theta', 0),
                iv=opp.get('iv', 0),
                iv_rank=opp.get('iv_rank', 0),
                opportunity_score=score,
                distance_to_strike_pct=opp.get('distance_pct', 0),
                has_earnings=opp.get('has_earnings', False),
                warnings=['Earnings within expiration'] if opp.get('has_earnings', False) else []
            ))

        # Sort by score
        scored_opps.sort(key=lambda x: x.opportunity_score, reverse=True)

        # Separate by strategy
        csp_opps = [o for o in scored_opps if o.strategy == 'csp'][:5]
        cc_opps = [o for o in scored_opps if o.strategy == 'cc'][:5]
        ic_opps = [o for o in scored_opps if o.strategy == 'ic'][:5]

        # IV environment
        vix = market_data.get('vix', 15)
        if vix > 25:
            iv_env = 'elevated'
        elif vix < 15:
            iv_env = 'depressed'
        else:
            iv_env = 'normal'

        return PremiumScanOutput(
            agent_name=self.name,
            scan_timestamp=datetime.now().isoformat(),
            scan_criteria=input_data.get('criteria', {}),
            total_scanned=len(opportunities),
            opportunities_found=len([o for o in scored_opps if o.opportunity_score >= 60]),
            market_iv_environment=iv_env,
            vix_level=vix,
            top_csp_opportunities=csp_opps,
            top_cc_opportunities=cc_opps,
            top_ic_opportunities=ic_opps,
            best_overall=scored_opps[:5],
            reasoning=response,
            confidence=AgentConfidence.MEDIUM
        )

    def _calculate_opportunity_score(self, opp: Dict) -> int:
        """Calculate base opportunity score"""
        score = 50

        # IV rank contribution
        iv_rank = opp.get('iv_rank', 50)
        if iv_rank > 60:
            score += 15
        elif iv_rank > 40:
            score += 5
        else:
            score -= 10

        # Delta (probability)
        delta = abs(opp.get('delta', 0.30))
        if 0.25 <= delta <= 0.35:
            score += 10
        elif delta < 0.20 or delta > 0.40:
            score -= 5

        # Premium yield
        annualized = opp.get('annualized_return', 0)
        if annualized > 0.50:  # >50% annualized
            score += 15
        elif annualized > 0.30:
            score += 10
        elif annualized > 0.20:
            score += 5
        elif annualized < 0.10:
            score -= 10

        # DTE
        dte = opp.get('dte', 30)
        if 30 <= dte <= 45:
            score += 5
        elif dte < 14 or dte > 60:
            score -= 5

        # Liquidity
        oi = opp.get('open_interest', 0)
        if oi > 500:
            score += 5
        elif oi < 100:
            score -= 10

        # Earnings penalty
        if opp.get('has_earnings', False):
            score -= 20

        return min(100, max(0, score))


# =============================================================================
# QUICK PREMIUM SCANNER
# =============================================================================

class QuickPremiumScanner:
    """Rule-based quick premium scanner"""

    @staticmethod
    def score_opportunity(
        iv_rank: float = 50,
        delta: float = 0.30,
        dte: int = 30,
        annualized_return: float = 0.25,
        open_interest: int = 100,
        has_earnings: bool = False
    ) -> Dict:
        """Quick opportunity scoring"""
        score = 50
        reasons = []

        # IV rank
        if iv_rank > 60:
            score += 15
            reasons.append(f"High IV Rank ({iv_rank:.0f}%)")
        elif iv_rank < 30:
            score -= 10
            reasons.append(f"Low IV Rank ({iv_rank:.0f}%)")

        # Delta
        if 0.25 <= delta <= 0.35:
            score += 10
            reasons.append(f"Optimal delta ({delta:.2f})")

        # DTE
        if 30 <= dte <= 45:
            score += 5
            reasons.append("Optimal DTE range")

        # Annualized return
        if annualized_return > 0.40:
            score += 15
            reasons.append(f"Excellent return ({annualized_return:.0%})")
        elif annualized_return > 0.25:
            score += 10
            reasons.append(f"Good return ({annualized_return:.0%})")
        elif annualized_return < 0.15:
            score -= 10
            reasons.append(f"Low return ({annualized_return:.0%})")

        # Liquidity
        if open_interest > 500:
            score += 5
        elif open_interest < 100:
            score -= 10
            reasons.append("Low liquidity")

        # Earnings
        if has_earnings:
            score -= 20
            reasons.append("⚠️ Earnings risk")

        score = min(100, max(0, score))

        if score >= 75:
            rating = "excellent"
        elif score >= 60:
            rating = "good"
        elif score >= 45:
            rating = "average"
        else:
            rating = "poor"

        return {
            'score': score,
            'rating': rating,
            'reasons': reasons
        }

    @staticmethod
    def filter_opportunities(
        opportunities: List[Dict],
        min_iv_rank: float = 30,
        max_delta: float = 0.35,
        min_annualized: float = 0.20,
        min_open_interest: int = 100,
        exclude_earnings: bool = True
    ) -> List[Dict]:
        """Filter opportunities based on criteria"""
        filtered = []

        for opp in opportunities:
            # Apply filters
            if opp.get('iv_rank', 0) < min_iv_rank:
                continue
            if abs(opp.get('delta', 0)) > max_delta:
                continue
            if opp.get('annualized_return', 0) < min_annualized:
                continue
            if opp.get('open_interest', 0) < min_open_interest:
                continue
            if exclude_earnings and opp.get('has_earnings', False):
                continue

            # Score it
            score_result = QuickPremiumScanner.score_opportunity(
                iv_rank=opp.get('iv_rank', 50),
                delta=abs(opp.get('delta', 0.30)),
                dte=opp.get('dte', 30),
                annualized_return=opp.get('annualized_return', 0.25),
                open_interest=opp.get('open_interest', 100),
                has_earnings=opp.get('has_earnings', False)
            )
            opp['opportunity_score'] = score_result['score']
            opp['rating'] = score_result['rating']
            filtered.append(opp)

        # Sort by score
        filtered.sort(key=lambda x: x['opportunity_score'], reverse=True)
        return filtered


# =============================================================================
# TESTING
# =============================================================================

if __name__ == "__main__":
    import asyncio

    print("\n=== Testing AI Premium Scanner Agent ===\n")

    async def test_agent():
        # Test quick scanner
        print("1. Testing Quick Premium Scanner...")
        scanner = QuickPremiumScanner()

        test_opps = [
            {"iv_rank": 75, "delta": 0.28, "dte": 35, "annualized_return": 0.45, "open_interest": 1200, "has_earnings": False},
            {"iv_rank": 25, "delta": 0.40, "dte": 14, "annualized_return": 0.15, "open_interest": 50, "has_earnings": True},
            {"iv_rank": 55, "delta": 0.30, "dte": 30, "annualized_return": 0.30, "open_interest": 500, "has_earnings": False},
        ]

        for opp in test_opps:
            result = scanner.score_opportunity(**opp)
            print(f"\n   IV={opp['iv_rank']}, Δ={opp['delta']}, Ret={opp['annualized_return']:.0%}:")
            print(f"      Score: {result['score']}")
            print(f"      Rating: {result['rating']}")
            print(f"      Reasons: {result['reasons']}")

        # Test filtering
        print("\n2. Testing Opportunity Filtering...")
        filtered = scanner.filter_opportunities(
            [
                {"symbol": "AAPL", "iv_rank": 65, "delta": 0.28, "dte": 35, "annualized_return": 0.35, "open_interest": 1000, "has_earnings": False},
                {"symbol": "MSFT", "iv_rank": 20, "delta": 0.30, "dte": 30, "annualized_return": 0.20, "open_interest": 800, "has_earnings": False},
                {"symbol": "NVDA", "iv_rank": 80, "delta": 0.45, "dte": 28, "annualized_return": 0.50, "open_interest": 2000, "has_earnings": True},
            ],
            min_iv_rank=30
        )
        print(f"   Filtered: {[f['symbol'] for f in filtered]}")

        # Test agent structure
        print("\n3. Testing Agent Structure...")
        agent = AIPremiumScannerAgent()
        print(f"   Name: {agent.name}")
        print(f"   Output model: {agent.output_model.__name__}")

        print("\n✅ AI Premium Scanner Agent ready!")

    asyncio.run(test_agent())

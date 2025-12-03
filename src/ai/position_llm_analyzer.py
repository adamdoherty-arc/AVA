"""
LLM-Based Position Recommendation Engine
Uses AI models to generate contextual position recommendations
"""

import logging
import json
import os
import asyncio
from typing import Dict, Optional
from datetime import datetime
from dotenv import load_dotenv

from src.services.llm_service import LLMService
from src.ai.position_data_aggregator import EnrichedPosition

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PositionLLMAnalyzer:
    """
    Generate AI-powered recommendations for option positions

    Features:
    - Contextual analysis using LLMs
    - Auto-selects FREE models: Groq > DeepSeek > Gemini
    - Automatic fallback if provider fails
    - Built-in caching and cost tracking
    - Response caching via LLMService
    """

    # Model selection tiers - Uses existing LLM Service with FREE models
    # Priority: Ollama (free) > Groq (free) > DeepSeek (cheap) > Gemini (free)
    ANALYSIS_TIERS = {
        'critical': {  # Losing positions, high risk
            'provider': None,  # Auto-select: groq > deepseek > gemini
            'temperature': 0.3,
            'max_tokens': 800
        },
        'standard': {  # Routine analysis
            'provider': None,  # Auto-select: groq > deepseek > gemini
            'temperature': 0.4,
            'max_tokens': 500
        },
        'bulk': {  # Batch processing
            'provider': None,  # Auto-select: groq > deepseek > gemini
            'temperature': 0.5,
            'max_tokens': 400
        }
    }

    def __init__(self) -> None:
        """Initialize LLM analyzer with existing LLM service"""
        self.llm_service = LLMService()

    async def analyze_position(
        self,
        position: EnrichedPosition,
        quant_analysis: Optional[Dict] = None,
        market_context: Optional[Dict] = None,
        force_model: Optional[str] = None
    ) -> Dict:
        """
        Generate AI recommendation for a position

        Args:
            position: Enriched position data
            quant_analysis: Optional quantitative analysis results
            market_context: Optional market context data
            force_model: Force specific model (overrides tier selection)

        Returns:
            Recommendation dictionary
        """
        try:
            # Select analysis tier
            tier = self._select_analysis_tier(position)
            tier_config = self.ANALYSIS_TIERS[tier]

            logger.info(f"Analyzing {position.symbol} (tier: {tier})")

            # Build prompt
            prompt = self._build_position_analysis_prompt(
                position,
                quant_analysis or {},
                market_context or {}
            )

            # Call LLM service (auto-selects free providers: groq > deepseek > gemini)
            # Async wrapper for the sync llm_service.generate method
            def sync_generate():
                return self.llm_service.generate(
                    prompt=prompt,
                    provider=force_model or tier_config['provider'],  # None = auto-select
                    max_tokens=tier_config['max_tokens'],
                    temperature=tier_config['temperature'],
                    use_cache=True  # Uses LLM service's built-in cache
                )

            # Run in thread pool since LLM service is sync
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(sync_generate)
                response = await asyncio.wrap_future(future)

            # Parse response
            recommendation = self._parse_llm_response(response['text'])

            # Add metadata
            recommendation['model_used'] = f"{response['provider']}/{response['model']}"
            recommendation['tier'] = tier
            recommendation['timestamp'] = datetime.now()
            recommendation['cost'] = response['cost']
            recommendation['cached'] = response['cached']

            logger.info(
                f"✓ {position.symbol}: {recommendation['action'].value} "
                f"(provider: {response['provider']}, cached: {response['cached']})"
            )

            return recommendation

        except Exception as e:
            logger.error(f"Error analyzing position with LLM: {e}")
            return self._get_fallback_recommendation(position)

    def _select_analysis_tier(self, position: EnrichedPosition) -> str:
        """
        Select analysis tier based on position characteristics

        Args:
            position: Position data

        Returns:
            Tier name ('critical', 'standard', or 'bulk')
        """
        # Critical conditions
        if position.pnl_dollar < -500:  # Big loss
            return 'critical'

        if position.dte <= 7 and position.moneyness == 'ITM':  # Assignment risk
            return 'critical'

        if position.pnl_percent < -50:  # Down 50%+
            return 'critical'

        # Bulk conditions (stable, profitable)
        if position.pnl_percent > 30 and position.dte > 30:
            return 'bulk'

        if position.moneyness == 'OTM' and position.dte > 45:
            return 'bulk'

        # Default to standard
        return 'standard'

    def _build_position_analysis_prompt(
        self,
        position: EnrichedPosition,
        quant_analysis: Dict,
        market_context: Dict
    ) -> str:
        """
        Build comprehensive analysis prompt

        Args:
            position: Position data
            quant_analysis: Quantitative analysis results
            market_context: Market context data

        Returns:
            Formatted prompt string
        """
        # Extract key metrics
        quant_action = quant_analysis.get('recommended_action', 'N/A')
        quant_reasoning = quant_analysis.get('reasoning', 'No quantitative analysis available')

        market_regime = market_context.get('regime', 'N/A')
        vix = market_context.get('vix', 'N/A')
        earnings_days = market_context.get('earnings_days', '?')
        top_news = market_context.get('top_headline', 'No recent news')

        prompt = f"""You are an expert options trading advisor analyzing a specific position.

=== POSITION DETAILS ===
Symbol: {position.symbol}
Type: {position.position_type}
Strike: ${position.strike}
Expiration: {position.expiration} ({position.dte} DTE)
Quantity: {position.quantity}

Current Stock Price: ${position.stock_price:.2f}
After-Hours Price: ${position.stock_price_ah or 'N/A'}
Stock Change Today: {position.stock_change_percent:+.2f}%

=== FINANCIAL PERFORMANCE ===
Premium Collected/Paid: ${position.premium_collected:.2f}
Current Value: ${position.current_value:.2f}
P/L: ${position.pnl_dollar:.2f} ({position.pnl_percent:+.1f}%)

=== GREEKS & RISK METRICS ===
Delta: {position.delta:.3f}
Theta: {position.theta:.3f} (Daily decay: ${abs(position.theta * position.quantity * 100):.2f})
Gamma: {position.gamma:.3f}
Vega: {position.vega:.3f}
Implied Volatility: {position.implied_volatility:.1f}%

Moneyness: {position.moneyness}
Distance to Strike: {position.distance_to_strike:+.1f}%
Probability ITM: {position.probability_itm:.1f}%

=== TECHNICAL ANALYSIS ===
Stock Trend: {position.stock_trend or 'N/A'}
RSI: {position.stock_rsi or 'N/A'}
Support Level: ${position.support_level or 0:.2f}
Resistance Level: ${position.resistance_level or 0:.2f}

=== MARKET CONTEXT ===
Market Regime: {market_regime}
VIX Level: {vix}
Earnings in: {earnings_days} days
Recent News: {top_news}

News Sentiment (24h): {position.news_sentiment or 'Neutral'}
News Count: {position.news_count_24h}

=== QUANTITATIVE RECOMMENDATION ===
Rule-Based Signal: {quant_action}
Reasoning: {quant_reasoning}

=== YOUR TASK ===
Analyze this position holistically and provide an actionable recommendation.

**Consider these key factors:**

1. **P/L Status**: Is this winning (+{position.pnl_percent:.1f}%) or losing? How does that affect our strategy?

2. **Time Decay**: With {position.dte} DTE remaining, is theta working for or against us?
   - For short positions (CSP, CC): Theta decay is our friend
   - For long positions: Time is working against us

3. **Moneyness & Assignment Risk**: Position is {position.moneyness}, {abs(position.distance_to_strike):.1f}% from strike.
   - If DTE < 7 and ITM, assignment risk is HIGH
   - If OTM with plenty of time, let it ride

4. **Market Environment**: Current regime is {market_regime}, VIX at {vix}
   - High volatility = higher premiums but more risk
   - Calm markets = favorable for theta strategies

5. **Upcoming Catalysts**: Earnings in {earnings_days} days, news sentiment is {position.news_sentiment or 'neutral'}
   - Should we exit before earnings?
   - Is there headline risk?

6. **Optimal Action**: What's the best move RIGHT NOW?
   - HOLD: Let position work, theta is in our favor
   - CLOSE_NOW: Take profit or cut loss
   - ROLL_OUT: Extend duration to next expiration
   - ROLL_STRIKE: Adjust strike price (up for calls, down for puts)
   - ADD_HEDGE: Add protective position
   - CUT_LOSS: Exit immediately to prevent further damage

**Guidelines:**
- For profitable positions (>30% profit), consider taking profit if DTE < 21
- For losing positions (<-50%), evaluate if the thesis is still valid
- Never hold through earnings unless intentional
- Consider transaction costs (~$0.65 per contract per side)

Respond in this EXACT JSON format:
{{
  "recommendation": "hold" | "close_now" | "roll_out" | "roll_strike" | "add_hedge" | "cut_loss",
  "confidence": 0-100,
  "rationale": "2-3 sentence explanation of your recommendation with specific reasoning",
  "key_factors": [
    "Most important factor 1",
    "Most important factor 2",
    "Most important factor 3"
  ],
  "risk_level": "low" | "medium" | "high",
  "action_details": {{
    "target_exit_price": 123.45,
    "roll_to_date": "2025-12-20",
    "hedge_suggestion": "Buy 1 protective put at $X strike"
  }},
  "urgency": "low" | "medium" | "high",
  "expected_outcome": "Brief description of what you expect to happen if recommendation is followed"
}}

IMPORTANT:
- Be specific and actionable
- Explain WHY this action is best RIGHT NOW
- Consider the trader's moderate risk tolerance
- Account for transaction costs
- Respond ONLY with valid JSON (no additional text)
"""

        return prompt

    def _parse_llm_response(self, content: str) -> Dict:
        """
        Parse LLM JSON response

        Args:
            content: LLM response text

        Returns:
            Parsed recommendation dict
        """
        try:
            # Try to extract JSON from response
            # Sometimes LLMs add extra text despite instructions
            start = content.find('{')
            end = content.rfind('}') + 1

            if start == -1 or end == 0:
                raise ValueError("No JSON found in response")

            json_str = content[start:end]
            recommendation = json.loads(json_str)

            # Validate required fields
            required = ['recommendation', 'confidence', 'rationale', 'key_factors']
            for field in required:
                if field not in recommendation:
                    raise ValueError(f"Missing required field: {field}")

            return recommendation

        except Exception as e:
            logger.error(f"Error parsing LLM response: {e}")
            logger.debug(f"Raw response: {content}")

            # Return fallback
            return {
                'recommendation': 'hold',
                'confidence': 50,
                'rationale': 'Unable to parse LLM response, defaulting to hold',
                'key_factors': ['Analysis error'],
                'risk_level': 'medium',
                'urgency': 'low',
                'action_details': {},
                'expected_outcome': 'Monitor position closely',
                'parse_error': True
            }

    def _get_fallback_recommendation(self, position: EnrichedPosition) -> Dict:
        """
        Generate fallback recommendation when LLM fails

        Args:
            position: Position data

        Returns:
            Basic recommendation dict
        """
        # Simple rule-based fallback
        if position.pnl_percent > 50:
            action = 'close_now'
            rationale = 'Take profit at 50%+'
        elif position.pnl_percent < -100:
            action = 'cut_loss'
            rationale = 'Cut loss before it worsens'
        elif position.dte < 7 and position.moneyness == 'ITM':
            action = 'roll_out'
            rationale = 'Avoid assignment risk'
        else:
            action = 'hold'
            rationale = 'Position within acceptable parameters'

        return {
            'recommendation': action,
            'confidence': 60,
            'rationale': rationale,
            'key_factors': ['Fallback rule-based analysis'],
            'risk_level': 'medium',
            'urgency': 'low',
            'action_details': {},
            'expected_outcome': 'Standard position management',
            'fallback': True
        }


# ============================================================================
# Batch Analysis
# ============================================================================

async def analyze_portfolio_batch(
    positions: list,
    analyzer: PositionLLMAnalyzer
) -> list:
    """
    Analyze multiple positions in parallel

    Args:
        positions: List of EnrichedPosition objects
        analyzer: LLM analyzer instance

    Returns:
        List of recommendations
    """
    tasks = [
        analyzer.analyze_position(pos)
        for pos in positions
    ]

    recommendations = await asyncio.gather(*tasks, return_exceptions=True)

    # Filter out exceptions
    results = []
    for i, rec in enumerate(recommendations):
        if isinstance(rec, Exception):
            logger.error(f"Error analyzing position {i}: {rec}")
            results.append(analyzer._get_fallback_recommendation(positions[i]))
        else:
            results.append(rec)

    return results


# ============================================================================
# Testing
# ============================================================================

async def test_llm_analyzer():
    """Test LLM position analysis"""

    print("\n" + "="*80)
    print("LLM POSITION ANALYZER TEST")
    print("="*80)

    # Import aggregator
    from src.ai.position_data_aggregator import PositionDataAggregator

    # Fetch positions
    aggregator = PositionDataAggregator()
    positions = aggregator.fetch_all_positions()

    if not positions:
        print("\nNo positions found. Create a mock position for testing...")

        # Create mock position
        from datetime import date, timedelta
        mock_position = EnrichedPosition(
            symbol='AAPL',
            position_type='CSP',
            strike=150.0,
            expiration=date.today() + timedelta(days=40),
            dte=40,
            quantity=1,
            premium_collected=250.0,
            current_value=175.0,
            pnl_dollar=-75.0,
            pnl_percent=-30.0,
            stock_price=165.5,
            stock_price_ah=165.8,
            stock_change_percent=2.3,
            delta=-0.15,
            gamma=0.05,
            theta=2.5,
            vega=0.10,
            implied_volatility=30.0,
            moneyness='OTM',
            distance_to_strike=10.3,
            probability_itm=15.0,
            iv_rank=25.0,
            iv_percentile=30.0,
            stock_rsi=62.0,
            stock_trend='bullish',
            support_level=160.0,
            resistance_level=170.0,
            news_sentiment=0.2,
            news_count_24h=3,
            analyzed_at=datetime.now(),
            position_id='AAPL_150_2025-12-20_put'
        )

        positions = [mock_position]

    # Initialize analyzer
    analyzer = PositionLLMAnalyzer()

    # Analyze first position
    position = positions[0]

    print(f"\nAnalyzing: {position.symbol} ${position.strike} {position.position_type}")
    print(f"P/L: ${position.pnl_dollar:.2f} ({position.pnl_percent:+.1f}%)")
    print(f"DTE: {position.dte} | Moneyness: {position.moneyness}")

    # Get recommendation
    recommendation = await analyzer.analyze_position(position)

    print("\n" + "="*80)
    print("RECOMMENDATION")
    print("="*80)
    print(f"Action: {recommendation['recommendation'].upper()}")
    print(f"Confidence: {recommendation['confidence']}%")
    print(f"Risk Level: {recommendation['risk_level'].upper()}")
    print(f"Urgency: {recommendation['urgency'].upper()}")
    print(f"\nRationale: {recommendation['rationale']}")
    print("\nKey Factors:")
    for factor in recommendation.get('key_factors', []):
        print(f"  - {factor}")

    print(f"\nModel Used: {recommendation.get('model_used')}")
    print(f"Cost: ${recommendation.get('cost', 0):.4f}")


if __name__ == "__main__":
    asyncio.run(test_llm_analyzer())

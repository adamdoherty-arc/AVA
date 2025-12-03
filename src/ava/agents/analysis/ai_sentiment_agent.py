"""
AI-Powered Sentiment Analysis Agent
====================================

Uses Claude to analyze market sentiment from multiple sources
and provide actionable trading signals.

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

class SentimentSignal(AgentOutputBase):
    """Individual sentiment signal"""
    source: str = ""  # news, social, options_flow, technical
    signal: str = "neutral"  # bullish, bearish, neutral
    strength: int = Field(default=50, ge=0, le=100)
    description: str = ""
    data_points: List[str] = Field(default_factory=list)


class MarketSentimentOutput(AgentOutputBase):
    """Complete sentiment analysis output"""
    symbol: str = ""

    # Overall sentiment
    overall_sentiment: str = "neutral"  # very_bullish, bullish, neutral, bearish, very_bearish
    sentiment_score: int = Field(default=50, ge=0, le=100)  # 0=very_bearish, 100=very_bullish
    sentiment_change: str = "stable"  # improving, stable, deteriorating

    # Individual signals
    signals: List[SentimentSignal] = Field(default_factory=list)

    # Market regime
    vix_analysis: str = ""
    put_call_ratio: float = 1.0
    put_call_signal: str = "neutral"
    fear_greed_index: Optional[int] = None

    # IV analysis
    iv_vs_hv: str = ""  # elevated, normal, depressed
    iv_percentile: float = 50
    iv_signal: str = "neutral"

    # Options flow
    unusual_activity: List[str] = Field(default_factory=list)
    large_trades: List[str] = Field(default_factory=list)

    # News sentiment
    recent_news_sentiment: str = "neutral"
    key_headlines: List[str] = Field(default_factory=list)

    # Trading implication
    trading_bias: str = "neutral"
    suggested_strategies: List[str] = Field(default_factory=list)
    caution_factors: List[str] = Field(default_factory=list)


# =============================================================================
# SENTIMENT ANALYSIS AGENT
# =============================================================================

class AISentimentAgent(LLMAgent[MarketSentimentOutput]):
    """
    AI-powered sentiment analysis agent.

    Analyzes multiple sentiment indicators to provide
    a comprehensive market sentiment assessment.

    Usage:
        agent = AISentimentAgent()
        result = await agent.execute({
            "symbol": "AAPL",
            "vix": 15.5,
            "iv_rank": 45,
            "put_call_ratio": 0.85,
            "recent_news": [...],
            "options_flow": [...]
        })
    """

    name = "ai_sentiment"
    description = "Analyzes market sentiment from multiple sources"
    output_model = MarketSentimentOutput
    temperature = 0.4

    system_prompt = """You are an expert market sentiment analyst specializing in:

1. VOLATILITY INDICATORS:
   - VIX interpretation (fear gauge)
   - IV Rank and IV Percentile meaning
   - Put/Call ratio analysis
   - Volatility skew

2. OPTIONS FLOW ANALYSIS:
   - Unusual options activity detection
   - Large block trades significance
   - Smart money vs retail flow
   - Sweep orders and their meaning

3. NEWS & SOCIAL SENTIMENT:
   - Headline sentiment analysis
   - Social media trend detection
   - Analyst rating changes
   - Insider trading signals

4. TECHNICAL SENTIMENT:
   - Market breadth indicators
   - Advance/decline ratios
   - New highs vs new lows
   - Sector rotation signals

Your role is to:
1. Synthesize all sentiment inputs into a clear signal
2. Score overall sentiment (0=very_bearish, 100=very_bullish)
3. Identify the strongest sentiment drivers
4. Suggest trading strategies that align with sentiment
5. Highlight any conflicting signals or caution factors

Be specific about what each indicator is telling us."""

    def build_prompt(self, input_data: Dict[str, Any]) -> str:
        """Build sentiment analysis prompt"""
        symbol = input_data.get('symbol', 'MARKET')
        vix = input_data.get('vix', 15)
        iv_rank = input_data.get('iv_rank', 50)
        iv_percentile = input_data.get('iv_percentile', iv_rank)
        hv_20 = input_data.get('hv_20', 0.20)
        put_call_ratio = input_data.get('put_call_ratio', 1.0)
        recent_news = input_data.get('recent_news', [])
        options_flow = input_data.get('options_flow', [])
        fear_greed = input_data.get('fear_greed_index', None)

        prompt = f"""## Sentiment Analysis Request: {symbol}

### Volatility Indicators
- **VIX**: {vix:.1f} ({"ELEVATED - fear present" if vix > 20 else "LOW - complacency" if vix < 15 else "NORMAL"})
- **IV Rank**: {iv_rank:.1f}%
- **IV Percentile**: {iv_percentile:.1f}%
- **Historical Vol (20d)**: {hv_20:.1%}
- **IV Premium**: {"Elevated" if iv_rank > iv_percentile else "Depressed" if iv_rank < iv_percentile else "Normal"}

### Options Market
- **Put/Call Ratio**: {put_call_ratio:.2f} ({"BEARISH - high put buying" if put_call_ratio > 1.2 else "BULLISH - low put buying" if put_call_ratio < 0.7 else "NEUTRAL"})
"""

        if fear_greed is not None:
            prompt += f"- **Fear & Greed Index**: {fear_greed}/100\n"

        # Add options flow
        if options_flow:
            prompt += "\n### Recent Options Flow\n"
            for flow in options_flow[:10]:
                prompt += f"- {flow}\n"
        else:
            prompt += "\n### Recent Options Flow\nNo unusual activity detected.\n"

        # Add news
        if recent_news:
            prompt += "\n### Recent News Headlines\n"
            for news in recent_news[:10]:
                if isinstance(news, dict):
                    prompt += f"- {news.get('headline', news)}\n"
                else:
                    prompt += f"- {news}\n"
        else:
            prompt += "\n### Recent News Headlines\nNo significant news.\n"

        prompt += """
### Your Analysis Task

1. **Overall Sentiment Score (0-100)**:
   - 0-20: Very Bearish
   - 21-40: Bearish
   - 41-60: Neutral
   - 61-80: Bullish
   - 81-100: Very Bullish

2. **Signal Breakdown**: For each source (VIX, IV, Put/Call, News, Flow):
   - What is the signal?
   - How strong is it?
   - Key data points

3. **IV Analysis**:
   - Is IV elevated or depressed relative to HV?
   - What does this mean for options strategies?

4. **Options Flow Interpretation**:
   - Any unusual activity?
   - Smart money signals?

5. **Trading Implications**:
   - What trading bias does sentiment suggest?
   - Which strategies align with current sentiment?
   - What should traders be cautious about?

6. **Conflicting Signals**: Note any indicators that disagree.
"""
        return prompt

    def parse_response(self, response: str, input_data: Dict[str, Any]) -> MarketSentimentOutput:
        """Parse sentiment response"""
        import json
        import re

        try:
            json_match = re.search(r'\{[\s\S]*\}', response)
            if json_match:
                data = json.loads(json_match.group())
                data['agent_name'] = self.name
                data['symbol'] = input_data.get('symbol', '')
                return MarketSentimentOutput(**data)

        except Exception as e:
            logger.warning(f"Failed to parse JSON: {e}")

        return self._parse_text_response(response, input_data)

    def _parse_text_response(
        self,
        response: str,
        input_data: Dict[str, Any]
    ) -> MarketSentimentOutput:
        """Parse unstructured response"""
        import re

        response_lower = response.lower()

        # Determine sentiment
        bullish_keywords = ['bullish', 'positive', 'optimistic', 'buy', 'upside']
        bearish_keywords = ['bearish', 'negative', 'pessimistic', 'sell', 'downside']

        bullish_count = sum(1 for k in bullish_keywords if k in response_lower)
        bearish_count = sum(1 for k in bearish_keywords if k in response_lower)

        if bullish_count > bearish_count + 2:
            overall_sentiment = "very_bullish"
            sentiment_score = 80
        elif bullish_count > bearish_count:
            overall_sentiment = "bullish"
            sentiment_score = 65
        elif bearish_count > bullish_count + 2:
            overall_sentiment = "very_bearish"
            sentiment_score = 20
        elif bearish_count > bullish_count:
            overall_sentiment = "bearish"
            sentiment_score = 35
        else:
            overall_sentiment = "neutral"
            sentiment_score = 50

        # Extract score if mentioned
        score_match = re.search(r'sentiment\s*score[:\s]*(\d+)', response_lower)
        if score_match:
            sentiment_score = min(100, max(0, int(score_match.group(1))))

        # Analyze IV
        iv_rank = input_data.get('iv_rank', 50)
        if iv_rank > 60:
            iv_vs_hv = "elevated"
            iv_signal = "sell_premium"
        elif iv_rank < 30:
            iv_vs_hv = "depressed"
            iv_signal = "buy_premium"
        else:
            iv_vs_hv = "normal"
            iv_signal = "neutral"

        # Put/call interpretation
        pcr = input_data.get('put_call_ratio', 1.0)
        if pcr > 1.2:
            pc_signal = "bearish"
        elif pcr < 0.7:
            pc_signal = "bullish"
        else:
            pc_signal = "neutral"

        return MarketSentimentOutput(
            agent_name=self.name,
            symbol=input_data.get('symbol', ''),
            overall_sentiment=overall_sentiment,
            sentiment_score=sentiment_score,
            put_call_ratio=pcr,
            put_call_signal=pc_signal,
            iv_vs_hv=iv_vs_hv,
            iv_percentile=iv_rank,
            iv_signal=iv_signal,
            reasoning=response,
            confidence=AgentConfidence.MEDIUM
        )


# =============================================================================
# QUICK SENTIMENT CHECKER
# =============================================================================

class QuickSentimentChecker:
    """Rule-based quick sentiment check"""

    @staticmethod
    def check(
        vix: float = 15,
        iv_rank: float = 50,
        put_call_ratio: float = 1.0,
        price_vs_sma20: float = 0  # % above/below
    ) -> Dict:
        """Quick sentiment assessment"""
        signals = []
        score = 50  # Start neutral

        # VIX signal
        if vix > 25:
            signals.append({"source": "VIX", "signal": "bearish", "note": "Fear elevated"})
            score -= 15
        elif vix < 15:
            signals.append({"source": "VIX", "signal": "bullish", "note": "Complacency/confidence"})
            score += 10
        else:
            signals.append({"source": "VIX", "signal": "neutral", "note": "Normal range"})

        # IV signal
        if iv_rank > 60:
            signals.append({"source": "IV", "signal": "neutral", "note": "Sell premium environment"})
        elif iv_rank < 30:
            signals.append({"source": "IV", "signal": "neutral", "note": "Buy premium environment"})

        # Put/Call signal (contrarian)
        if put_call_ratio > 1.2:
            signals.append({"source": "Put/Call", "signal": "bullish", "note": "Extreme fear = contrarian buy"})
            score += 10
        elif put_call_ratio < 0.7:
            signals.append({"source": "Put/Call", "signal": "bearish", "note": "Extreme greed = contrarian sell"})
            score -= 10

        # Price vs SMA
        if price_vs_sma20 > 5:
            signals.append({"source": "Technical", "signal": "bullish", "note": "Price above SMA20"})
            score += 5
        elif price_vs_sma20 < -5:
            signals.append({"source": "Technical", "signal": "bearish", "note": "Price below SMA20"})
            score -= 5

        # Determine overall
        score = min(100, max(0, score))

        if score >= 70:
            overall = "bullish"
        elif score >= 55:
            overall = "slightly_bullish"
        elif score <= 30:
            overall = "bearish"
        elif score <= 45:
            overall = "slightly_bearish"
        else:
            overall = "neutral"

        return {
            "sentiment_score": score,
            "overall_sentiment": overall,
            "signals": signals
        }


# =============================================================================
# TESTING
# =============================================================================

if __name__ == "__main__":
    import asyncio

    print("\n=== Testing AI Sentiment Agent ===\n")

    async def test_agent():
        # Test quick checker
        print("1. Testing Quick Sentiment Checker...")
        checker = QuickSentimentChecker()

        test_cases = [
            {"vix": 30, "iv_rank": 70, "put_call_ratio": 1.4},  # Fearful
            {"vix": 12, "iv_rank": 25, "put_call_ratio": 0.6},  # Greedy
            {"vix": 18, "iv_rank": 45, "put_call_ratio": 1.0},  # Neutral
        ]

        for case in test_cases:
            result = checker.check(**case)
            print(f"\n   VIX={case['vix']}, PCR={case['put_call_ratio']}:")
            print(f"      Score: {result['sentiment_score']}")
            print(f"      Sentiment: {result['overall_sentiment']}")

        # Test agent structure
        print("\n2. Testing Agent Structure...")
        agent = AISentimentAgent()
        print(f"   Name: {agent.name}")
        print(f"   Output model: {agent.output_model.__name__}")

        print("\n✅ AI Sentiment Agent ready!")

    asyncio.run(test_agent())

"""
AI-Powered Technical Analysis Agent
====================================

Uses Claude to analyze price action, technical indicators,
and chart patterns with intelligent interpretation.

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

class TechnicalIndicator(AgentOutputBase):
    """Individual technical indicator result"""
    name: str = ""
    value: float = 0.0
    signal: str = ""  # BULLISH, BEARISH, NEUTRAL
    interpretation: str = ""


class SupportResistanceLevel(AgentOutputBase):
    """Support or resistance level"""
    price: float = 0.0
    level_type: str = ""  # support, resistance
    strength: str = ""  # weak, moderate, strong
    distance_pct: float = 0.0
    notes: str = ""


class ChartPattern(AgentOutputBase):
    """Detected chart pattern"""
    pattern_name: str = ""
    pattern_type: str = ""  # bullish, bearish, neutral
    confidence: str = ""  # low, medium, high
    target_price: Optional[float] = None
    description: str = ""


class TradingSignal(AgentOutputBase):
    """Overall trading signal"""
    direction: str = ""  # BUY, SELL, HOLD
    strength: str = ""  # weak, moderate, strong
    confidence_score: float = 0.0
    entry_price: Optional[float] = None
    stop_loss: Optional[float] = None
    target_price: Optional[float] = None
    risk_reward_ratio: Optional[float] = None
    rationale: str = ""


class TechnicalOutput(AgentOutputBase):
    """Complete technical analysis output"""
    symbol: str = ""
    current_price: float = 0.0
    analysis_timestamp: str = ""

    # Trend
    overall_trend: str = ""  # UPTREND, DOWNTREND, SIDEWAYS
    trend_strength: str = ""  # weak, moderate, strong
    trend_duration: str = ""  # short-term, medium-term, long-term

    # Moving Averages
    sma_20: Optional[float] = None
    sma_50: Optional[float] = None
    sma_200: Optional[float] = None
    ma_alignment: str = ""  # bullish, bearish, mixed

    # Key Indicators
    indicators: List[TechnicalIndicator] = Field(default_factory=list)

    # Support/Resistance
    support_levels: List[SupportResistanceLevel] = Field(default_factory=list)
    resistance_levels: List[SupportResistanceLevel] = Field(default_factory=list)
    nearest_support: Optional[float] = None
    nearest_resistance: Optional[float] = None

    # Patterns
    patterns_detected: List[ChartPattern] = Field(default_factory=list)

    # Volume Analysis
    volume_trend: str = ""  # increasing, decreasing, stable
    volume_signal: str = ""  # confirms_trend, diverges, neutral

    # Trading Signal
    signal: TradingSignal = Field(default_factory=TradingSignal)

    # Options implications
    iv_environment: str = ""  # high, normal, low
    options_bias: str = ""  # sell_premium, buy_premium, neutral
    strategy_suggestions: List[str] = Field(default_factory=list)

    # Key insights
    key_insights: List[str] = Field(default_factory=list)
    risk_factors: List[str] = Field(default_factory=list)


# =============================================================================
# TECHNICAL AGENT
# =============================================================================

class AITechnicalAgent(LLMAgent[TechnicalOutput]):
    """
    AI-powered technical analysis agent.

    Provides comprehensive technical analysis with intelligent
    interpretation using Claude.

    Usage:
        agent = AITechnicalAgent()
        result = await agent.execute({
            "symbol": "AAPL",
            "price_data": {
                "current": 175.50,
                "open": 174.00,
                "high": 176.20,
                "low": 173.80,
                "previous_close": 174.50
            },
            "indicators": {
                "rsi": 65.5,
                "macd": {"line": 1.25, "signal": 0.95, "histogram": 0.30},
                "bb": {"upper": 180.0, "middle": 175.0, "lower": 170.0}
            },
            "volume_data": {...},
            "historical_data": [...]
        })
    """

    name = "ai_technical"
    description = "AI-powered technical analysis for options trading"
    output_model = TechnicalOutput
    temperature = 0.2  # Lower for more consistent technical interpretation

    system_prompt = """You are an expert technical analyst specializing in:

1. PRICE ACTION ANALYSIS:
   - Trend identification and strength
   - Support/resistance levels
   - Chart patterns (head & shoulders, triangles, flags, etc.)
   - Candlestick patterns
   - Market structure (higher highs, lower lows)

2. TECHNICAL INDICATORS:
   - Momentum (RSI, MACD, Stochastic)
   - Trend (Moving averages, ADX)
   - Volatility (Bollinger Bands, ATR)
   - Volume (OBV, Volume Profile, VWAP)

3. SMART MONEY CONCEPTS:
   - Order blocks and fair value gaps
   - Liquidity pools
   - Market structure breaks
   - Institutional positioning

4. OPTIONS-SPECIFIC ANALYSIS:
   - IV environment assessment
   - Premium selling vs buying conditions
   - Strike selection based on technicals
   - Expiration timing with price targets

5. RISK ASSESSMENT:
   - Stop loss placement
   - Risk/reward calculations
   - Position sizing recommendations
   - Key levels to watch

Your analysis should:
1. Be actionable with specific price levels
2. Consider multiple timeframes
3. Account for current IV environment
4. Provide options strategy suggestions
5. Include clear risk parameters"""

    def build_prompt(self, input_data: Dict[str, Any]) -> str:
        """Build technical analysis prompt"""
        symbol = input_data.get('symbol', 'UNKNOWN')
        price_data = input_data.get('price_data', {})
        indicators = input_data.get('indicators', {})
        volume_data = input_data.get('volume_data', {})
        historical = input_data.get('historical_data', [])
        options_data = input_data.get('options_data', {})

        current_price = price_data.get('current', 0)

        prompt = f"""## Technical Analysis Request

### Symbol: {symbol}
### Current Price: ${current_price:.2f}

### Price Action
"""

        if price_data:
            prompt += f"""
- Open: ${price_data.get('open', 0):.2f}
- High: ${price_data.get('high', 0):.2f}
- Low: ${price_data.get('low', 0):.2f}
- Previous Close: ${price_data.get('previous_close', 0):.2f}
- Change: {((current_price - price_data.get('previous_close', current_price)) / price_data.get('previous_close', 1) * 100):.2f}%
"""

        prompt += "\n### Technical Indicators\n"

        if indicators:
            # RSI
            if 'rsi' in indicators:
                rsi = indicators['rsi']
                prompt += f"- RSI (14): {rsi:.1f}"
                if rsi < 30:
                    prompt += " [OVERSOLD]"
                elif rsi > 70:
                    prompt += " [OVERBOUGHT]"
                prompt += "\n"

            # MACD
            if 'macd' in indicators:
                macd = indicators['macd']
                prompt += f"- MACD Line: {macd.get('line', 0):.2f}\n"
                prompt += f"- MACD Signal: {macd.get('signal', 0):.2f}\n"
                prompt += f"- MACD Histogram: {macd.get('histogram', 0):.2f}\n"

            # Bollinger Bands
            if 'bb' in indicators:
                bb = indicators['bb']
                prompt += f"- Bollinger Upper: ${bb.get('upper', 0):.2f}\n"
                prompt += f"- Bollinger Middle: ${bb.get('middle', 0):.2f}\n"
                prompt += f"- Bollinger Lower: ${bb.get('lower', 0):.2f}\n"

            # Moving Averages
            if 'sma_20' in indicators:
                prompt += f"- SMA 20: ${indicators['sma_20']:.2f}\n"
            if 'sma_50' in indicators:
                prompt += f"- SMA 50: ${indicators['sma_50']:.2f}\n"
            if 'sma_200' in indicators:
                prompt += f"- SMA 200: ${indicators['sma_200']:.2f}\n"

            # ATR
            if 'atr' in indicators:
                prompt += f"- ATR (14): ${indicators['atr']:.2f}\n"

            # Stochastic
            if 'stochastic' in indicators:
                stoch = indicators['stochastic']
                prompt += f"- Stochastic K: {stoch.get('k', 0):.1f}\n"
                prompt += f"- Stochastic D: {stoch.get('d', 0):.1f}\n"

        prompt += "\n### Volume Analysis\n"
        if volume_data:
            prompt += f"""
- Current Volume: {volume_data.get('current', 0):,}
- Average Volume (20d): {volume_data.get('avg_20d', 0):,}
- Volume Ratio: {volume_data.get('ratio', 1.0):.2f}x
- OBV Trend: {volume_data.get('obv_trend', 'unknown')}
"""

        # Historical data for context
        if historical and len(historical) >= 5:
            prompt += "\n### Recent Price History (Last 5 days)\n"
            for day in historical[-5:]:
                prompt += f"- {day.get('date', 'N/A')}: ${day.get('close', 0):.2f} (H: ${day.get('high', 0):.2f}, L: ${day.get('low', 0):.2f})\n"

        # Support/Resistance if provided
        if input_data.get('support_levels'):
            prompt += f"\n### Identified Support Levels\n"
            for level in input_data['support_levels'][:3]:
                prompt += f"- ${level:.2f}\n"

        if input_data.get('resistance_levels'):
            prompt += f"\n### Identified Resistance Levels\n"
            for level in input_data['resistance_levels'][:3]:
                prompt += f"- ${level:.2f}\n"

        # Options context
        if options_data:
            prompt += f"""
### Options Context
- Current IV: {options_data.get('iv', 0):.1f}%
- IV Rank: {options_data.get('iv_rank', 0):.0f}%
- IV Percentile: {options_data.get('iv_percentile', 0):.0f}%
- Put/Call Ratio: {options_data.get('put_call_ratio', 1.0):.2f}
"""

        prompt += """

### Analysis Required

1. **Trend Analysis**:
   - Overall trend direction and strength
   - Key trend line levels
   - Trend duration/maturity

2. **Indicator Interpretation**:
   - RSI: overbought/oversold, divergences
   - MACD: momentum direction, crossovers
   - Bollinger Bands: squeeze, breakout potential
   - MA alignment: bullish/bearish/mixed

3. **Support/Resistance**:
   - Key levels to watch
   - Strength of each level
   - Distance from current price

4. **Pattern Recognition**:
   - Any chart patterns forming
   - Candlestick patterns
   - Pattern completion probability

5. **Volume Confirmation**:
   - Is volume confirming price action?
   - Any volume divergences?

6. **Trading Signal**:
   - Clear BUY/SELL/HOLD recommendation
   - Entry price, stop loss, target
   - Risk/reward ratio

7. **Options Strategy Suggestions**:
   - Given the technical setup, what options strategies fit?
   - For wheel strategy: is this a good entry?
   - Strike selection based on support/resistance

8. **Risk Factors**:
   - Key levels that would invalidate the thesis
   - Upcoming catalysts to watch
   - Volatility considerations

Provide specific price levels and actionable recommendations.
"""
        return prompt

    def parse_response(self, response: str, input_data: Dict[str, Any]) -> TechnicalOutput:
        """Parse technical analysis response"""
        import json
        import re

        try:
            json_match = re.search(r'\{[\s\S]*\}', response)
            if json_match:
                data = json.loads(json_match.group())
                data['agent_name'] = self.name
                return TechnicalOutput(**data)
        except Exception as e:
            logger.warning(f"Failed to parse JSON response: {e}")

        return self._parse_text_response(response, input_data)

    def _parse_text_response(
        self,
        response: str,
        input_data: Dict[str, Any]
    ) -> TechnicalOutput:
        """Parse unstructured text response"""
        symbol = input_data.get('symbol', 'UNKNOWN')
        price_data = input_data.get('price_data', {})
        indicators = input_data.get('indicators', {})

        current_price = price_data.get('current', 0)
        response_lower = response.lower()

        # Determine trend
        if 'uptrend' in response_lower or 'bullish trend' in response_lower:
            trend = 'UPTREND'
        elif 'downtrend' in response_lower or 'bearish trend' in response_lower:
            trend = 'DOWNTREND'
        else:
            trend = 'SIDEWAYS'

        # Determine signal
        if 'strong buy' in response_lower:
            direction = 'BUY'
            strength = 'strong'
        elif 'buy' in response_lower:
            direction = 'BUY'
            strength = 'moderate'
        elif 'strong sell' in response_lower:
            direction = 'SELL'
            strength = 'strong'
        elif 'sell' in response_lower:
            direction = 'SELL'
            strength = 'moderate'
        else:
            direction = 'HOLD'
            strength = 'moderate'

        # Build indicators list
        parsed_indicators = []
        if 'rsi' in indicators:
            rsi = indicators['rsi']
            parsed_indicators.append(TechnicalIndicator(
                name='RSI',
                value=rsi,
                signal='OVERSOLD' if rsi < 30 else 'OVERBOUGHT' if rsi > 70 else 'NEUTRAL',
                interpretation=f"RSI at {rsi:.1f}"
            ))

        if 'macd' in indicators:
            macd = indicators['macd']
            hist = macd.get('histogram', 0)
            parsed_indicators.append(TechnicalIndicator(
                name='MACD',
                value=hist,
                signal='BULLISH' if hist > 0 else 'BEARISH',
                interpretation=f"MACD histogram at {hist:.2f}"
            ))

        # Options bias
        if 'sell premium' in response_lower or 'high iv' in response_lower:
            options_bias = 'sell_premium'
        elif 'buy premium' in response_lower or 'low iv' in response_lower:
            options_bias = 'buy_premium'
        else:
            options_bias = 'neutral'

        # Extract key insights from response
        insights = []
        sentences = response.split('.')
        for sent in sentences[:5]:
            if len(sent.strip()) > 20:
                insights.append(sent.strip())

        return TechnicalOutput(
            agent_name=self.name,
            symbol=symbol,
            current_price=current_price,
            analysis_timestamp=datetime.now().isoformat(),
            overall_trend=trend,
            trend_strength=strength,
            trend_duration='medium-term',
            sma_20=indicators.get('sma_20'),
            sma_50=indicators.get('sma_50'),
            sma_200=indicators.get('sma_200'),
            ma_alignment='bullish' if trend == 'UPTREND' else 'bearish' if trend == 'DOWNTREND' else 'mixed',
            indicators=parsed_indicators,
            signal=TradingSignal(
                direction=direction,
                strength=strength,
                confidence_score=75.0 if strength == 'strong' else 60.0 if strength == 'moderate' else 45.0,
                rationale=f"Based on {trend.lower()} with {len(parsed_indicators)} confirming indicators"
            ),
            options_bias=options_bias,
            key_insights=insights[:3],
            reasoning=response,
            confidence=AgentConfidence.MEDIUM
        )


# =============================================================================
# QUICK TECHNICAL CALCULATOR
# =============================================================================

class QuickTechnicalCalculator:
    """Rule-based quick technical analysis without LLM"""

    @staticmethod
    def calculate_trend(
        current_price: float,
        sma_20: float,
        sma_50: float,
        sma_200: Optional[float] = None
    ) -> Dict:
        """Quick trend calculation"""
        if current_price > sma_20 > sma_50:
            if sma_200 is None or current_price > sma_200:
                return {'trend': 'UPTREND', 'strength': 'strong', 'alignment': 'bullish'}
            return {'trend': 'UPTREND', 'strength': 'moderate', 'alignment': 'mixed'}
        elif current_price < sma_20 < sma_50:
            if sma_200 is None or current_price < sma_200:
                return {'trend': 'DOWNTREND', 'strength': 'strong', 'alignment': 'bearish'}
            return {'trend': 'DOWNTREND', 'strength': 'moderate', 'alignment': 'mixed'}
        else:
            return {'trend': 'SIDEWAYS', 'strength': 'weak', 'alignment': 'mixed'}

    @staticmethod
    def interpret_rsi(rsi: float) -> Dict:
        """Interpret RSI value"""
        if rsi < 30:
            return {
                'signal': 'OVERSOLD',
                'action': 'Consider buying',
                'strength': 'strong' if rsi < 20 else 'moderate'
            }
        elif rsi > 70:
            return {
                'signal': 'OVERBOUGHT',
                'action': 'Consider selling',
                'strength': 'strong' if rsi > 80 else 'moderate'
            }
        elif rsi < 40:
            return {
                'signal': 'BEARISH',
                'action': 'Caution on longs',
                'strength': 'weak'
            }
        elif rsi > 60:
            return {
                'signal': 'BULLISH',
                'action': 'Momentum building',
                'strength': 'weak'
            }
        return {
            'signal': 'NEUTRAL',
            'action': 'No clear signal',
            'strength': 'none'
        }

    @staticmethod
    def calculate_support_resistance(
        highs: List[float],
        lows: List[float],
        current_price: float
    ) -> Dict:
        """Calculate basic support/resistance"""
        if not highs or not lows:
            return {'support': None, 'resistance': None}

        # Find levels below and above current price
        supports = sorted([l for l in lows if l < current_price], reverse=True)
        resistances = sorted([h for h in highs if h > current_price])

        return {
            'support': supports[0] if supports else None,
            'resistance': resistances[0] if resistances else None,
            'support_distance_pct': ((current_price - supports[0]) / current_price * 100) if supports else None,
            'resistance_distance_pct': ((resistances[0] - current_price) / current_price * 100) if resistances else None
        }

    @staticmethod
    def generate_signal(
        rsi: float,
        macd_histogram: float,
        trend: str,
        bb_position: str
    ) -> Dict:
        """Generate trading signal from indicators"""
        bullish_count = 0
        bearish_count = 0

        # RSI
        if rsi < 30:
            bullish_count += 2
        elif rsi > 70:
            bearish_count += 2
        elif rsi < 50:
            bearish_count += 1
        else:
            bullish_count += 1

        # MACD
        if macd_histogram > 0:
            bullish_count += 2
        else:
            bearish_count += 2

        # Trend
        if trend == 'UPTREND':
            bullish_count += 2
        elif trend == 'DOWNTREND':
            bearish_count += 2

        # BB position
        if bb_position in ['BELOW_LOWER', 'LOWER_HALF']:
            bullish_count += 1
        elif bb_position in ['ABOVE_UPPER', 'UPPER_HALF']:
            bearish_count += 1

        total = bullish_count + bearish_count
        if bullish_count > bearish_count:
            confidence = (bullish_count / total) * 100
            return {
                'direction': 'BUY',
                'confidence': confidence,
                'strength': 'strong' if confidence > 75 else 'moderate'
            }
        elif bearish_count > bullish_count:
            confidence = (bearish_count / total) * 100
            return {
                'direction': 'SELL',
                'confidence': confidence,
                'strength': 'strong' if confidence > 75 else 'moderate'
            }
        return {
            'direction': 'HOLD',
            'confidence': 50.0,
            'strength': 'weak'
        }

    @staticmethod
    def options_strategy_for_setup(
        trend: str,
        iv_rank: float,
        signal_direction: str
    ) -> List[str]:
        """Suggest options strategies based on technical setup"""
        strategies = []

        if iv_rank > 50:  # High IV - favor selling
            if trend == 'UPTREND':
                strategies.append('Sell cash-secured puts at support')
                strategies.append('Bull put spread')
            elif trend == 'DOWNTREND':
                strategies.append('Sell covered calls at resistance')
                strategies.append('Bear call spread')
            else:
                strategies.append('Iron condor')
                strategies.append('Short strangle')
        else:  # Low IV - consider buying
            if signal_direction == 'BUY':
                strategies.append('Long call')
                strategies.append('Call debit spread')
            elif signal_direction == 'SELL':
                strategies.append('Long put')
                strategies.append('Put debit spread')
            else:
                strategies.append('Long straddle if expecting breakout')

        return strategies


# =============================================================================
# TESTING
# =============================================================================

if __name__ == "__main__":
    import asyncio

    print("\n=== Testing AI Technical Agent ===\n")

    async def test_agent():
        # Test quick calculator
        print("1. Testing Quick Technical Calculator...")
        calc = QuickTechnicalCalculator()

        trend = calc.calculate_trend(
            current_price=175.50,
            sma_20=173.00,
            sma_50=170.00,
            sma_200=165.00
        )
        print(f"\n   Trend Analysis:")
        print(f"      Trend: {trend['trend']}")
        print(f"      Strength: {trend['strength']}")
        print(f"      MA Alignment: {trend['alignment']}")

        rsi_result = calc.interpret_rsi(28.5)
        print(f"\n   RSI Interpretation:")
        print(f"      Signal: {rsi_result['signal']}")
        print(f"      Action: {rsi_result['action']}")

        signal = calc.generate_signal(
            rsi=28.5,
            macd_histogram=0.35,
            trend='UPTREND',
            bb_position='LOWER_HALF'
        )
        print(f"\n   Trading Signal:")
        print(f"      Direction: {signal['direction']}")
        print(f"      Confidence: {signal['confidence']:.0f}%")
        print(f"      Strength: {signal['strength']}")

        strategies = calc.options_strategy_for_setup(
            trend='UPTREND',
            iv_rank=65.0,
            signal_direction='BUY'
        )
        print(f"\n   Suggested Strategies: {strategies}")

        # Test agent structure
        print("\n2. Testing Agent Structure...")
        agent = AITechnicalAgent()
        print(f"   Name: {agent.name}")
        print(f"   Output model: {agent.output_model.__name__}")

        print("\n✅ AI Technical Agent ready!")

    asyncio.run(test_agent())

"""
AI-Powered Options Flow Agent
=============================

Uses Claude to analyze unusual options activity and smart money flow
to provide actionable trading signals.

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

class UnusualTrade(AgentOutputBase):
    """Individual unusual options trade"""
    symbol: str = ""
    trade_type: str = ""  # call, put
    direction: str = ""  # buy, sell
    strike: float = 0.0
    expiration: str = ""
    premium: float = 0.0
    volume: int = 0
    open_interest: int = 0
    volume_oi_ratio: float = 0.0

    # Classification
    is_sweep: bool = False
    is_block: bool = False
    trade_size: str = ""  # small, medium, large, whale

    # Analysis
    sentiment: str = ""  # bullish, bearish, neutral
    conviction_level: str = ""  # low, medium, high
    smart_money_indicator: bool = False

    notes: str = ""


class FlowSummary(AgentOutputBase):
    """Summary of options flow"""
    total_call_premium: float = 0.0
    total_put_premium: float = 0.0
    call_put_premium_ratio: float = 1.0

    bullish_flow_pct: float = 50.0
    bearish_flow_pct: float = 50.0

    largest_single_trade: float = 0.0
    sweep_count: int = 0
    block_count: int = 0


class OptionsFlowOutput(AgentOutputBase):
    """Complete options flow analysis output"""
    symbol: str = ""
    underlying_price: float = 0.0
    analysis_timestamp: str = ""

    # Overall flow signal
    flow_signal: str = "neutral"  # very_bullish, bullish, neutral, bearish, very_bearish
    signal_strength: int = Field(default=50, ge=0, le=100)
    confidence_level: str = ""

    # Flow summary
    flow_summary: FlowSummary = Field(default_factory=FlowSummary)

    # Notable trades
    unusual_trades: List[UnusualTrade] = Field(default_factory=list)
    whale_trades: List[UnusualTrade] = Field(default_factory=list)
    smart_money_trades: List[UnusualTrade] = Field(default_factory=list)

    # Pattern detection
    accumulation_detected: bool = False
    distribution_detected: bool = False
    hedging_activity: bool = False

    # Interpretation
    key_observations: List[str] = Field(default_factory=list)
    trading_implications: List[str] = Field(default_factory=list)

    # Suggested actions
    suggested_trades: List[str] = Field(default_factory=list)
    caution_factors: List[str] = Field(default_factory=list)


# =============================================================================
# OPTIONS FLOW AGENT
# =============================================================================

class AIOptionsFlowAgent(LLMAgent[OptionsFlowOutput]):
    """
    AI-powered options flow analysis agent.

    Analyzes unusual options activity to detect smart money
    positioning and provide trading signals.

    Usage:
        agent = AIOptionsFlowAgent()
        result = await agent.execute({
            "symbol": "AAPL",
            "underlying_price": 185.50,
            "flow_data": [...],  # List of recent trades
            "historical_flow": {...}
        })
    """

    name = "ai_options_flow"
    description = "Analyzes unusual options activity and smart money flow"
    output_model = OptionsFlowOutput
    temperature = 0.3

    system_prompt = """You are an expert options flow analyst specializing in:

1. UNUSUAL ACTIVITY DETECTION:
   - Volume vs Open Interest spikes
   - Large block trades
   - Sweep orders (aggressive fills)
   - Dark pool activity

2. SMART MONEY INDICATORS:
   - Institutional positioning patterns
   - Hedging vs directional trades
   - Size classification (retail vs whale)
   - Repeat patterns from known traders

3. FLOW INTERPRETATION:
   - Call/Put premium imbalances
   - Strike clustering patterns
   - Expiration preferences
   - ITM vs OTM positioning

4. TRADE CLASSIFICATION:
   - Opening vs closing trades
   - Spread trades vs directional
   - Volatility plays vs directional
   - Event-driven positioning

5. SENTIMENT ANALYSIS:
   - Bullish flow patterns
   - Bearish flow patterns
   - Hedging activity detection
   - Accumulation vs distribution

Your role is to:
1. Identify the most significant unusual trades
2. Classify sentiment (bullish/bearish/neutral)
3. Detect smart money patterns
4. Provide actionable trading signals
5. Highlight any concerning patterns

Focus on trades that matter - ignore noise from retail flow."""

    def build_prompt(self, input_data: Dict[str, Any]) -> str:
        """Build options flow analysis prompt"""
        symbol = input_data.get('symbol', 'UNKNOWN')
        price = input_data.get('underlying_price', 0)
        flow_data = input_data.get('flow_data', [])

        prompt = f"""## Options Flow Analysis Request: {symbol}

### Current Data
- **Symbol**: {symbol}
- **Underlying Price**: ${price:.2f}
- **Time**: {datetime.now().strftime('%Y-%m-%d %H:%M')}

### Recent Options Flow
"""

        if flow_data:
            # Sort by premium (largest first)
            sorted_flow = sorted(flow_data, key=lambda x: x.get('premium', 0), reverse=True)

            total_call_prem = sum(t.get('premium', 0) for t in flow_data if t.get('type') == 'call')
            total_put_prem = sum(t.get('premium', 0) for t in flow_data if t.get('type') == 'put')

            prompt += f"""
**Flow Summary:**
- Total Call Premium: ${total_call_prem:,.0f}
- Total Put Premium: ${total_put_prem:,.0f}
- Call/Put Ratio: {total_call_prem / max(total_put_prem, 1):.2f}

**Notable Trades:**
"""
            for trade in sorted_flow[:15]:
                trade_type = trade.get('type', 'unknown').upper()
                direction = '📈' if trade.get('direction', 'buy') == 'buy' else '📉'
                is_sweep = '🔥' if trade.get('is_sweep', False) else ''
                is_block = '🐋' if trade.get('premium', 0) > 100000 else ''

                prompt += f"""
{direction} {trade_type} {is_sweep}{is_block}
   Strike: ${trade.get('strike', 0):.0f} | Exp: {trade.get('expiration', 'N/A')}
   Premium: ${trade.get('premium', 0):,.0f} | Vol: {trade.get('volume', 0):,} | OI: {trade.get('open_interest', 0):,}
   Vol/OI: {trade.get('volume', 0) / max(trade.get('open_interest', 1), 1):.1f}x
"""
        else:
            prompt += "\nNo flow data provided - analyze based on general patterns.\n"

        # Historical context
        historical = input_data.get('historical_flow', {})
        if historical:
            prompt += f"""
### Historical Context
- **Avg Daily Call Premium**: ${historical.get('avg_call_premium', 0):,.0f}
- **Avg Daily Put Premium**: ${historical.get('avg_put_premium', 0):,.0f}
- **Typical Vol/OI Range**: {historical.get('typical_vol_oi', '1-3x')}
"""

        prompt += """

### Analysis Tasks

1. **Overall Flow Signal**:
   - very_bullish: Strong call buying, whale accumulation
   - bullish: Net positive call flow
   - neutral: Balanced or unclear
   - bearish: Net negative put flow
   - very_bearish: Strong put buying, distribution

2. **Signal Strength (0-100)**: How confident is the signal?

3. **Unusual Trade Classification**:
   - Which trades are notable?
   - Any sweeps (urgent/aggressive)?
   - Any whale activity (>$100k)?
   - Smart money indicators?

4. **Pattern Detection**:
   - Accumulation (building positions)?
   - Distribution (exiting)?
   - Hedging (protective)?

5. **Key Observations**: Top 3-5 insights from the flow

6. **Trading Implications**:
   - What does this flow suggest?
   - How should traders position?
   - Any strategies aligned with flow?

7. **Caution Factors**: Any concerns or conflicting signals?

Focus on actionable insights for options traders.
"""
        return prompt

    def parse_response(self, response: str, input_data: Dict[str, Any]) -> OptionsFlowOutput:
        """Parse flow analysis response"""
        import json
        import re

        try:
            json_match = re.search(r'\{[\s\S]*\}', response)
            if json_match:
                data = json.loads(json_match.group())
                data['agent_name'] = self.name
                data['symbol'] = input_data.get('symbol', '')
                data['underlying_price'] = input_data.get('underlying_price', 0)
                return OptionsFlowOutput(**data)
        except Exception as e:
            logger.warning(f"Failed to parse JSON: {e}")

        return self._parse_text_response(response, input_data)

    def _parse_text_response(
        self,
        response: str,
        input_data: Dict[str, Any]
    ) -> OptionsFlowOutput:
        """Parse unstructured response"""
        response_lower = response.lower()
        flow_data = input_data.get('flow_data', [])

        # Determine signal
        bullish_indicators = ['bullish', 'call buying', 'accumulation', 'positive']
        bearish_indicators = ['bearish', 'put buying', 'distribution', 'negative']

        bullish_count = sum(1 for i in bullish_indicators if i in response_lower)
        bearish_count = sum(1 for i in bearish_indicators if i in response_lower)

        if bullish_count > bearish_count + 1:
            flow_signal = 'very_bullish'
            signal_strength = 75
        elif bullish_count > bearish_count:
            flow_signal = 'bullish'
            signal_strength = 65
        elif bearish_count > bullish_count + 1:
            flow_signal = 'very_bearish'
            signal_strength = 75
        elif bearish_count > bullish_count:
            flow_signal = 'bearish'
            signal_strength = 65
        else:
            flow_signal = 'neutral'
            signal_strength = 50

        # Build flow summary from data
        total_call = sum(t.get('premium', 0) for t in flow_data if t.get('type') == 'call')
        total_put = sum(t.get('premium', 0) for t in flow_data if t.get('type') == 'put')

        return OptionsFlowOutput(
            agent_name=self.name,
            symbol=input_data.get('symbol', ''),
            underlying_price=input_data.get('underlying_price', 0),
            analysis_timestamp=datetime.now().isoformat(),
            flow_signal=flow_signal,
            signal_strength=signal_strength,
            flow_summary=FlowSummary(
                total_call_premium=total_call,
                total_put_premium=total_put,
                call_put_premium_ratio=total_call / max(total_put, 1)
            ),
            reasoning=response,
            confidence=AgentConfidence.MEDIUM
        )


# =============================================================================
# QUICK FLOW ANALYZER
# =============================================================================

class QuickFlowAnalyzer:
    """Rule-based quick flow analysis"""

    @staticmethod
    def analyze(
        call_premium: float = 0,
        put_premium: float = 0,
        sweep_count: int = 0,
        whale_trades: int = 0,
        avg_vol_oi_ratio: float = 1.0
    ) -> Dict:
        """Quick flow assessment"""
        signals = []
        score = 50  # Neutral start

        # Call/Put ratio
        if call_premium > 0 and put_premium > 0:
            ratio = call_premium / put_premium
            if ratio > 2.0:
                signals.append({"type": "Call/Put", "signal": "very_bullish", "note": f"High call premium ({ratio:.1f}x)"})
                score += 20
            elif ratio > 1.3:
                signals.append({"type": "Call/Put", "signal": "bullish", "note": f"Elevated call premium ({ratio:.1f}x)"})
                score += 10
            elif ratio < 0.5:
                signals.append({"type": "Call/Put", "signal": "very_bearish", "note": f"High put premium ({ratio:.1f}x)"})
                score -= 20
            elif ratio < 0.75:
                signals.append({"type": "Call/Put", "signal": "bearish", "note": f"Elevated put premium ({ratio:.1f}x)"})
                score -= 10

        # Sweep activity (urgency)
        if sweep_count > 5:
            signals.append({"type": "Sweeps", "signal": "high_urgency", "note": f"{sweep_count} sweeps detected"})
            score += 10 if score > 50 else -10  # Amplify existing signal

        # Whale activity
        if whale_trades > 2:
            signals.append({"type": "Whales", "signal": "institutional", "note": f"{whale_trades} whale trades"})
            score += 5 if score > 50 else -5

        # Volume/OI (unusual activity)
        if avg_vol_oi_ratio > 3:
            signals.append({"type": "Vol/OI", "signal": "unusual", "note": f"High activity ({avg_vol_oi_ratio:.1f}x avg)"})

        score = min(100, max(0, score))

        # Determine overall signal
        if score >= 70:
            flow_signal = "very_bullish"
        elif score >= 55:
            flow_signal = "bullish"
        elif score <= 30:
            flow_signal = "very_bearish"
        elif score <= 45:
            flow_signal = "bearish"
        else:
            flow_signal = "neutral"

        return {
            'flow_signal': flow_signal,
            'signal_strength': score,
            'signals': signals
        }


# =============================================================================
# TESTING
# =============================================================================

if __name__ == "__main__":
    import asyncio

    print("\n=== Testing AI Options Flow Agent ===\n")

    async def test_agent():
        # Test quick analyzer
        print("1. Testing Quick Flow Analyzer...")
        analyzer = QuickFlowAnalyzer()

        test_cases = [
            {"call_premium": 5000000, "put_premium": 1000000, "sweep_count": 8, "whale_trades": 3},  # Bullish
            {"call_premium": 500000, "put_premium": 3000000, "sweep_count": 5, "whale_trades": 2},   # Bearish
            {"call_premium": 1000000, "put_premium": 1000000, "sweep_count": 1, "whale_trades": 0},  # Neutral
        ]

        for case in test_cases:
            result = analyzer.analyze(**case)
            print(f"\n   Call=${case['call_premium']:,}, Put=${case['put_premium']:,}:")
            print(f"      Signal: {result['flow_signal']}")
            print(f"      Strength: {result['signal_strength']}")

        # Test agent structure
        print("\n2. Testing Agent Structure...")
        agent = AIOptionsFlowAgent()
        print(f"   Name: {agent.name}")
        print(f"   Output model: {agent.output_model.__name__}")

        print("\n✅ AI Options Flow Agent ready!")

    asyncio.run(test_agent())

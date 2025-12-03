"""
AI-Powered Supply/Demand Zone Agent
====================================

Uses Claude to identify and analyze supply/demand zones
for optimal options entry points.

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

class PriceZone(AgentOutputBase):
    """Individual supply or demand zone"""
    zone_type: str = ""  # supply, demand
    price_low: float = 0.0
    price_high: float = 0.0
    midpoint: float = 0.0

    # Zone characteristics
    strength: str = ""  # weak, moderate, strong, very_strong
    strength_score: int = Field(default=50, ge=0, le=100)
    freshness: str = ""  # fresh (untested), tested, broken
    touches: int = 0

    # Formation
    formation_date: str = ""
    formation_type: str = ""  # base, rally-base-rally, drop-base-drop, etc.
    candle_pattern: str = ""

    # Distance analysis
    distance_from_current: float = 0.0
    distance_percent: float = 0.0

    # Trading relevance
    csp_strike_candidate: bool = False
    cc_strike_candidate: bool = False
    support_level: bool = False
    resistance_level: bool = False

    notes: str = ""


class ZoneAnalysis(AgentOutputBase):
    """Analysis of zone significance"""
    primary_trend: str = ""  # bullish, bearish, sideways
    trend_strength: str = ""

    nearest_demand: Optional[PriceZone] = None
    nearest_supply: Optional[PriceZone] = None

    key_support_levels: List[float] = Field(default_factory=list)
    key_resistance_levels: List[float] = Field(default_factory=list)


class SupplyDemandOutput(AgentOutputBase):
    """Complete supply/demand analysis output"""
    symbol: str = ""
    current_price: float = 0.0
    analysis_timestamp: str = ""

    # Zone lists
    demand_zones: List[PriceZone] = Field(default_factory=list)
    supply_zones: List[PriceZone] = Field(default_factory=list)

    # Analysis
    zone_analysis: ZoneAnalysis = Field(default_factory=ZoneAnalysis)

    # Current position in structure
    price_position: str = ""  # in_demand, in_supply, between_zones, at_support, at_resistance

    # Trading implications
    bullish_bias: bool = False
    bearish_bias: bool = False

    # CSP recommendations
    optimal_csp_zone: Optional[PriceZone] = None
    csp_strike_suggestions: List[float] = Field(default_factory=list)

    # CC recommendations
    optimal_cc_zone: Optional[PriceZone] = None
    cc_strike_suggestions: List[float] = Field(default_factory=list)

    # Entry timing
    buy_opportunities: List[str] = Field(default_factory=list)
    sell_opportunities: List[str] = Field(default_factory=list)

    # Warnings
    warnings: List[str] = Field(default_factory=list)


# =============================================================================
# SUPPLY/DEMAND AGENT
# =============================================================================

class AISupplyDemandAgent(LLMAgent[SupplyDemandOutput]):
    """
    AI-powered supply/demand zone analysis agent.

    Identifies and analyzes supply/demand zones to optimize
    options entry points for wheel strategies.

    Usage:
        agent = AISupplyDemandAgent()
        result = await agent.execute({
            "symbol": "AAPL",
            "current_price": 185.50,
            "price_history": [...],  # OHLCV data
            "timeframe": "daily"
        })
    """

    name = "ai_supply_demand"
    description = "Identifies supply/demand zones for optimal entries"
    output_model = SupplyDemandOutput
    temperature = 0.3

    system_prompt = """You are an expert supply/demand zone analyst specializing in:

1. ZONE IDENTIFICATION:
   - Supply zones (selling pressure areas)
   - Demand zones (buying pressure areas)
   - Rally-Base-Rally (RBR) patterns
   - Drop-Base-Drop (DBD) patterns
   - Rally-Base-Drop (RBD) - supply formation
   - Drop-Base-Rally (DBR) - demand formation

2. ZONE QUALITY ASSESSMENT:
   - Strength scoring (weak to very strong)
   - Freshness (untested vs retested)
   - Width and range analysis
   - Volume confirmation
   - Time spent in base

3. ZONE CHARACTERISTICS:
   - Explosive departure quality
   - Base tightness
   - Number of touches/tests
   - Break patterns

4. OPTIONS TRADING APPLICATION:
   - CSP strike selection at demand zones
   - CC strike selection at supply zones
   - Entry timing optimization
   - Risk management levels

5. TREND CONTEXT:
   - Higher timeframe trend alignment
   - Zone confluence analysis
   - Multiple timeframe confirmation

Your role is to:
1. Identify significant supply/demand zones
2. Score zone quality and strength
3. Map current price to zone structure
4. Recommend optimal CSP/CC strikes based on zones
5. Provide entry timing guidance

Focus on zones that matter for options trading - quality over quantity."""

    def build_prompt(self, input_data: Dict[str, Any]) -> str:
        """Build supply/demand analysis prompt"""
        symbol = input_data.get('symbol', 'UNKNOWN')
        price = input_data.get('current_price', 0)
        price_history = input_data.get('price_history', [])
        timeframe = input_data.get('timeframe', 'daily')

        prompt = f"""## Supply/Demand Zone Analysis: {symbol}

### Current Data
- **Symbol**: {symbol}
- **Current Price**: ${price:.2f}
- **Timeframe**: {timeframe}
- **Analysis Date**: {datetime.now().strftime('%Y-%m-%d')}

### Price History (Recent)
"""

        if price_history:
            # Show recent candles
            for candle in price_history[-20:]:
                date = candle.get('date', 'N/A')
                o = candle.get('open', 0)
                h = candle.get('high', 0)
                l = candle.get('low', 0)
                c = candle.get('close', 0)
                v = candle.get('volume', 0)

                # Candle type
                if c > o:
                    candle_type = "🟢"
                elif c < o:
                    candle_type = "🔴"
                else:
                    candle_type = "⚪"

                prompt += f"{date}: {candle_type} O:{o:.2f} H:{h:.2f} L:{l:.2f} C:{c:.2f} V:{v:,.0f}\n"
        else:
            prompt += "No price history provided.\n"

        # Key levels if provided
        key_levels = input_data.get('key_levels', {})
        if key_levels:
            prompt += f"""
### Known Key Levels
- **52-Week High**: ${key_levels.get('high_52w', 'N/A')}
- **52-Week Low**: ${key_levels.get('low_52w', 'N/A')}
- **SMA 20**: ${key_levels.get('sma_20', 'N/A')}
- **SMA 50**: ${key_levels.get('sma_50', 'N/A')}
- **SMA 200**: ${key_levels.get('sma_200', 'N/A')}
"""

        prompt += f"""

### Analysis Tasks

1. **Identify Demand Zones** (buying pressure - potential support):
   - Look for Drop-Base-Rally (DBR) formations
   - Rally-Base-Rally (RBR) continuations
   - Score each zone 0-100 on strength
   - Note freshness (untested = stronger)

2. **Identify Supply Zones** (selling pressure - potential resistance):
   - Look for Rally-Base-Drop (RBD) formations
   - Drop-Base-Drop (DBD) continuations
   - Score each zone 0-100 on strength
   - Note freshness and quality

3. **Zone Quality Criteria**:
   - Strong explosive move away from zone
   - Tight, clean base formation
   - Minimal wicks into zone
   - High volume on departure

4. **Current Price Position**:
   - Where is ${price:.2f} relative to key zones?
   - Near support/demand?
   - Near resistance/supply?
   - Mid-range between zones?

5. **CSP Strike Recommendations**:
   - Which demand zones are optimal for selling puts?
   - Strike prices that align with strong demand
   - Consider distance vs premium trade-off

6. **CC Strike Recommendations**:
   - Which supply zones are optimal for selling calls?
   - Strike prices that align with strong resistance
   - Probability of being called away

7. **Entry Timing**:
   - Is price approaching a zone?
   - Is this a good time to initiate positions?
   - Should we wait for a zone test?

8. **Warnings**:
   - Any broken zones that may not hold?
   - Trend concerns?
   - Conflicting signals?

Provide specific price levels and actionable recommendations.
"""
        return prompt

    def parse_response(self, response: str, input_data: Dict[str, Any]) -> SupplyDemandOutput:
        """Parse supply/demand analysis response"""
        import json
        import re

        try:
            json_match = re.search(r'\{[\s\S]*\}', response)
            if json_match:
                data = json.loads(json_match.group())
                data['agent_name'] = self.name
                data['symbol'] = input_data.get('symbol', '')
                data['current_price'] = input_data.get('current_price', 0)
                return SupplyDemandOutput(**data)
        except Exception as e:
            logger.warning(f"Failed to parse JSON: {e}")

        return self._parse_text_response(response, input_data)

    def _parse_text_response(
        self,
        response: str,
        input_data: Dict[str, Any]
    ) -> SupplyDemandOutput:
        """Parse unstructured response"""
        import re

        price = input_data.get('current_price', 0)
        response_lower = response.lower()

        # Try to extract price levels mentioned
        price_matches = re.findall(r'\$(\d+(?:\.\d{2})?)', response)
        prices = [float(p) for p in price_matches]

        # Separate into likely demand (below price) and supply (above price)
        demand_levels = sorted([p for p in prices if p < price], reverse=True)[:3]
        supply_levels = sorted([p for p in prices if p > price])[:3]

        # Create zones from extracted prices
        demand_zones = []
        for level in demand_levels:
            zone = PriceZone(
                zone_type="demand",
                price_low=level * 0.99,
                price_high=level * 1.01,
                midpoint=level,
                distance_from_current=price - level,
                distance_percent=((price - level) / price) * 100,
                csp_strike_candidate=True,
                support_level=True
            )
            demand_zones.append(zone)

        supply_zones = []
        for level in supply_levels:
            zone = PriceZone(
                zone_type="supply",
                price_low=level * 0.99,
                price_high=level * 1.01,
                midpoint=level,
                distance_from_current=level - price,
                distance_percent=((level - price) / price) * 100,
                cc_strike_candidate=True,
                resistance_level=True
            )
            supply_zones.append(zone)

        # Determine bias
        bullish_indicators = ['bullish', 'uptrend', 'support', 'demand strong']
        bearish_indicators = ['bearish', 'downtrend', 'resistance', 'supply strong']

        bullish = sum(1 for i in bullish_indicators if i in response_lower)
        bearish = sum(1 for i in bearish_indicators if i in response_lower)

        return SupplyDemandOutput(
            agent_name=self.name,
            symbol=input_data.get('symbol', ''),
            current_price=price,
            analysis_timestamp=datetime.now().isoformat(),
            demand_zones=demand_zones,
            supply_zones=supply_zones,
            bullish_bias=bullish > bearish,
            bearish_bias=bearish > bullish,
            csp_strike_suggestions=demand_levels,
            cc_strike_suggestions=supply_levels,
            reasoning=response,
            confidence=AgentConfidence.MEDIUM
        )


# =============================================================================
# QUICK ZONE FINDER
# =============================================================================

class QuickZoneFinder:
    """Rule-based quick zone detection"""

    @staticmethod
    def find_zones(
        price_history: List[Dict],
        current_price: float,
        lookback: int = 50
    ) -> Dict:
        """Find supply/demand zones from price history"""
        if not price_history or len(price_history) < 10:
            return {
                'demand_zones': [],
                'supply_zones': [],
                'error': 'Insufficient price history'
            }

        recent = price_history[-lookback:] if len(price_history) > lookback else price_history

        demand_zones = []
        supply_zones = []

        # Find swing lows (potential demand)
        for i in range(2, len(recent) - 2):
            low = recent[i].get('low', 0)
            prev_low_1 = recent[i-1].get('low', 0)
            prev_low_2 = recent[i-2].get('low', 0)
            next_low_1 = recent[i+1].get('low', 0)
            next_low_2 = recent[i+2].get('low', 0)

            # Swing low
            if low < prev_low_1 and low < prev_low_2 and low < next_low_1 and low < next_low_2:
                high = recent[i].get('high', 0)
                zone = {
                    'zone_type': 'demand',
                    'price_low': low,
                    'price_high': high,
                    'midpoint': (low + high) / 2,
                    'date': recent[i].get('date', ''),
                    'distance_from_current': current_price - low,
                    'distance_percent': ((current_price - low) / current_price) * 100
                }
                # Only include zones below current price
                if low < current_price:
                    demand_zones.append(zone)

        # Find swing highs (potential supply)
        for i in range(2, len(recent) - 2):
            high = recent[i].get('high', 0)
            prev_high_1 = recent[i-1].get('high', 0)
            prev_high_2 = recent[i-2].get('high', 0)
            next_high_1 = recent[i+1].get('high', 0)
            next_high_2 = recent[i+2].get('high', 0)

            # Swing high
            if high > prev_high_1 and high > prev_high_2 and high > next_high_1 and high > next_high_2:
                low = recent[i].get('low', 0)
                zone = {
                    'zone_type': 'supply',
                    'price_low': low,
                    'price_high': high,
                    'midpoint': (low + high) / 2,
                    'date': recent[i].get('date', ''),
                    'distance_from_current': high - current_price,
                    'distance_percent': ((high - current_price) / current_price) * 100
                }
                # Only include zones above current price
                if high > current_price:
                    supply_zones.append(zone)

        # Sort by proximity
        demand_zones.sort(key=lambda x: x['distance_from_current'])
        supply_zones.sort(key=lambda x: x['distance_from_current'])

        return {
            'demand_zones': demand_zones[:5],  # Top 5 nearest
            'supply_zones': supply_zones[:5],
            'nearest_demand': demand_zones[0] if demand_zones else None,
            'nearest_supply': supply_zones[0] if supply_zones else None
        }

    @staticmethod
    def suggest_csp_strikes(
        demand_zones: List[Dict],
        current_price: float,
        target_otm_pct: float = 0.10
    ) -> List[float]:
        """Suggest CSP strikes based on demand zones"""
        suggestions = []

        # First, add standard OTM target
        standard_strike = round(current_price * (1 - target_otm_pct))
        suggestions.append(standard_strike)

        # Add strikes at demand zone midpoints
        for zone in demand_zones[:3]:
            midpoint = zone.get('midpoint', 0)
            if midpoint > 0 and midpoint < current_price:
                # Round to nearest 0.50 or 1.00
                rounded = round(midpoint * 2) / 2
                if rounded not in suggestions:
                    suggestions.append(rounded)

        return sorted(suggestions, reverse=True)[:5]


# =============================================================================
# TESTING
# =============================================================================

if __name__ == "__main__":
    import asyncio

    print("\n=== Testing AI Supply/Demand Agent ===\n")

    async def test_agent():
        # Generate sample price history
        print("1. Testing Quick Zone Finder...")

        import random
        base_price = 185
        price_history = []

        for i in range(60):
            change = random.uniform(-2, 2)
            base_price += change
            price_history.append({
                'date': f'2024-{10+(i//30):02d}-{(i%30)+1:02d}',
                'open': base_price,
                'high': base_price + random.uniform(0, 3),
                'low': base_price - random.uniform(0, 3),
                'close': base_price + random.uniform(-1, 1),
                'volume': random.randint(1000000, 5000000)
            })

        current_price = base_price

        finder = QuickZoneFinder()
        zones = finder.find_zones(price_history, current_price)

        print(f"   Current Price: ${current_price:.2f}")
        print(f"   Demand Zones Found: {len(zones['demand_zones'])}")
        print(f"   Supply Zones Found: {len(zones['supply_zones'])}")

        if zones['nearest_demand']:
            print(f"   Nearest Demand: ${zones['nearest_demand']['midpoint']:.2f}")
        if zones['nearest_supply']:
            print(f"   Nearest Supply: ${zones['nearest_supply']['midpoint']:.2f}")

        # Test CSP strike suggestions
        print("\n2. Testing CSP Strike Suggestions...")
        strikes = finder.suggest_csp_strikes(zones['demand_zones'], current_price)
        print(f"   Suggested CSP Strikes: {strikes}")

        # Test agent structure
        print("\n3. Testing Agent Structure...")
        agent = AISupplyDemandAgent()
        print(f"   Name: {agent.name}")
        print(f"   Output model: {agent.output_model.__name__}")

        print("\n✅ AI Supply/Demand Agent ready!")

    asyncio.run(test_agent())

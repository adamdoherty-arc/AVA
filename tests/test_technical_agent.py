"""
Test script for TechnicalAnalysisAgent
Tests the complete implementation with a sample stock
"""

import asyncio
import logging
import json
from src.ava.agents.analysis.technical_agent import TechnicalAnalysisAgent

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)


async def test_technical_agent():
    """Test the TechnicalAnalysisAgent"""
    print("=" * 80)
    print("TECHNICAL ANALYSIS AGENT TEST")
    print("=" * 80)

    # Initialize agent
    print("\n1. Initializing TechnicalAnalysisAgent...")
    agent = TechnicalAnalysisAgent()

    print(f"   Agent Name: {agent.name}")
    print(f"   Description: {agent.description}")
    print(f"   Capabilities: {', '.join(agent.get_capabilities())}")

    # Test with AAPL
    test_symbol = "AAPL"
    print(f"\n2. Testing analysis for {test_symbol}...")

    # Create agent state
    state = {
        'input': f'Analyze {test_symbol}',
        'context': {
            'symbol': test_symbol,
            'user_id': 'test_user',
            'platform': 'test'
        },
        'tools': agent.tools,
        'result': None,
        'error': None,
        'metadata': {}
    }

    # Execute agent
    print(f"\n3. Executing technical analysis...")
    result_state = await agent.execute(state)

    # Check for errors
    if result_state.get('error'):
        print(f"\n   ERROR: {result_state['error']}")
        return

    # Display results
    result = result_state.get('result', {})

    print(f"\n4. ANALYSIS RESULTS FOR {test_symbol}")
    print("-" * 80)

    # Current Price
    print(f"\nCurrent Price: ${result.get('current_price', 0):.2f}")
    print(f"Analysis Period: {result.get('analysis_period', 'N/A')}")
    print(f"Data Points: {result.get('data_points', 0)}")

    # Technical Indicators
    print("\n--- TECHNICAL INDICATORS ---")
    indicators = result.get('indicators', {})

    print("\nMoving Averages:")
    mas = indicators.get('moving_averages', {})
    print(f"  SMA-20:  ${mas.get('sma_20', 0):.2f}")
    print(f"  SMA-50:  ${mas.get('sma_50', 0):.2f}")
    if mas.get('sma_200'):
        print(f"  SMA-200: ${mas.get('sma_200', 0):.2f}")

    print("\nRSI:")
    rsi = indicators.get('rsi', {})
    print(f"  Value: {rsi.get('value', 0):.2f}")
    print(f"  Signal: {rsi.get('signal', 'N/A')}")

    print("\nMACD:")
    macd = indicators.get('macd', {})
    print(f"  MACD Line: {macd.get('macd_line', 0):.4f}")
    print(f"  Signal Line: {macd.get('signal_line', 0):.4f}")
    print(f"  Histogram: {macd.get('histogram', 0):.4f}")
    print(f"  Signal: {macd.get('signal', 'N/A')}")

    print("\nBollinger Bands:")
    bb = indicators.get('bollinger_bands', {})
    print(f"  Upper:  ${bb.get('upper', 0):.2f}")
    print(f"  Middle: ${bb.get('middle', 0):.2f}")
    print(f"  Lower:  ${bb.get('lower', 0):.2f}")
    print(f"  Position: {bb.get('position', 'N/A')}")

    print(f"\nTrend: {indicators.get('trend', 'N/A')}")
    print(f"ATR: ${indicators.get('atr', 0):.2f}")

    # Volume Analysis
    print("\n--- VOLUME ANALYSIS ---")
    vol = indicators.get('volume', {})
    print(f"Current Volume: {vol.get('current', 0):,}")
    print(f"Average Volume: {vol.get('average', 0):,}")
    print(f"Volume Ratio: {vol.get('ratio', 0):.2f}x")
    print(f"Volume Signal: {vol.get('signal', 'N/A')}")

    # Support/Resistance
    print("\n--- SUPPORT/RESISTANCE ZONES ---")
    zones = result.get('zones', {})
    print(f"Current Trend: {zones.get('current_trend', 'N/A')}")

    nearest_support = zones.get('nearest_support')
    nearest_resistance = zones.get('nearest_resistance')

    if nearest_support:
        print(f"Nearest Support: ${nearest_support:.2f}")
    else:
        print("Nearest Support: N/A")

    if nearest_resistance:
        print(f"Nearest Resistance: ${nearest_resistance:.2f}")
    else:
        print("Nearest Resistance: N/A")

    # Patterns
    print("\n--- CHART PATTERNS ---")
    patterns = result.get('patterns', {})
    patterns_detected = patterns.get('patterns_detected', [])

    if patterns_detected:
        for pattern in patterns_detected:
            print(f"  - {pattern.get('pattern', 'N/A')}: {pattern.get('description', 'N/A')} ({pattern.get('confidence', 'N/A')} confidence)")
    else:
        print("  No patterns detected")

    # Volume Profile & Order Flow
    print("\n--- VOLUME PROFILE & ORDER FLOW ---")
    volume_analysis = result.get('volume_analysis', {})

    vp = volume_analysis.get('volume_profile', {})
    if 'poc' in vp:
        print(f"POC (Point of Control): ${vp.get('poc', 0):.2f}")
        print(f"VAH (Value Area High): ${vp.get('vah', 0):.2f}")
        print(f"VAL (Value Area Low): ${vp.get('val', 0):.2f}")
        print(f"Position: {vp.get('position', 'N/A')}")
        print(f"Bias: {vp.get('bias', 'N/A')}")
        print(f"Setup Quality: {vp.get('setup_quality', 'N/A')}")

    of = volume_analysis.get('order_flow', {})
    if 'cvd_trend' in of:
        print(f"\nOrder Flow Trend: {of.get('cvd_trend', 'N/A')}")
        if of.get('divergences', 0) > 0:
            print(f"Divergences Detected: {of.get('divergences', 0)}")

    # Smart Money Concepts
    print("\n--- SMART MONEY CONCEPTS ---")
    smart_money = result.get('smart_money', {})

    obs = smart_money.get('order_blocks', [])
    print(f"Order Blocks: {len(obs)}")
    for ob in obs[:3]:
        print(f"  - {ob.get('type', 'N/A')}: {ob.get('price_range', 'N/A')} (Strength: {ob.get('strength', 0)})")

    fvgs = smart_money.get('fair_value_gaps', [])
    print(f"\nFair Value Gaps: {len(fvgs)}")
    for fvg in fvgs[:3]:
        print(f"  - {fvg.get('type', 'N/A')}: {fvg.get('price_range', 'N/A')} ({fvg.get('gap_pct', 0):.2f}%)")

    ms = smart_money.get('market_structure', {})
    if ms:
        print(f"\nMarket Structure:")
        print(f"  Trend: {ms.get('current_trend', 'N/A')}")
        print(f"  BOS: {ms.get('bos_count', 0)}")
        print(f"  CHoCH: {ms.get('choch_count', 0)}")

    # Trading Signal
    print("\n--- TRADING SIGNAL ---")
    signal = result.get('signal', {})
    print(f"Signal: {signal.get('signal', 'N/A')}")
    print(f"Confidence: {signal.get('confidence', 0):.2f}%")
    print(f"Strength: {signal.get('strength', 'N/A')}")
    print(f"Reasoning: {signal.get('reasoning', 'N/A')}")

    # LLM Insights
    print("\n--- LLM INSIGHTS ---")
    llm_insights = result.get('llm_insights', {})
    summary = llm_insights.get('summary', 'N/A')

    if len(summary) > 500:
        print(summary[:500] + "...")
    else:
        print(summary)

    print("\n" + "=" * 80)
    print("TEST COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    asyncio.run(test_technical_agent())

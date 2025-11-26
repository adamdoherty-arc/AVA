# Technical Analysis Agent - Quick Start Guide

## Overview

The TechnicalAnalysisAgent provides comprehensive technical analysis for stocks using advanced indicators, volume profile analysis, Smart Money Concepts, and LLM-powered insights.

---

## Quick Test (30 seconds)

```bash
cd c:\code\Magnus
python test_technical_agent.py
```

Expected output: Complete technical analysis for AAPL with all indicators, zones, patterns, and trading signals.

---

## Basic Usage

### 1. Import and Initialize

```python
from src.ava.agents.analysis.technical_agent import TechnicalAnalysisAgent
import asyncio

agent = TechnicalAnalysisAgent()
```

### 2. Create State

```python
state = {
    'input': 'Analyze AAPL',
    'context': {
        'symbol': 'AAPL',
        'user_id': 'trader001'
    },
    'tools': agent.tools,
    'result': None,
    'error': None,
    'metadata': {}
}
```

### 3. Execute Analysis

```python
async def analyze():
    result_state = await agent.execute(state)
    return result_state['result']

result = asyncio.run(analyze())
```

### 4. Access Results

```python
# Current Price
price = result['current_price']

# Trading Signal
signal = result['signal']['signal']  # BUY/SELL/HOLD
confidence = result['signal']['confidence']  # 0-100
reasoning = result['signal']['reasoning']

# Technical Indicators
rsi = result['indicators']['rsi']['value']
macd_signal = result['indicators']['macd']['signal']
trend = result['indicators']['trend']

# Support/Resistance
nearest_support = result['zones']['nearest_support']
nearest_resistance = result['zones']['nearest_resistance']

# LLM Insights
summary = result['llm_insights']['summary']
```

---

## Key Features

### 1. Technical Indicators
- Moving Averages (SMA-20, 50, 200, EMA-12, 26)
- RSI with signal interpretation
- MACD with histogram
- Bollinger Bands with position
- ATR for volatility
- Volume analysis

### 2. Volume Profile
- POC (Point of Control)
- VAH/VAL (Value Area High/Low)
- Volume bias (Bullish/Bearish/Neutral)

### 3. Order Flow
- CVD (Cumulative Volume Delta)
- Divergence detection

### 4. Smart Money Concepts
- Order Blocks (institutional entry points)
- Fair Value Gaps (FVG)
- Market Structure (BOS/CHoCH)

### 5. LLM Insights
- Pattern recognition
- Entry/exit recommendations
- Risk assessment

---

## Response Structure

```python
{
    'symbol': 'AAPL',
    'current_price': 182.45,
    'timestamp': '2025-01-20T12:00:00',

    'indicators': {
        'moving_averages': {...},
        'rsi': {'value': 62.5, 'signal': 'BULLISH'},
        'macd': {'signal': 'BULLISH', ...},
        'bollinger_bands': {...},
        'atr': 3.25,
        'volume': {...},
        'trend': 'UPTREND'
    },

    'zones': {
        'nearest_support': 178.50,
        'nearest_resistance': 185.00,
        'current_trend': 'BULLISH',
        ...
    },

    'patterns': {
        'patterns_detected': [...],
        'count': 1
    },

    'volume_analysis': {
        'volume_profile': {
            'poc': 180.25,
            'vah': 183.50,
            'val': 177.00,
            'bias': 'NEUTRAL'
        },
        'order_flow': {
            'cvd_trend': 'BULLISH',
            'divergences': 0
        }
    },

    'smart_money': {
        'order_blocks': [...],
        'fair_value_gaps': [...],
        'market_structure': {...}
    },

    'signal': {
        'signal': 'BUY',
        'confidence': 72.5,
        'strength': 'MODERATE',
        'reasoning': '...'
    },

    'llm_insights': {
        'summary': '...'
    }
}
```

---

## Common Patterns

### Pattern 1: Simple Signal Check

```python
async def get_signal(symbol):
    state = {
        'input': f'Analyze {symbol}',
        'context': {'symbol': symbol},
        'tools': agent.tools,
        'result': None,
        'error': None,
        'metadata': {}
    }

    result = await agent.execute(state)

    if result.get('error'):
        return None

    signal = result['result']['signal']
    return f"{signal['signal']} ({signal['confidence']:.1f}%)"
```

### Pattern 2: Check Multiple Timeframes

```python
async def multi_timeframe_analysis(symbol):
    # Short-term (1 month)
    # Mid-term (3 months) - default
    # Long-term (1 year)

    # Currently uses 3 months
    # Can be extended for multiple timeframes
    pass
```

### Pattern 3: Filter by Confidence

```python
def filter_high_confidence_signals(results):
    return [
        r for r in results
        if r['signal']['confidence'] >= 70
        and r['signal']['strength'] in ['STRONG', 'MODERATE']
    ]
```

### Pattern 4: Extract Key Levels

```python
def get_key_levels(result):
    return {
        'current': result['current_price'],
        'support': result['zones']['nearest_support'],
        'resistance': result['zones']['nearest_resistance'],
        'poc': result['volume_analysis']['volume_profile']['poc'],
        'vah': result['volume_analysis']['volume_profile']['vah'],
        'val': result['volume_analysis']['volume_profile']['val']
    }
```

---

## Signal Interpretation

### BUY Signal
- **Strong (75%+)**: High conviction buy, multiple confirming factors
- **Moderate (60-74%)**: Good buy setup, some confirming factors
- **Weak (<60%)**: Weak buy, mixed signals

### SELL Signal
- **Strong (75%+)**: High conviction sell, multiple confirming factors
- **Moderate (60-74%)**: Good sell setup, some confirming factors
- **Weak (<60%)**: Weak sell, mixed signals

### HOLD Signal
- **Mixed signals**, no clear direction
- Wait for confirmation

---

## Error Handling

```python
result = await agent.execute(state)

if result.get('error'):
    print(f"Error: {result['error']}")
    # Handle error
else:
    analysis = result['result']
    # Process analysis
```

---

## Performance Tips

1. **Cache Results**: Store results for recently analyzed symbols
2. **Batch Processing**: Analyze multiple symbols in parallel
3. **Limit LLM Calls**: Skip LLM for quick checks (still get indicators)
4. **Use Specific Timeframes**: Adjust data period based on needs

---

## Requirements

- Python 3.8+
- Internet connection (for yfinance data)
- Ollama running (optional, for LLM insights)

---

## Files

| File | Purpose |
|------|---------|
| `src/ava/agents/analysis/technical_agent.py` | Main agent implementation |
| `test_technical_agent.py` | Test script |
| `docs/technical_agent_implementation.md` | Full documentation |
| `TECHNICAL_AGENT_SUMMARY.md` | Implementation summary |
| `TECHNICAL_AGENT_QUICKSTART.md` | This guide |

---

## Support

- Check `test_technical_agent.py` for working examples
- See `docs/technical_agent_implementation.md` for detailed docs
- Review `TECHNICAL_AGENT_SUMMARY.md` for architecture details

---

## Next Steps

1. **Run Test**: `python test_technical_agent.py`
2. **Try Different Symbols**: Modify test script with your symbols
3. **Integrate with AVA**: Use in chatbot workflows
4. **Customize**: Adjust scoring weights and thresholds
5. **Extend**: Add custom indicators or patterns

---

**Status**: Production Ready ✅
**Last Updated**: January 20, 2025

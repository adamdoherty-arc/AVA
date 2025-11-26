# Portfolio Agent - Quick Start Guide

## 5-Minute Setup

### 1. Prerequisites
```bash
# Ensure environment variables are set
ROBINHOOD_USERNAME=your_username
ROBINHOOD_PASSWORD=your_password
OLLAMA_HOST=http://localhost:11434
```

### 2. Basic Usage

```python
from src.ava.agents.trading.portfolio_agent import PortfolioAgent
import asyncio

async def main():
    # Initialize agent
    agent = PortfolioAgent()

    # Get portfolio summary
    metrics = await agent.get_portfolio_metrics()
    print(f"Total Value: ${metrics.total_value:,.2f}")
    print(f"Net Delta: {metrics.net_delta:.2f}")
    print(f"Risk: {metrics.concentration_risk:.1f}%")

asyncio.run(main())
```

### 3. Common Operations

#### Portfolio Summary
```python
metrics = await agent.get_portfolio_metrics()
print(f"""
Portfolio Summary:
- Value: ${metrics.total_value:,.2f}
- Cash: ${metrics.cash:,.2f}
- P&L: ${metrics.total_pnl:,.2f} ({metrics.total_pnl_pct:.2f}%)
- Positions: {metrics.num_positions}
- Greeks: Δ{metrics.net_delta:.0f} Θ${metrics.net_theta:.2f}/day
""")
```

#### Risk Check
```python
risk = await agent.assess_portfolio_risk()
print(f"Risk Level: {risk.overall_risk_level.upper()}")
print(f"Risk Score: {risk.risk_score}/100")

for rec in risk.recommendations[:3]:
    print(f"→ {rec}")
```

#### AI Recommendations
```python
analysis = await agent.generate_ai_recommendations()
print(analysis)
```

#### Hedging Strategies
```python
hedging = await agent.suggest_hedging_strategies()
print(hedging)
```

### 4. AVA Integration

```python
# In AVA chatbot handler
state = {
    'input': user_message,
    'context': {},
    'tools': [],
    'result': None,
    'error': None,
    'metadata': {}
}

result_state = await portfolio_agent.execute(state)

if result_state['error']:
    return f"Error: {result_state['error']}"
else:
    return result_state['result']
```

### 5. Query Keywords

The agent automatically detects intent from these keywords:

| Intent | Keywords |
|--------|----------|
| Summary | "summary", "overview", "status", "portfolio" |
| Risk | "risk", "assess", "danger", "exposure" |
| Allocation | "sector", "allocation", "diversification" |
| Greeks | "greek", "delta", "theta", "gamma", "vega" |
| Recommendations | "recommend", "suggest", "advice", "what should" |
| Hedging | "hedge", "protect", "insurance" |

**Examples:**
- "What's my portfolio status?" → Summary
- "How risky is my portfolio?" → Risk assessment
- "Should I hedge my positions?" → Hedging strategies
- "What's my delta exposure?" → Greeks analysis

### 6. Testing

```bash
# Run standalone test
python src/ava/agents/trading/portfolio_agent.py

# Run unit tests
pytest tests/test_portfolio_agent.py -v

# Check specific test
pytest tests/test_portfolio_agent.py::TestPortfolioAgent::test_get_portfolio_metrics -v
```

### 7. Troubleshooting

#### Robinhood Login Failed
```python
# Force fresh login
agent._get_robinhood_client().login(force_fresh=True)
```

#### Clear Cache
```python
# Clear all cached data
agent._cache.clear()
agent._cache_expiry.clear()
```

#### Check LLM Connection
```python
# Verify Ollama is running
import requests
response = requests.get('http://localhost:11434/api/tags')
print("Available models:", response.json())
```

### 8. Key Data Structures

#### PortfolioMetrics
```python
metrics.total_value          # Total portfolio value
metrics.cash                 # Cash available
metrics.buying_power         # Buying power
metrics.total_pnl            # Total P&L ($)
metrics.total_pnl_pct        # Total P&L (%)
metrics.num_positions        # Number of positions
metrics.net_delta            # Net delta exposure
metrics.net_theta            # Net theta ($/day)
metrics.concentration_risk   # Concentration (%)
```

#### RiskAssessment
```python
risk.overall_risk_level      # 'low', 'medium', 'high', 'critical'
risk.risk_score              # 0-100
risk.recommendations         # List of recommendations
risk.concentration_issues    # Concentration warnings
risk.greeks_warnings         # Greeks exposure warnings
risk.liquidity_concerns      # Liquidity issues
```

### 9. Performance Tips

1. **Use caching** - Results cached for 5 minutes
2. **Batch queries** - Request multiple analyses at once
3. **Async all the way** - Always use `await` for agent calls
4. **Clear cache sparingly** - Only when you need fresh data

### 10. Advanced Usage

#### Custom Context
```python
context = {
    'market_condition': 'volatile',
    'user_risk_tolerance': 'conservative',
    'strategy_focus': 'wheel'
}

recommendations = await agent.generate_ai_recommendations(context)
```

#### Parallel Execution
```python
# Get multiple analyses at once
results = await asyncio.gather(
    agent.get_portfolio_metrics(),
    agent.assess_portfolio_risk(),
    agent.get_sector_allocation()
)

metrics, risk, allocation = results
```

#### Custom Risk Thresholds
```python
# Modify in agent instance (future feature)
# agent.risk_thresholds['concentration'] = 40  # Lower threshold
```

## Quick Reference

### Import
```python
from src.ava.agents.trading.portfolio_agent import PortfolioAgent
```

### Initialize
```python
agent = PortfolioAgent()
```

### Core Methods
```python
await agent.get_portfolio_metrics()          # Get all metrics
await agent.assess_portfolio_risk()          # Risk analysis
await agent.get_sector_allocation()          # Allocation breakdown
await agent.generate_ai_recommendations()    # AI analysis
await agent.suggest_hedging_strategies()     # Hedging advice
await agent.execute(state)                   # AVA integration
```

### Tools (LangChain)
```python
from src.ava.agents.trading.portfolio_agent import (
    get_portfolio_summary_tool,
    calculate_greeks_exposure_tool
)
```

## Common Patterns

### Daily Portfolio Check
```python
async def daily_check():
    agent = PortfolioAgent()
    metrics = await agent.get_portfolio_metrics()
    risk = await agent.assess_portfolio_risk()

    print(f"Portfolio: ${metrics.total_value:,.2f}")
    print(f"P&L: {metrics.total_pnl_pct:+.2f}%")
    print(f"Risk: {risk.overall_risk_level}")

    if risk.risk_score > 50:
        print("⚠ High risk detected!")
        for rec in risk.recommendations:
            print(f"  - {rec}")
```

### Before Trading
```python
async def pre_trade_check():
    agent = PortfolioAgent()

    # Check current exposure
    metrics = await agent.get_portfolio_metrics()

    print(f"Current Delta: {metrics.net_delta:.0f}")
    print(f"Buying Power: ${metrics.buying_power:,.2f}")

    # Get recommendations
    if abs(metrics.net_delta) > 100:
        hedging = await agent.suggest_hedging_strategies()
        print("\nHedging needed:")
        print(hedging)
```

### Weekly Review
```python
async def weekly_review():
    agent = PortfolioAgent()

    # Comprehensive analysis
    state = {
        'input': 'comprehensive portfolio analysis',
        'context': {},
        'tools': [],
        'result': None,
        'error': None,
        'metadata': {}
    }

    result = await agent.execute(state)

    # Save to report
    with open('weekly_report.txt', 'w') as f:
        f.write(str(result['result']))
```

## Next Steps

1. **Read Full Guide:** `docs/portfolio_agent_guide.md`
2. **Review Tests:** `tests/test_portfolio_agent.py`
3. **Check Implementation:** `src/ava/agents/trading/portfolio_agent.py`
4. **Integrate with AVA:** Add to AVA orchestrator
5. **Customize:** Modify risk thresholds and prompts

## Support

- Documentation: `docs/portfolio_agent_guide.md`
- Implementation: `PORTFOLIO_AGENT_IMPLEMENTATION.md`
- Tests: `tests/test_portfolio_agent.py`
- Source: `src/ava/agents/trading/portfolio_agent.py`

---

**Quick Win:** Run the standalone test to verify everything works:
```bash
python src/ava/agents/trading/portfolio_agent.py
```

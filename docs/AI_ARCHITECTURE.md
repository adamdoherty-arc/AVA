# AVA AI Architecture

## Overview

AVA Trading Platform uses a sophisticated multi-agent AI architecture designed for maximum performance, reliability, and accuracy in options trading analysis.

## Architecture Diagram

```
                    ┌─────────────────────────────────┐
                    │     AI Trading Orchestrator     │
                    │   (Coordinates All AI Agents)   │
                    └────────────────┬────────────────┘
                                     │
          ┌──────────────────────────┼──────────────────────────┐
          │                          │                          │
    ┌─────▼─────┐            ┌───────▼───────┐          ┌───────▼───────┐
    │   Core    │            │   Extended    │          │   Fallback    │
    │  Agents   │            │    Agents     │          │   Checkers    │
    │ (Always)  │            │(Deep Analysis)│          │ (Rule-Based)  │
    └─────┬─────┘            └───────┬───────┘          └───────┬───────┘
          │                          │                          │
    ┌─────┴─────┐            ┌───────┴───────┐          ┌───────┴───────┐
    │ Strategy  │            │ Fundamental   │          │ QuickStrategy │
    │ Sentiment │            │ Options Flow  │          │ QuickRisk     │
    │ Risk      │            │ Earnings      │          │ QuickSentiment│
    └───────────┘            │ Technical     │          └───────────────┘
                             └───────────────┘
```

## Agent Types

### 1. Core Analysis Agents (3 agents, always run)

| Agent | Purpose | Output |
|-------|---------|--------|
| **StrategyRecommendationAgent** | Determines optimal options strategy | Strategy type, strikes, DTE |
| **AIRiskManagementAgent** | Evaluates position risk | Risk score, warnings, limits |
| **AISentimentAgent** | Analyzes market sentiment | Sentiment score, news impact |

### 2. Extended Analysis Agents (4 agents, deep analysis mode)

| Agent | Purpose | Output |
|-------|---------|--------|
| **AIFundamentalAgent** | Financial health analysis | F-score, earnings quality |
| **AIOptionsAnalysisAgent** | Greeks and premium analysis | Greeks sensitivity, optimal strikes |
| **AIOptionsFlowAgent** | Unusual activity detection | Flow signals, institutional moves |
| **AIEarningsAgent** | Earnings impact assessment | Earnings warnings, IV crush |

### 3. Technical Analysis Agents

| Agent | Purpose | Output |
|-------|---------|--------|
| **AITechnicalAgent** | Price action analysis | Trend, S/R levels, patterns |
| **AISupplyDemandAgent** | Zone identification | Entry zones, liquidity pools |

## FREE LLM Provider System (Updated 2025-11-29)

AVA now defaults to **FREE** LLM providers to minimize costs while maintaining high-quality AI analysis.

### Supported Providers

| Provider | Cost | Speed | Models | Use Case |
|----------|------|-------|--------|----------|
| **Groq** | FREE (30 req/min) | Ultra Fast | Llama 3.3 70B | Default - Best balance |
| **Ollama** | FREE (local) | Fast | Llama 3.1, Mistral | Offline / Privacy |
| **HuggingFace** | FREE (limited) | Medium | Llama 2, Mistral | Backup option |
| Anthropic | Paid | Fast | Claude 3.5 Sonnet | Premium analysis |
| OpenAI | Paid | Fast | GPT-4 | Fallback for complex |

### Configuration

```bash
# .env file
AVA_PROVIDER=groq                    # FREE by default!
AVA_DEFAULT_MODEL=llama-3.3-70b-versatile
GROQ_API_KEY=gsk_xxx                 # Get free key at console.groq.com
OLLAMA_HOST=http://localhost:11434   # For local models
HF_TOKEN=hf_xxx                      # HuggingFace token (optional)
```

### Cost Savings

| Provider | Cost per 1M tokens (input/output) | Monthly Savings vs Claude |
|----------|-----------------------------------|---------------------------|
| Groq | $0 / $0 (free tier) | 100% |
| Ollama | $0 / $0 (local) | 100% |
| DeepSeek | $0.14 / $0.28 | 97% |
| Anthropic | $3.00 / $15.00 | - |

## LLM Client Architecture

The centralized `LLMClient` automatically selects the configured provider:

```python
from src.ava.core.llm_engine import LLMClient, LLMProvider
from src.ava.core.config import get_config

# Uses configured FREE provider (Groq by default)
config = get_config()
client = LLMClient(
    provider=LLMProvider.GROQ,  # From AVA_PROVIDER env
    model=config.ai.default_model,
    cache_enabled=True
)

# Generate response
response = await client.generate(
    system="You are a trading assistant.",
    messages=[{"role": "user", "content": "Analyze AAPL"}],
    temperature=0.3
)
```

## LLM Agent Base Class

All AI agents extend the `LLMAgent` base class which provides:

```python
class LLMAgent(Generic[T]):
    """Base class for LLM-powered agents"""

    # Required overrides
    name: str                    # Agent identifier
    description: str             # What the agent does
    output_model: Type[T]        # Pydantic output model
    system_prompt: str           # LLM system instructions

    # Optional settings
    temperature: float = 0.3    # LLM temperature (lower = more consistent)
    cache_enabled: bool = True  # Enable response caching

    # Automatic provider selection
    # Uses config.ai.provider (defaults to Groq FREE)

    # Methods
    def build_prompt(self, input_data: Dict) -> str
    def parse_response(self, response: str, input_data: Dict) -> T
    async def execute(self, input_data: Dict) -> T
```

## Execution Flow

### Standard Analysis (3 agents)
```
Input → StrategyAgent ─┐
     → RiskAgent ──────┼→ Synthesize → TradingDecision
     → SentimentAgent ─┘
```

### Deep Analysis (7 agents)
```
Input → Core Agents (3) ─────────────────┐
     → FundamentalAgent ─┐               │
     → OptionsAnalysisAgent              ├→ Synthesize → TradingDecision
     → OptionsFlowAgent  ┼→ Extended ────┘
     → EarningsAgent ────┘
```

## Fallback System

Each AI agent has a corresponding Quick*Checker for rule-based fallback:

| AI Agent | Fallback Class | Fallback Trigger |
|----------|----------------|------------------|
| StrategyRecommendationAgent | QuickStrategySelector | LLM unavailable |
| AIRiskManagementAgent | QuickRiskChecker | LLM timeout |
| AISentimentAgent | QuickSentimentChecker | API error |
| AIFundamentalAgent | QuickFundamentalChecker | Rate limit |
| AIOptionsAnalysisAgent | QuickOptionsScorer | Exception |
| AIOptionsFlowAgent | QuickFlowAnalyzer | Fallback enabled |
| AIEarningsAgent | QuickEarningsChecker | Any error |
| AITechnicalAgent | QuickTechnicalCalculator | LLM unavailable |
| AISupplyDemandAgent | QuickZoneFinder | API error |

## Caching Strategy

### Multi-Tier Cache
```
Request → Redis Cache (distributed) → Local LRU Cache → LLM API
              │                              │
              └──────── 5 min TTL ───────────┘
```

### Cache Keys
- Agent responses: `agent:{name}:{input_hash}`
- Market data: `market:{symbol}:{interval}`
- Options chains: `chain:{symbol}:{expiry}`

## Scoring System

### Composite Score Calculation

**Standard Mode (3 agents):**
```
composite = (
    strategy_score * 0.40 +
    risk_score * 0.30 +
    sentiment_score * 0.30
)
```

**Deep Analysis Mode (7 agents):**
```
composite = (
    strategy_score * 0.30 +
    risk_score * 0.25 +
    sentiment_score * 0.20 +
    fundamental_score * 0.15 +
    options_score * 0.10
)
```

### Action Thresholds

| Composite Score | Action | Conviction |
|-----------------|--------|------------|
| >= 75 | EXECUTE | High |
| 60-74 | EXECUTE | Medium |
| 45-59 | HOLD | Low |
| < 45 | AVOID | N/A |

## API Client Architecture

### RobustAPIClient Features
- Automatic retry with exponential backoff
- Circuit breaker per host (5 failures = 60s cooldown)
- Token bucket rate limiting (10 req/s default)
- Response caching with TTL
- Both sync and async methods

```python
from src.ava.core import http_get, get_api_client

# Simple usage
data = http_get("https://api.example.com/data", cache=True)

# With custom client
client = get_api_client()
result = await client.async_get(url, params=params)
```

## Data Validation Layer

All external data is validated through Pydantic models:

```python
from src.ava.core import (
    ValidatedOptionData,
    ValidatedStockData,
    ValidatedEarningsData,
    validate_option_chain,
    validate_stock_quote
)

# Validate API response
options = validate_option_chain(raw_api_data)
for opt in options:
    print(f"{opt.symbol}: IV={opt.iv:.1%}, Delta={opt.delta:.2f}")
```

## Configuration

All settings are environment-variable configurable:

```bash
# AI Settings (FREE by default!)
AVA_PROVIDER=groq               # groq (FREE), ollama (FREE), huggingface (FREE), anthropic, openai
AVA_DEFAULT_MODEL=llama-3.3-70b-versatile  # Model to use

# FREE Provider API Keys
GROQ_API_KEY=gsk_xxx            # Get free key at console.groq.com
OLLAMA_HOST=http://localhost:11434  # Local Ollama server
HF_TOKEN=hf_xxx                 # HuggingFace token (optional)
HUGGINGFACE_API_KEY=hf_xxx      # Alternative env var name

# Paid Provider API Keys (optional)
ANTHROPIC_API_KEY=              # Only if using paid Claude
OPENAI_API_KEY=                 # Only if using paid GPT-4

# API Client
API_MAX_RETRIES=3
API_CIRCUIT_FAILURES=5
API_RATE_LIMIT=10.0

# Cache
CACHE_TTL_POSITIONS=30
CACHE_TTL_OPTIONS_CHAIN=60
CACHE_TTL_PREDICTIONS=300

# Prediction Confidence (centralized)
PREDICTION_CONFIDENCE_HIGH=85
PREDICTION_CONFIDENCE_MEDIUM=70
PREDICTION_CONFIDENCE_LOW=55
```

### Quick Start (FREE)

1. Get a free Groq API key at [console.groq.com](https://console.groq.com)
2. Add to `.env`: `GROQ_API_KEY=gsk_your_key_here`
3. Set provider: `AVA_PROVIDER=groq`
4. That's it! AI analysis is now FREE

## Usage Example

```python
from src.ava.agents.orchestrator import AITradingOrchestrator

# Initialize orchestrator
orchestrator = AITradingOrchestrator(
    use_parallel=True,
    cache_enabled=True,
    fallback_enabled=True,
    deep_analysis_enabled=False  # Standard 3-agent mode
)

# Run analysis
decision = await orchestrator.analyze(
    symbol="AAPL",
    underlying_price=175.50,
    iv_rank=65.0,
    trend="bullish",
    portfolio_value=100000,
    risk_tolerance="moderate"
)

# Use the decision
print(f"Action: {decision.action}")
print(f"Strategy: {decision.recommended_strategy}")
print(f"Confidence: {decision.confidence_score}%")
print(f"Warnings: {decision.warnings}")
```

## Performance Metrics

| Metric | Standard Mode | Deep Mode |
|--------|---------------|-----------|
| Agents run | 3 | 7 |
| Avg latency | ~500ms | ~1200ms |
| Cache hit rate | ~70% | ~65% |
| Fallback rate | <5% | <3% |

## File Locations

### Core Infrastructure
- `src/ava/core/llm_engine.py` - **LLM Client (FREE providers)**
- `src/ava/core/config.py` - Configuration system (AVA_PROVIDER)
- `src/ava/core/api_client.py` - Robust API client
- `src/ava/core/http_client.py` - HTTP helpers
- `src/ava/core/data_validation.py` - Data validators
- `src/ava/core/cache.py` - Caching layer
- `src/ava/core/async_utils.py` - Rate limiter, circuit breaker

### AI Agents
- `src/ava/agents/base/llm_agent.py` - Base LLM agent class (auto-FREE)
- `src/ava/agents/orchestrator/ai_trading_orchestrator.py` - Main orchestrator
- `src/ava/agents/analysis/` - Analysis agents
- `src/ava/agents/trading/` - Trading agents

### Updated for FREE Providers (2025-11-29)
- `src/enhanced_sports_predictor.py` - Sports predictions (now FREE)
- `src/ava/research_agent.py` - Research agent (now FREE)
- `src/rag/rag_query_engine.py` - RAG recommendations (now FREE)
- `src/ai/model_clients.py` - Model clients (Groq/DeepSeek added)

### Backend
- `backend/config.py` - Backend settings
- `backend/services/portfolio_service.py` - Portfolio (N+1 optimized)
- `backend/services/` - Business logic
- `backend/routers/` - API endpoints

---

*Last Updated: 2025-11-29 - Added FREE LLM provider system (Groq, Ollama, HuggingFace)*

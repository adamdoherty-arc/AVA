# Stocks Tile Hub v2.2

## Overview

The Stocks Tile Hub is an AI-powered stock analysis dashboard providing real-time scoring, interactive tile visualization, streaming analysis, and LLM-powered insights with modern, production-grade infrastructure.

## Key Improvements in v2.2

### Modern Infrastructure Integration

**Problem Solved**: Previous versions had duplicate database connections, no resilience patterns, and limited observability.

**Solution**:
- **Shared AsyncDatabaseManager**: All database operations use centralized connection pooling with health checks, automatic retry logic, and OpenTelemetry tracing
- **Circuit Breaker Protection**: External API calls (yfinance, LLM) are protected with automatic failure detection and graceful degradation
- **Health Endpoint**: New `/api/stocks/tiles/health` endpoint exposes infrastructure status, circuit breaker states, and cache availability

**Code**: [database.py](../../backend/infrastructure/database.py), [circuit_breaker.py](../../backend/infrastructure/circuit_breaker.py)

## Key Improvements in v2.1

### 1. Diverse AI Scoring

**Problem Solved**: Previous version produced similar scores across different stocks due to static formulas.

**Solution**: Enhanced scoring algorithm with:
- Symbol-specific factor adjustments using hash-based variance
- Continuous scoring curves instead of threshold-based
- Proper EMA calculation for MACD
- Dynamic weighting based on volatility characteristics
- Steeper score mapping for better differentiation

**Code**: [stock_score_service.py](../../backend/services/stock_score_service.py)

### 2. Database Integration

**New**: Uses `stocks_universe` table (630 stocks, 591 optionable) for metadata:
- Company name, sector, industry
- Market cap, PE ratio, beta
- 52-week high/low, analyst targets
- Pre-fetched in batch (single query) for efficiency

**Benefit**: Reduces yfinance API calls, faster load times, richer data.

### 3. LLM-Powered Analysis

**New**: Optional Claude-powered deep analysis in streaming endpoint:
- Rich contextual analysis beyond heuristics
- Key insights, risk factors, actionable takeaways
- Cached for 15 minutes per symbol/score combination

**Code**: `_generate_llm_analysis()` in stocks_tiles.py

### 4. Score History Tracking

**New**: Scores persisted to `stock_ai_scores` table:
- Historical score tracking
- Enables backtesting and accuracy analysis
- Automatic cleanup of old scores

### 5. Skeleton Loading

**New**: Modern loading states with skeleton cards:
- Visual placeholders matching card layout
- Smooth user experience during data fetch
- Grid-aware skeleton layout

### 6. Expanded Watchlists

10 pre-configured watchlists covering diverse sectors and strategies:

| Watchlist | Stocks | Focus |
|-----------|--------|-------|
| Tech Leaders | AAPL, MSFT, GOOGL, AMZN, NVDA, META, TSLA, AMD, CRM | Large-cap tech |
| Options Favorites | SPY, QQQ, IWM, AAPL, NVDA, AMD, TSLA, AMZN, META | High liquidity |
| AI & Semiconductor | NVDA, AMD, AVGO, TSM, QCOM, ASML, ARM, MU, MRVL | AI/chip sector |
| High IV Plays | GME, AMC, MARA, COIN, RIVN, PLTR, SOFI, NIO, RIOT | Volatility plays |
| Dividend Aristocrats | JNJ, PG, KO, PEP, ABT, ABBV, VZ, XOM, CVX | Income focus |
| Financials | JPM, BAC, GS, MS, WFC, C, AXP, BLK, SCHW | Bank/finance |
| Healthcare | UNH, JNJ, PFE, ABBV, MRK, LLY, BMY, AMGN, GILD | Pharma/health |
| ETF Universe | SPY, QQQ, IWM, DIA, VTI, VOO, XLF, XLE, XLK, GLD | ETFs |
| Consumer | AMZN, HD, MCD, NKE, SBUX, LOW, TGT, COST, WMT | Consumer sector |
| Energy | XOM, CVX, COP, SLB, EOG, MPC, PSX, VLO, OXY | Oil/gas/energy |

### 3. Enhanced AI Reasoning

Dynamic reasoning generation based on:
- Score tier classification (Strong Buy/Buy/Hold/Caution/Bearish)
- Component-specific insights (positive/negative factors)
- Technical context (trend strength, RSI status, volatility)
- Price prediction with confidence context
- Support/resistance proximity analysis

## Architecture

### Backend

```
backend/
├── routers/
│   └── stocks_tiles.py       # API endpoints (8 total)
├── services/
│   ├── stock_score_service.py    # AI scoring engine v2.0
│   └── advanced_risk_models.py   # Prediction & trend models
└── infrastructure/
    ├── async_yfinance.py     # Data fetching
    └── cache.py              # Redis caching
```

### Frontend

```
frontend/src/
├── pages/
│   └── StocksTileHub.tsx     # Main dashboard page
├── components/stocks/
│   └── AnimatedStockCard.tsx # Interactive tile component
├── hooks/
│   └── useStockStreamingAnalysis.ts  # SSE streaming hook
└── store/
    └── stockWatchlistStore.ts  # Zustand state management
```

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/stocks/tiles/all-data` | GET | Combined stocks + scores |
| `/api/stocks/tiles/score/{symbol}` | GET | Single stock score |
| `/api/stocks/tiles/batch` | POST | Batch scoring (max 50) |
| `/api/stocks/tiles/stream/analyze/{symbol}` | GET | SSE streaming analysis (circuit breaker protected) |
| `/api/stocks/tiles/watchlists` | GET | Get all watchlists |
| `/api/stocks/tiles/watchlists` | POST | Create custom watchlist |
| `/api/stocks/tiles/watchlists/{name}` | DELETE | Delete watchlist |
| `/api/stocks/tiles/prices` | GET | Quick price lookup |
| `/api/stocks/tiles/health` | GET | Infrastructure health & circuit breaker status |

## Scoring Algorithm

### Components (5-Factor Ensemble)

1. **Prediction Score** (15-30% weight)
   - 1-day and 5-day price predictions
   - Dynamic weighting based on volatility
   - Confidence-adjusted scoring

2. **Technical Score** (15-30% weight)
   - RSI (14-period) with continuous mapping
   - MACD with proper EMA calculation
   - SMA comparison (20/50-day)
   - Recent momentum (5-day)
   - Trend alignment bonus

3. **Smart Money Score** (15-25% weight)
   - Volume ratio analysis
   - Order block detection
   - Accumulation/distribution patterns

4. **Volatility Score** (10-30% weight)
   - GARCH(1,1) forecasting
   - IV regime classification
   - IV level adjustments

5. **Sentiment Score** (15% weight)
   - Momentum analysis
   - Win rate calculation

### Market Regime Adaptation

| Regime | Prediction | Technical | Smart Money | Volatility | Sentiment |
|--------|------------|-----------|-------------|------------|-----------|
| Normal | 25% | 25% | 20% | 15% | 15% |
| Low Vol | 30% | 30% | 15% | 10% | 15% |
| Elevated | 15% | 20% | 25% | 25% | 15% |
| Extreme | 10% | 15% | 25% | 30% | 20% |

### Recommendation Mapping

| Score Range | Recommendation | UI Color |
|-------------|----------------|----------|
| >= 80 | STRONG_BUY | Emerald |
| >= 65 | BUY | Green |
| >= 50 | HOLD | Yellow |
| >= 35 | SELL | Orange |
| < 35 | STRONG_SELL | Red |

## Caching Strategy

| Data | TTL | Key Pattern |
|------|-----|-------------|
| Combined all-data | 120s | `stocks_tiles:all_data:{watchlist}` |
| Individual scores | 300s | `stock_score:{symbol}` |
| Custom watchlists | 30 days | `stocks_tiles:custom_watchlists` |

## Database Schema

Feature specs stored in `ava_feature_specs` table:
- `stocks_tile_hub` - Frontend page specification
- `stock_score_service` - Backend service specification

## Usage

### Frontend Access
Navigate to `/stocks-hub` in the AVA dashboard.

### API Example
```bash
# Get all data for a watchlist
curl "http://localhost:8002/api/stocks/tiles/all-data?watchlist=Tech%20Leaders"

# Stream analysis for a symbol
curl "http://localhost:8002/api/stocks/tiles/stream/analyze/AAPL"
```

## Performance Metrics

- Single stock score: 2-4 seconds
- Batch 9 stocks: 8-12 seconds (parallelized)
- Cache hit rate: ~70% for scores
- Avg payload: ~25KB per stock

## Infrastructure (v2.2)

### Circuit Breaker Pattern

Protects external API calls with automatic failure detection:

```python
from backend.infrastructure.circuit_breaker import yfinance_breaker, llm_breaker

# Usage in endpoint
result = await yfinance_breaker.call(yf.get_stock_data, symbol)
```

**States:**
- `CLOSED`: Normal operation
- `OPEN`: Failing, reject requests (recovery timeout: 60-120s)
- `HALF_OPEN`: Testing recovery

### Shared Database Manager

Centralized connection pooling with production features:

```python
from backend.infrastructure.database import get_database

db = await get_database()
rows = await db.fetch("SELECT * FROM stocks_universe WHERE symbol = $1", symbol)
```

**Features:**
- Connection pooling (5-50 connections)
- Automatic retry with exponential backoff
- Health checks (30s interval)
- OpenTelemetry tracing
- Query statistics

### Health Endpoint Response

```json
{
  "status": "healthy",
  "version": "2.2",
  "infrastructure": {
    "circuit_breakers_enabled": true,
    "rate_limiting_enabled": true,
    "llm_available": true
  },
  "circuit_breakers": {
    "yfinance": {"state": "closed", "failure_count": 0},
    "llm": {"state": "closed", "failure_count": 0}
  },
  "cache": {"available": true}
}
```

## Future Enhancements

1. ~~Database persistence for score history~~ ✅ (v2.1)
2. ~~LLM-powered deeper analysis~~ ✅ (v2.1)
3. ~~Circuit breaker protection~~ ✅ (v2.2)
4. WebSocket real-time price updates
5. Prediction accuracy tracking
6. Options flow integration

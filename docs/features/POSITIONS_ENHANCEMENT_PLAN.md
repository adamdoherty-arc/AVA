# Positions Page Enhancement Plan

## Issues Identified

### 1. Duplicate AI Research Buttons
**Problem**: If you have the same stock in multiple positions (e.g., AAPL stock + AAPL CSP), the AI Research button appears multiple times.

**Current Structure**:
- Stock positions show AI Research buttons (lines 397-411)
- Each option strategy section shows AI Research buttons (lines 629-652)
- If AAPL appears in both, user sees 2 AI Research buttons

**Solution**:
- Collect all unique symbols across ALL position types
- Display consolidated AI Research section once
- Use set() to deduplicate symbols
- Global session state for research display

---

## New Features to Implement

### 2. Theta Decay Forecasting for Cash-Secured Puts

**What**: Day-by-day theta decay forecast showing projected cash value until expiration

**Features**:
- **Daily Theta Decay Calculation**:
  - Calculate theta (time decay) per day
  - Project P/L from today until expiration
  - Show cumulative cash value gain

- **Visual Chart**:
  - Line chart showing P/L projection over time
  - Current value marker
  - Breakeven line
  - Expiration date marker

- **Data Table**:
  - Date, Days Remaining, Theta Decay, Projected Value, Cumulative P/L
  - Export to CSV option

**Formula**:
```
Daily Theta = Option Theta × Days to Expiration
Projected Value = Current Premium - (Theta × Days Passed)
Cash Gain = (Entry Premium - Projected Value) × Quantity × 100
```

**Implementation**:
- Use Black-Scholes model for accurate theta calculation
- Account for weekends/holidays (market closed days)
- Update daily with latest market data

---

### 3. External Links & Resources

**Links to Add for Each Position**:

#### Company Information:
- **Yahoo Finance**: `https://finance.yahoo.com/quote/{symbol}`
- **TradingView**: `https://www.tradingview.com/symbols/{symbol}`
- **Finviz**: `https://finviz.com/quote.ashx?t={symbol}`
- **SEC Filings (EDGAR)**: `https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&CIK={symbol}`

#### Research & Analysis:
- **Seeking Alpha**: `https://seekingalpha.com/symbol/{symbol}`
- **MarketWatch**: `https://www.marketwatch.com/investing/stock/{symbol}`
- **Bloomberg**: `https://www.bloomberg.com/quote/{symbol}:US`
- **Earnings Whispers**: `https://www.earningswhispers.com/stocks/{symbol}`

#### Options-Specific:
- **Options Chain (Yahoo)**: `https://finance.yahoo.com/quote/{symbol}/options`
- **Barchart Options**: `https://www.barchart.com/stocks/quotes/{symbol}/options`
- **CBOE Options Data**: For volatility metrics

#### News & Sentiment:
- **Google News**: `https://news.google.com/search?q={symbol}+stock`
- **Finnhub News API**: Real-time company news
- **Polygon News API**: Market news and sentiment
- **Reddit r/wallstreetbets**: Community sentiment

---

### 4. Real-Time News Integration

**News Sources** (Already have API keys):
- **Finnhub API** (`FINNHUB_API_KEY`): Company news, sentiment
- **Polygon API** (`POLYGON_API_KEY`): Market news, filings

**Display Format**:
```
📰 Latest News for AAPL:
- [2 hours ago] Apple announces new iPhone - Seeking Alpha
- [5 hours ago] AAPL upgraded by Goldman Sachs - Bloomberg
- [1 day ago] Q4 Earnings beat expectations - MarketWatch
```

**Features**:
- Last 5-10 news articles per symbol
- Timestamp (relative time)
- Source indication
- Sentiment indicator (🟢 positive, 🔴 negative, ⚪ neutral)
- Click to open full article

---

## Implementation Plan

### Phase 1: Deduplicate AI Research (30 min)
1. Create `get_all_unique_symbols()` function
2. Collect symbols from:
   - Stock positions
   - CSP positions
   - CC positions
   - Long Call positions
   - Long Put positions
3. Create single consolidated AI Research section
4. Update session state management

### Phase 2: External Links Section (30 min)
1. Create `generate_external_links()` function
2. Add links section to each position table
3. Organize by category:
   - Company Info (Yahoo, TradingView, Finviz)
   - Research (Seeking Alpha, MarketWatch)
   - Options (Options Chain, Barchart)
   - News (Google News, Finnhub)

### Phase 3: News Integration (45 min)
1. Create `fetch_finnhub_news(symbol)` function
2. Create `fetch_polygon_news(symbol)` function
3. Implement caching (Redis) for news (30-min TTL)
4. Add news section to positions page
5. Format with timestamps and sentiment

### Phase 4: Theta Decay Forecasting (90 min)
1. Create `calculate_theta_decay_forecast()` function
2. Implement Black-Scholes theta calculation
3. Generate day-by-day projection data
4. Create Plotly line chart visualization
5. Create data table with daily projections
6. Add expandable section to CSP positions
7. Add "Export Forecast" button (CSV download)

### Phase 5: UI Polish & Testing (30 min)
1. Organize page layout with tabs:
   - **Active Positions** (existing)
   - **Theta Decay Forecasts** (new)
   - **News & Research** (new)
   - **Trade History** (existing)
2. Add loading spinners
3. Error handling for API failures
4. Test with real positions

---

## Data Models

### Theta Decay Forecast
```python
@dataclass
class ThetaDecayForecast:
    symbol: str
    strike: float
    expiration: date
    current_premium: float
    entry_premium: float
    quantity: int

    # Daily projections
    dates: List[date]
    days_remaining: List[int]
    theta_values: List[float]
    projected_values: List[float]
    cumulative_pnl: List[float]

    # Summary
    total_theta_decay: float
    max_profit: float
    breakeven_date: Optional[date]
```

### News Article
```python
@dataclass
class NewsArticle:
    symbol: str
    headline: str
    source: str
    url: str
    published_at: datetime
    summary: str
    sentiment: str  # 'positive', 'negative', 'neutral'
    relevance_score: float  # 0-1
```

---

## UI Mockup

```
=== POSITIONS PAGE ===

┌─────────────────────────────────────────────────────────┐
│ [Stock Positions] [Options] [Theta Forecasts] [News]    │
└─────────────────────────────────────────────────────────┘

📊 Cash-Secured Puts (4 positions)
┌──────────────────────────────────────────────────────────────┐
│ Symbol  Strike  Exp    Premium  P/L   Delta  Theta  Chart   │
│ AAPL    $180   12/20  $3.50    +15%  -0.30  -0.12  📈      │
│ TSLA    $240   12/27  $5.20    +8%   -0.28  -0.18  📈      │
└──────────────────────────────────────────────────────────────┘

🔗 Quick Links: [Yahoo Finance] [Options Chain] [News] [Earnings]

📉 Theta Decay Forecast (Click to expand)
  ↓ [Show AAPL $180 CSP Daily Projection]

📰 Latest News:
  • [2h ago] 🟢 Apple announces new product line - MarketWatch
  • [5h ago] ⚪ AAPL trading sideways - Bloomberg

🤖 AI Research: [AAPL] [TSLA] [NVDA] [AMD]
```

---

## Success Criteria

✅ No duplicate AI Research buttons across sections
✅ Theta decay forecast with daily projections
✅ Visual chart showing P/L trajectory
✅ External links to research resources
✅ Real-time news integration with sentiment
✅ Fast load times (<2 seconds)
✅ Mobile-responsive design
✅ Error handling for API failures

---

## APIs Used

- **Finnhub**: Company news, sentiment analysis
- **Polygon**: Market news, filings, insider trades
- **Alpha Vantage**: Financial data (already integrated)
- **yfinance**: Options data, Greeks calculations
- **Black-Scholes Model**: Theta calculations

---

## Timeline

**Total Estimated Time**: 3.5 hours

1. Phase 1 (Dedupe): 30 min ✓
2. Phase 2 (Links): 30 min ✓
3. Phase 3 (News): 45 min ✓
4. Phase 4 (Theta): 90 min ✓
5. Phase 5 (Polish): 30 min ✓

**Start**: Now
**Completion**: ~3.5 hours

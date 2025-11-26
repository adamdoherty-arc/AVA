# Prediction Markets Feature - Implementation Plan

## Executive Summary

Add a new **Prediction Markets** feature to Magnus dashboard that:
- Integrates with Kalshi API to pull all available event contracts
- Uses AI to rate opportunities (0-100 score)
- Shows best trades across Politics, Sports, Economics, and more
- Complements options trading with event contract opportunities

---

## Feature Overview

### **What It Does:**
Displays prediction market opportunities from Kalshi (the same markets Robinhood shows) with AI-powered ratings to identify the best trading opportunities.

### **Categories Covered:**
1. **Politics** - Elections, legislation, appointments
2. **Sports** - NFL, NBA, MLB, MMA, Tennis, Golf
3. **Economics** - Fed rates, CPI, GDP, recession, gas prices
4. **Crypto** - Bitcoin price levels, ETF approvals
5. **Companies** - Earnings, M&A, product launches
6. **Tech & Science** - AI milestones, space launches
7. **Climate** - Temperature records, climate events
8. **World** - International events, geopolitics

---

## Architecture

### **Tech Stack:**
```
Frontend:    Streamlit UI (prediction_markets_page.py)
Backend:     Kalshi API integration (src/kalshi_integration.py)
AI Engine:   OpenAI GPT-4 for opportunity scoring (src/prediction_market_analyzer.py)
Database:    PostgreSQL (prediction_markets table)
Caching:     Redis or session state for API rate limiting
```

### **Data Flow:**
```
1. Kalshi API → Pull all active markets (no auth needed for market data)
2. PostgreSQL → Cache market data (refresh every 5 minutes)
3. AI Analyzer → Score each market based on multiple factors
4. UI Display → Show top opportunities sorted by AI score
5. Trade Execution → Link to Robinhood app (can't execute via API)
```

---

## AI Rating System (0-100 Score)

### **Scoring Factors:**

**1. Probability Edge (35%)**
- Mispriced contracts (market price vs true probability)
- Information asymmetry opportunities
- Recent odds movements suggesting value

**2. Liquidity (25%)**
- Volume in last 24 hours
- Order book depth
- Bid-ask spread tightness

**3. Time Value (20%)**
- Days until resolution
- Time decay vs premium collected
- Event timing (sooner = more certain)

**4. Risk-Reward Ratio (15%)**
- Max profit vs max loss
- Probability-weighted expected value
- Sharpe ratio calculation

**5. Market Sentiment (5%)**
- Trend direction (bullish/bearish)
- Social media buzz
- News catalyst analysis

### **AI Prompt Template:**
```
You are a prediction market analyst. Analyze this event contract:

Market: {title}
Category: {category}
Current Price: {yes_price} Yes / {no_price} No
Volume (24h): ${volume}
Close Date: {close_date}
Description: {description}

Recent news/context: {news_summary}

Score this opportunity 0-100 based on:
1. Probability mispricing (is the market wrong?)
2. Liquidity (can you enter/exit easily?)
3. Time value (risk vs time to resolution)
4. Risk-reward (expected value calculation)
5. Catalysts (upcoming events that could move price)

Provide:
- Overall Score (0-100)
- Reasoning (2-3 sentences)
- Recommended Position (Yes/No/Skip)
- Risk Level (Low/Medium/High)
```

---

## Database Schema

```sql
CREATE TABLE prediction_markets (
    id SERIAL PRIMARY KEY,
    ticker VARCHAR(50) UNIQUE NOT NULL,          -- Kalshi ticker (e.g., PRES-2024)
    title VARCHAR(500) NOT NULL,                 -- Market title
    category VARCHAR(50),                         -- Politics, Sports, etc.
    subcategory VARCHAR(50),                      -- Presidential, NFL, etc.

    -- Pricing
    yes_price DECIMAL(10, 4),                    -- Current Yes price (0-1)
    no_price DECIMAL(10, 4),                     -- Current No price (0-1)
    yes_bid DECIMAL(10, 4),
    yes_ask DECIMAL(10, 4),
    no_bid DECIMAL(10, 4),
    no_ask DECIMAL(10, 4),

    -- Volume & Liquidity
    volume_24h INTEGER,                          -- Contracts traded (24h)
    open_interest INTEGER,                       -- Total open contracts
    bid_ask_spread DECIMAL(10, 4),              -- Spread in cents

    -- Timing
    open_date TIMESTAMP,
    close_date TIMESTAMP,
    resolution_date TIMESTAMP,
    days_to_close INTEGER,

    -- AI Scoring
    ai_score DECIMAL(5, 2),                     -- 0-100 rating
    ai_reasoning TEXT,                           -- Why this score
    recommended_position VARCHAR(10),            -- Yes/No/Skip
    risk_level VARCHAR(20),                      -- Low/Medium/High

    -- Metadata
    description TEXT,
    rules TEXT,
    market_status VARCHAR(20),                   -- active, closed, settled
    last_updated TIMESTAMP DEFAULT NOW(),

    CONSTRAINT price_range CHECK (yes_price BETWEEN 0 AND 1),
    CONSTRAINT spread_positive CHECK (bid_ask_spread >= 0)
);

CREATE INDEX idx_ai_score ON prediction_markets(ai_score DESC);
CREATE INDEX idx_category ON prediction_markets(category);
CREATE INDEX idx_close_date ON prediction_markets(close_date);
CREATE INDEX idx_status ON prediction_markets(market_status);
```

---

## Implementation Steps

### **Phase 1: Core Integration (Week 1)**

**Step 1.1: Install Dependencies**
```bash
pip install kalshi-python openai requests
```

**Step 1.2: Create Kalshi Integration Module**
File: `src/kalshi_integration.py`
- Authenticate with Kalshi API
- Fetch all active markets
- Get market orderbook data
- Cache results to reduce API calls

**Step 1.3: Database Setup**
- Create prediction_markets table
- Add sync script for daily updates
- Implement caching layer

### **Phase 2: AI Scoring Engine (Week 1)**

**Step 2.1: Build Analyzer**
File: `src/prediction_market_analyzer.py`
- Connect to OpenAI API
- Implement scoring algorithm
- Batch process markets (100 at a time)
- Cache AI scores for 6 hours

**Step 2.2: News Integration** (Optional Enhancement)
- Pull recent news for each market category
- Use NewsAPI or Google News
- Feed context to AI for better scoring

### **Phase 3: UI Development (Week 1)**

**Step 3.1: Create Main Page**
File: `prediction_markets_page.py`
- Display top 20 opportunities
- Filter by category
- Sort by AI score, liquidity, time
- Show detailed market info on click

**Step 3.2: Dashboard Integration**
- Add navigation button
- Update dashboard.py routing
- Add to sidebar menu

**Step 3.3: Visualizations**
- Price charts (historical if available)
- Volume charts
- AI score distribution
- Category breakdown pie chart

### **Phase 4: Documentation & Testing (Week 1)**

**Step 4.1: Feature Documentation**
Create `features/prediction_markets/` folder:
- README.md
- ARCHITECTURE.md
- SPEC.md
- WISHLIST.md
- tests/

**Step 4.2: Testing**
- Unit tests for Kalshi integration
- AI scoring accuracy validation
- UI responsiveness testing
- Load testing (100+ markets)

---

## File Structure

```
WheelStrategy/
├── src/
│   ├── kalshi_integration.py          # Kalshi API wrapper
│   ├── prediction_market_analyzer.py  # AI scoring engine
│   └── prediction_market_sync.py      # Daily sync script
├── prediction_markets_page.py         # Streamlit UI
├── database_schema_predictions.sql    # DB schema
├── features/
│   └── prediction_markets/
│       ├── README.md
│       ├── ARCHITECTURE.md
│       ├── SPEC.md
│       ├── WISHLIST.md
│       └── tests/
│           └── test_prediction_markets.py
└── PREDICTION_MARKETS_IMPLEMENTATION_PLAN.md
```

---

## API Usage & Costs

### **Kalshi API:**
- **Cost**: FREE for market data (no authentication required)
- **Rate Limit**: 100 requests/minute (generous for our use)
- **Authentication**: Only needed for trading (not viewing markets)

### **OpenAI API (for AI scoring):**
- **Model**: GPT-4-turbo
- **Cost**: ~$0.01 per market analysis
- **Usage**: Score 100 markets = $1.00
- **Frequency**: Update scores every 6 hours
- **Monthly Cost**: ~$120/month for continuous scoring

### **Alternative: Use Free AI**
- Use local Ollama with Llama 3 (FREE)
- Slightly lower quality but $0 cost
- Good enough for scoring

---

## UI Mockup

```
╔══════════════════════════════════════════════════════════════╗
║  🎲 PREDICTION MARKETS - AI-Powered Opportunities           ║
╠══════════════════════════════════════════════════════════════╣
║                                                              ║
║  Filters: [All Categories ▼] [Min Score: 70] [Refresh]     ║
║                                                              ║
║  ┌────────────────────────────────────────────────────────┐ ║
║  │ AI Score: 94 🔥 | Politics | 5 days to close           │ ║
║  │ Presidential Election 2024 Winner                       │ ║
║  │                                                          │ ║
║  │ Yes: $0.52 | No: $0.48 | Volume: $2.4M | Spread: 1¢    │ ║
║  │                                                          │ ║
║  │ AI Analysis: Strong mispricing based on recent polls.  │ ║
║  │ Expected value calculation shows 15% edge on Yes side. │ ║
║  │ Recommended: BUY YES @ $0.52                           │ ║
║  │ Risk: Medium | Max Profit: $48 per $100 invested      │ ║
║  │                                                          │ ║
║  │ [View on Robinhood] [Detailed Analysis]                │ ║
║  └────────────────────────────────────────────────────────┘ ║
║                                                              ║
║  ┌────────────────────────────────────────────────────────┐ ║
║  │ AI Score: 89 🔥 | Sports | 2 days to close             │ ║
║  │ NFL: Chiefs to win Super Bowl                          │ ║
║  │ ...                                                      │ ║
║  └────────────────────────────────────────────────────────┘ ║
║                                                              ║
╚══════════════════════════════════════════════════════════════╝
```

---

## Success Metrics

**Week 1 Goals:**
- ✅ Kalshi integration pulling 100+ markets
- ✅ AI scoring engine rating all markets
- ✅ UI displaying top 20 opportunities
- ✅ Database storing market data
- ✅ Feature documented and tested

**Future Enhancements:**
- Portfolio tracking (track your positions)
- Profit/loss history
- Backtesting AI scores vs actual outcomes
- Multi-market arbitrage detection
- Mobile alerts for high-score opportunities
- Integration with Polymarket for comparison

---

## Risk Considerations

**1. API Changes**
- Kalshi could modify API without notice
- Build error handling and logging

**2. AI Accuracy**
- AI scores are predictions, not guarantees
- Display disclaimers
- Track accuracy over time

**3. Regulatory**
- Prediction markets are regulated by CFTC
- Only show markets, don't execute trades via API
- Link to Robinhood app for actual trading

**4. Costs**
- OpenAI API costs could grow
- Implement usage limits
- Consider free alternatives (Ollama)

---

## Next Steps

**Approve this plan?**

Once approved, I will:
1. Install Kalshi Python SDK
2. Build the integration module
3. Create AI scoring engine
4. Build the UI page
5. Set up database schema
6. Add to dashboard navigation
7. Create full documentation
8. Test and commit to GitHub

**Timeline: 1 week for full implementation**

---

*This feature will make Magnus the ONLY trading dashboard combining options trading + prediction markets with AI-powered recommendations!*

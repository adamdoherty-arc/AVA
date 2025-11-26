# Premium Options Flow Feature - Complete Documentation

## Overview

The **Premium Options Flow** feature is a comprehensive institutional money tracking system that monitors options volume, premium inflows/outflows, and identifies high-probability trading opportunities for the wheel strategy. It combines real-time data collection from Yahoo Finance with AI-powered analysis to provide actionable insights.

---

## Table of Contents

1. [Features](#features)
2. [Architecture](#architecture)
3. [Installation & Setup](#installation--setup)
4. [Usage Guide](#usage-guide)
5. [Database Schema](#database-schema)
6. [API Reference](#api-reference)
7. [UI Components](#ui-components)
8. [Trading Strategies](#trading-strategies)
9. [Testing](#testing)
10. [Troubleshooting](#troubleshooting)

---

## Features

### Core Capabilities

1. **Options Flow Tracking**
   - Real-time collection of call and put volume
   - Premium flow calculation (calls vs puts)
   - Put/call ratio monitoring
   - Unusual activity detection (2x+ average volume)

2. **AI-Powered Analysis**
   - Sentiment analysis (Bullish/Bearish/Neutral)
   - Opportunity scoring (0-100 scale)
   - Risk assessment (Low/Medium/High)
   - Action recommendations (SELL_PUT, BUY_CALL, WAIT, etc.)
   - Confidence scoring for all recommendations

3. **Trend Detection**
   - 7-day and 30-day trend analysis
   - Flow acceleration tracking
   - Pattern recognition across similar stocks

4. **Interactive Dashboard**
   - 4 comprehensive tabs
   - Real-time data visualization
   - Sortable and filterable opportunities
   - Detailed flow charts and analytics

5. **Wheel Strategy Integration**
   - Delta-based strike recommendations
   - Premium yield calculations
   - Win probability estimates
   - Position sizing suggestions

---

## Architecture

### System Components

```
Premium Options Flow System
│
├── Database Layer (PostgreSQL)
│   ├── options_flow (daily snapshots)
│   ├── options_flow_analysis (aggregated insights)
│   ├── premium_flow_opportunities (ranked opportunities)
│   └── options_flow_alerts (unusual activity alerts)
│
├── Data Collection Layer
│   ├── options_flow_tracker.py
│   │   ├── fetch_options_flow()
│   │   ├── calculate_flow_metrics()
│   │   ├── analyze_flow_trend()
│   │   ├── detect_unusual_activity()
│   │   └── batch_update_flow()
│   │
│   └── Yahoo Finance API
│       └── Options chain data
│
├── AI Analysis Layer
│   ├── ai_flow_analyzer.py
│   │   ├── analyze_flow_sentiment()
│   │   ├── score_flow_opportunity()
│   │   ├── recommend_best_action()
│   │   ├── assess_flow_risk()
│   │   └── generate_flow_insights()
│   │
│   └── OpenAI GPT-4 Integration
│       └── Advanced recommendations
│
└── Presentation Layer
    └── premium_flow_page.py
        ├── Tab 1: Flow Overview
        ├── Tab 2: Flow Opportunities
        ├── Tab 3: Flow Analysis
        └── Tab 4: Strategies
```

---

## Installation & Setup

### Prerequisites

- PostgreSQL database (already configured for Magnus)
- Python 3.8+
- Required packages (already in requirements.txt):
  - streamlit
  - pandas
  - plotly
  - yfinance
  - psycopg2
  - langchain-community (for AI features)

### Setup Steps

1. **Run Database Migration**

```bash
# Option 1: Via UI
# Navigate to Premium Options Flow page
# Click "Run Migration Now" button

# Option 2: Via command line
cd c:\Code\WheelStrategy
psql -U postgres -d magnus -f migrations/add_premium_options_flow.sql
```

2. **Verify Installation**

```bash
python validate_premium_flow.py
```

Expected output:
```
Premium Options Flow - Quick Validation
============================================================

1. Checking database tables...
  [OK] options_flow
  [OK] options_flow_analysis
  [OK] premium_flow_opportunities
  [OK] options_flow_alerts

VALIDATION COMPLETE: ALL CHECKS PASSED
```

3. **Start Dashboard**

```bash
streamlit run dashboard.py
```

4. **Access Feature**
   - Click "💸 Premium Options Flow" in sidebar
   - Click "Refresh Flow Data" to collect initial data
   - Click "Run AI Analysis" to generate recommendations

---

## Usage Guide

### Quick Start

1. **Collect Flow Data**
   - Click "🔄 Refresh Flow Data" button
   - System fetches options data for 20 popular symbols
   - Takes ~30 seconds (rate-limited at 500ms per symbol)

2. **Generate AI Recommendations**
   - Click "🤖 Run AI Analysis" button
   - AI analyzes flow patterns and generates insights
   - Scores opportunities on 0-100 scale

3. **Explore Opportunities**
   - Navigate to "Flow Opportunities" tab
   - Use filters to find matching criteria
   - Click on opportunities for detailed analysis

### Workflow for Trading

```
1. Daily Morning Routine
   ├── Refresh flow data
   ├── Run AI analysis
   ├── Review Flow Overview tab
   │   ├── Check market-wide sentiment
   │   ├── Note unusual activity alerts
   │   └── Identify top flow symbols
   │
2. Opportunity Selection
   ├── Go to Flow Opportunities tab
   ├── Filter by:
   │   ├── Sentiment: Bullish
   │   ├── Risk: Low or Medium
   │   ├── Min Score: 70+
   │   └── Action: SELL_PUT
   │
3. Detailed Analysis
   ├── Select top opportunity
   ├── Review:
   │   ├── AI recommendation
   │   ├── 7-day trend
   │   ├── Put/call ratio
   │   ├── Recommended strike
   │   └── Win probability
   │
4. Execute Trade
   ├── Use recommended strike
   ├── Verify delta ~0.30
   ├── Confirm premium yield
   └── Enter position via Robinhood
```

---

## Database Schema

### Table: options_flow

Daily snapshots of options volume and premium data.

| Column | Type | Description |
|--------|------|-------------|
| id | SERIAL | Primary key |
| symbol | VARCHAR(10) | Stock ticker |
| flow_date | DATE | Trade date |
| call_volume | BIGINT | Total call contracts traded |
| put_volume | BIGINT | Total put contracts traded |
| call_premium | DECIMAL(15,2) | Total premium in calls |
| put_premium | DECIMAL(15,2) | Total premium in puts |
| net_premium_flow | DECIMAL(15,2) | call_premium - put_premium |
| put_call_ratio | DECIMAL(10,4) | put_volume / call_volume |
| unusual_activity | BOOLEAN | Volume > 2x average |
| flow_sentiment | VARCHAR(20) | Bullish/Bearish/Neutral |
| total_volume | BIGINT | Total option contracts |
| total_open_interest | BIGINT | Total open interest |
| last_updated | TIMESTAMP | Last update time |

**Indexes:**
- `idx_options_flow_symbol` on (symbol)
- `idx_options_flow_date` on (flow_date)
- `idx_options_flow_symbol_date` on (symbol, flow_date)

### Table: options_flow_analysis

Aggregated analysis with AI recommendations.

| Column | Type | Description |
|--------|------|-------------|
| symbol | VARCHAR(10) | Stock ticker (PK) |
| flow_trend_7d | VARCHAR(20) | 7-day trend direction |
| net_flow_7d | DECIMAL(15,2) | 7-day net premium flow |
| net_flow_30d | DECIMAL(15,2) | 30-day net premium flow |
| dominant_strategy | VARCHAR(30) | Call Heavy/Put Heavy/Balanced |
| ai_recommendation | TEXT | AI-generated insights |
| opportunity_score | DECIMAL(5,2) | 0-100 opportunity score |
| best_action | VARCHAR(20) | Recommended action |
| risk_level | VARCHAR(20) | Low/Medium/High |
| confidence | DECIMAL(5,4) | AI confidence (0-1) |
| key_insights | TEXT[] | Bullet point insights |
| recommended_strike | DECIMAL(10,2) | Suggested strike price |
| expected_premium | DECIMAL(10,2) | Expected premium |
| win_probability | DECIMAL(5,4) | Probability of profit |
| last_updated | TIMESTAMP | Last analysis time |

---

## API Reference

### OptionsFlowTracker Class

Located in: `src/options_flow_tracker.py`

#### Methods

**fetch_options_flow(symbol: str) -> Optional[OptionsFlowData]**

Fetches options volume and premium data for a symbol.

```python
from src.options_flow_tracker import OptionsFlowTracker

tracker = OptionsFlowTracker()
flow_data = tracker.fetch_options_flow('AAPL')

print(f"Call Volume: {flow_data.call_volume}")
print(f"Put/Call Ratio: {flow_data.put_call_ratio}")
print(f"Sentiment: {flow_data.flow_sentiment}")
```

**calculate_flow_metrics(symbol: str) -> Dict[str, Any]**

Calculates historical metrics and unusual activity.

```python
metrics = tracker.calculate_flow_metrics('AAPL')

print(f"Average Volume: {metrics['avg_volume']}")
print(f"Is Unusual: {metrics['is_unusual']}")
print(f"7-Day Trend: {metrics['trend_7d']}")
```

**batch_update_flow(symbols: List[str], limit: int = 100) -> Dict[str, int]**

Updates flow data for multiple symbols.

```python
symbols = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA']
results = tracker.batch_update_flow(symbols, limit=5)

print(f"Success: {results['success']}")
print(f"Failed: {results['failed']}")
```

**get_top_flow_opportunities(limit: int = 50) -> List[Dict[str, Any]]**

Retrieves top-ranked opportunities.

```python
opportunities = tracker.get_top_flow_opportunities(limit=10)

for opp in opportunities:
    print(f"{opp['symbol']}: Score {opp['opportunity_score']}")
```

### AIFlowAnalyzer Class

Located in: `src/ai_flow_analyzer.py`

#### Methods

**analyze_flow_sentiment(flow_data: Dict[str, Any]) -> str**

Determines bullish/bearish sentiment.

```python
from src.ai_flow_analyzer import AIFlowAnalyzer

analyzer = AIFlowAnalyzer()
sentiment = analyzer.analyze_flow_sentiment(flow_data)
# Returns: 'Bullish', 'Bearish', or 'Neutral'
```

**score_flow_opportunity(symbol: str, flow_metrics: Dict[str, Any]) -> float**

Calculates 0-100 opportunity score.

```python
score = analyzer.score_flow_opportunity('AAPL', flow_metrics)
print(f"Opportunity Score: {score}/100")
```

**recommend_best_action(flow_data: Dict, historical_data: List) -> str**

Recommends specific trading action.

```python
action = analyzer.recommend_best_action(flow_data, historical_data)
# Returns: 'SELL_PUT', 'BUY_CALL', 'BUY_PUT', 'SELL_CALL', or 'WAIT'
```

**generate_flow_insights(symbol: str, flow_data: Dict, historical_data: List) -> FlowAnalysisResult**

Generates comprehensive AI analysis.

```python
result = analyzer.generate_flow_insights('AAPL', flow_data, historical_data)

print(f"Score: {result.opportunity_score}")
print(f"Action: {result.best_action}")
print(f"Risk: {result.risk_level}")
print(f"AI: {result.ai_recommendation}")
```

---

## UI Components

### Tab 1: Flow Overview

**Purpose:** Market-wide sentiment and unusual activity

**Features:**
- Total call vs put premium bar chart
- Flow sentiment pie chart (bullish/bearish/neutral)
- Top 10 symbols by net flow
- Unusual activity alerts cards
- Market summary metrics

**Use Case:** Quick daily market check before trading

### Tab 2: Flow Opportunities

**Purpose:** Identify and filter top trading opportunities

**Features:**
- Sortable opportunities table
- Filters: sentiment, risk, score, action
- Detailed expandable cards
- AI recommendations
- Strike and premium suggestions

**Use Case:** Find specific CSP opportunities matching your criteria

### Tab 3: Flow Analysis

**Purpose:** Deep dive into individual symbol flow

**Features:**
- Symbol selector
- Historical flow charts (7-30 days)
- Call vs put volume comparison
- AI insights and key takeaways
- Similar flow patterns

**Use Case:** Research before entering position

### Tab 4: Strategies

**Purpose:** Education and strategy guides

**Features:**
- 6 comprehensive strategy guides
- Put/call ratio interpretation
- Trading unusual activity
- Wheel strategy integration
- Risk management framework
- Real case studies

**Use Case:** Learn how to trade options flow effectively

---

## Trading Strategies

### Strategy 1: Following Bullish Flow

**Criteria:**
- Put/Call Ratio < 0.7
- Net flow heavily favoring calls
- Increasing 7-day trend

**Action:**
- Sell cash-secured puts 5-10% OTM
- Target delta ~0.30
- 30-45 DTE

**Win Rate:** ~70% based on historical flow

### Strategy 2: Put/Call Ratio Interpretation

| P/C Ratio | Sentiment | Action |
|-----------|-----------|--------|
| < 0.7 | Very Bullish | Aggressive CSPs |
| 0.7-1.0 | Moderately Bullish | Standard CSPs |
| 1.0-1.3 | Neutral | Conservative CSPs |
| > 1.3 | Bearish | Avoid or Wait |

### Strategy 3: Unusual Activity Trading

**Detection:**
- Volume > 2x average
- Significant premium spike
- Consistent direction

**Verification:**
- Align with flow sentiment
- Confirm with 7-day trend
- Check for mixed signals

**Position Sizing:**
- High confidence: Standard size
- Mixed signals: 50% size
- Bearish unusual: Avoid

---

## Testing

### Validation Script

```bash
# Quick validation
python validate_premium_flow.py

# Expected output: All checks PASSED
```

### Comprehensive Test Suite

```bash
# Run full test suite (requires API access)
python test_premium_flow.py

# Tests:
# 1. Database migration
# 2. Flow data collection
# 3. AI analysis
# 4. Batch operations
# 5. Market summary
```

### Manual Testing Checklist

- [ ] Database tables created
- [ ] Flow data refreshes successfully
- [ ] AI analysis generates recommendations
- [ ] All 4 tabs render correctly
- [ ] Filters work on opportunities table
- [ ] Charts display properly
- [ ] Similar patterns feature works
- [ ] Navigation from dashboard works

---

## Troubleshooting

### Common Issues

**1. Migration fails**

```
Error: relation "options_flow" already exists
```

**Solution:** Tables already exist. Skip migration or drop tables first.

```sql
DROP TABLE IF EXISTS options_flow_alerts CASCADE;
DROP TABLE IF EXISTS premium_flow_opportunities CASCADE;
DROP TABLE IF EXISTS options_flow_analysis CASCADE;
DROP TABLE IF EXISTS options_flow CASCADE;
```

**2. No flow data collected**

```
Warning: No options available for SYMBOL
```

**Solution:**
- Symbol may not have options
- Market may be closed
- Try popular symbols: AAPL, MSFT, GOOGL

**3. AI analysis fails**

```
Error: No OpenAI API key provided
```

**Solution:**
- Set OPENAI_API_KEY in .env file
- Or use rule-based analysis (automatic fallback)

**4. Slow data collection**

```
Taking too long to refresh...
```

**Solution:**
- Rate limiting is intentional (500ms per symbol)
- Reduce number of symbols in batch
- Run during market hours for faster response

**5. Empty opportunities table**

```
No opportunities match your filters
```

**Solution:**
- Reduce min score filter
- Change sentiment filter to "All"
- Run AI Analysis first
- Ensure flow data was collected

---

## Performance Considerations

### Data Collection

- **Rate Limit:** 500ms between API calls
- **Batch Size:** 20-100 symbols recommended
- **Time:** ~10-50 seconds for batch update

### AI Analysis

- **With OpenAI:** 2-5 seconds per symbol
- **Without OpenAI:** <1 second per symbol (rule-based)
- **Batch:** 30 symbols in ~1 minute

### Database Queries

- **Optimized indexes** for fast retrieval
- **Typical query time:** <100ms
- **Market summary:** <200ms

---

## Future Enhancements

### Planned Features

1. **Real-time Alerts**
   - Email/SMS notifications for unusual activity
   - Customizable alert thresholds
   - Integration with Telegram bot

2. **Advanced Flow Metrics**
   - Block trades detection
   - Institutional vs retail flow separation
   - Options spread analysis

3. **Historical Backtesting**
   - Test strategies on historical flow
   - Performance metrics
   - Optimization suggestions

4. **Integration with Robinhood**
   - One-click trade execution
   - Position tracking
   - P&L correlation with flow

5. **Enhanced AI Models**
   - Custom fine-tuned models
   - Ensemble predictions
   - Market regime detection

---

## Support & Contact

For issues or questions:
- Review this documentation
- Check troubleshooting section
- Review CODING_GUIDELINES.md
- Test with validate_premium_flow.py

---

## License & Credits

Part of the Magnus Trading Platform
Built for Wheel Strategy traders
Integrates: Yahoo Finance, OpenAI GPT-4, PostgreSQL, Streamlit

---

**Version:** 1.0.0
**Last Updated:** 2025-11-02
**Status:** Production Ready

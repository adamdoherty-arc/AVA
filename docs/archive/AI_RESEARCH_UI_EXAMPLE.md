# AI Research UI Preview

## Visual Layout Example

### 1. Position Table with AI Research Buttons

```
📊 Stock Positions
2 stock position(s)

┌─────────┬────────┬───────────────┬────────────────┬─────────────┬───────────────┬──────────┬─────────┬───────┐
│ Symbol  │ Shares │ Avg Buy Price │ Current Price  │ Cost Basis  │ Current Value │   P/L    │  P/L %  │ Chart │
├─────────┼────────┼───────────────┼────────────────┼─────────────┼───────────────┼──────────┼─────────┼───────┤
│ AAPL    │  100   │    $172.50    │    $175.30     │  $17,250.00 │  $17,530.00   │ $280.00  │  1.6%   │  📈   │
│ MSFT    │   50   │    $380.20    │    $375.10     │  $19,010.00 │  $18,755.00   │ -$255.00 │ -1.3%   │  📈   │
└─────────┴────────┴───────────────┴────────────────┴─────────────┴───────────────┴──────────┴─────────┴───────┘

AI Research:
┌─────────────┐  ┌─────────────┐
│  🤖 AAPL   │  │  🤖 MSFT   │
└─────────────┘  └─────────────┘
```

### 2. Research Display (When Button Clicked)

```
▼ 🤖 AI Research: AAPL

AI Research: AAPL
Overall Rating: ⭐⭐⭐⭐☆ (4.2/5.0)
Last updated: 02:34 PM

─────────────────────────────────────────────────────────────

Quick Summary
ℹ️ Strong fundamentals with bullish technical setup. Good environment for options strategies.

─────────────────────────────────────────────────────────────

Recommendation

┌─────────────────┐  Confidence: 85%
│      BUY       │  Analysis combines fundamental strength (82/100), technical
│                 │  setup (78/100), and sentiment indicators (88/100) to
└─────────────────┘  generate BUY recommendation with 85% confidence.

⚠️ Time-Sensitive Factors:
- Earnings in 12 days - expect increased volatility
- Strong positive sentiment momentum

─────────────────────────────────────────────────────────────

💡 Advice for Your Long Stock Position:
Favorable entry point based on fundamental and technical analysis. Set stops at key support levels.

─────────────────────────────────────────────────────────────

Detailed Analysis

┌──────────────┬─────────────┬──────────────┬──────────────┐
│ 📊 Fundamental│ 📈 Technical│ 💬 Sentiment │ 🎯 Options   │
└──────────────┴─────────────┴──────────────┴──────────────┘

📊 Fundamental Tab:

Score: 82/100 (in green)

Revenue Growth YoY          Earnings Beat Streak
     12.3%                         4

P/E Ratio                   Valuation:
   28.5                     Fair valuation vs peers

Sector Avg P/E
   25.0

Key Strengths:
✅ Strong revenue growth trajectory
✅ Healthy balance sheet
✅ Competitive moat advantages

Key Risks:
⚠️ Valuation concerns
⚠️ Competitive pressures

─────────────────────────────────────────────────────────────

📈 Technical Tab:

Score: 78/100 (in green)

Trend                    MACD Signal              Support:
UPTREND                  BULLISH                  $170.00, $165.00

RSI                                               Resistance:
52.3                                              $180.00, $185.00

Volume Analysis:
Above average volume confirming trend

Chart Patterns:
📊 Bull flag formation
📊 Breakout confirmed

ℹ️ Recommendation: Strong uptrend - consider entries on pullbacks

─────────────────────────────────────────────────────────────

💬 Sentiment Tab:

Score: 88/100 (in green)

News Sentiment           Social Sentiment         Institutional Flow
   POSITIVE                 NEUTRAL               Moderate Buying

News Articles (7d)       Reddit Mentions (24h)    Analyst Rating
      32                       456                    Buy

Analyst Consensus:

Strong Buy    Buy      Hold     Sell    Strong Sell
    8         12        5        1          0

─────────────────────────────────────────────────────────────

🎯 Options Tab:

Score: 75/100 (in green)

IV Rank               Current IV              Days to Earnings
 45/100                 28.0%                      12

IV Percentile         30d Avg IV              Avg Earnings Move
 62/100                 26.0%                     4.5%

Put/Call Ratio
    0.85

📋 Cash Secured Put: Optimal risk/reward given current IV environment

─────────────────────────────────────────────────────────────

▶ ℹ️ Report Metadata
```

### 3. Multiple Positions Example

```
💰 Cash-Secured Puts
3 active position(s)

┌─────────┬──────────────┬─────────────┬──────────┬────────┬────────────┬─────┬───────────┬─────────┬─────────┬─────────┬─────────┬───────┐
│ Symbol  │ Stock Price  │ After-Hours │ Strategy │ Strike │ Expiration │ DTE │ Contracts │ Premium │ Current │   P/L   │  P/L %  │ Chart │
├─────────┼──────────────┼─────────────┼──────────┼────────┼────────────┼─────┼───────────┼─────────┼─────────┼─────────┼─────────┼───────┤
│ AAPL    │   $175.30    │      -      │   CSP    │ $170.00│ 2025-11-15 │  14 │     1     │ $250.00 │ $125.00 │ $125.00 │  50.0%  │  📈   │
│ MSFT    │   $375.10    │      -      │   CSP    │ $370.00│ 2025-11-22 │  21 │     1     │ $320.00 │ $180.00 │ $140.00 │  43.8%  │  📈   │
│ GOOGL   │   $142.50    │      -      │   CSP    │ $140.00│ 2025-11-08 │   7 │     1     │ $180.00 │ $220.00 │ -$40.00 │ -22.2%  │  📈   │
└─────────┴──────────────┴─────────────┴──────────┴────────┴────────────┴─────┴───────────┴─────────┴─────────┴─────────┴─────────┴───────┘

AI Research:
┌─────────────┐  ┌─────────────┐  ┌─────────────┐
│  🤖 AAPL   │  │  🤖 MSFT   │  │  🤖 GOOGL  │
└─────────────┘  └─────────────┘  └─────────────┘

▼ 🤖 AI Research: AAPL
[Full research display as shown above]

▼ 🤖 AI Research: GOOGL
[Full research display for GOOGL]
```

## Color Coding Reference

### Scores (0-100)
- **80-100**: 🟢 Green (#00AA00) - Excellent
- **60-79**: 🟠 Orange (#FFA500) - Good
- **0-59**: 🔴 Red (#DD0000) - Poor

### Trade Actions
- **STRONG_BUY**: 🟢 Bright Green (#00AA00)
- **BUY**: 🟢 Light Green (#66CC66)
- **HOLD**: 🟠 Orange (#FFA500)
- **SELL**: 🔴 Light Red (#FF6666)
- **STRONG_SELL**: 🔴 Bright Red (#DD0000)

### Profit/Loss
- **Positive P/L**: 🟢 Green text (#00AA00)
- **Negative P/L**: 🔴 Red text (#DD0000)
- **Zero P/L**: ⚪ Default text

## Interactive Elements

### Loading State
```
Loading AI research for AAPL...
[Spinner animation]
```

### Error State
```
❌ Error loading AI research: Connection timeout
Please try again or contact support if the issue persists
```

### Cache Indicator
```
Last updated: 02:34 PM
Cache Expires: 03:04 PM
```

## Mobile Responsive Behavior

### Desktop (Wide Screen)
- Buttons displayed in multi-column layout
- All tabs visible horizontally
- Full metrics grid

### Tablet (Medium Screen)
- Buttons may wrap to 2 rows
- Tabs scroll horizontally if needed
- Metrics in 2 columns

### Mobile (Narrow Screen)
- Buttons stack vertically
- Tabs in dropdown selector
- Metrics in single column

## Accessibility Features

1. **Semantic Colors**: Color + text labels (not color-only)
2. **ARIA Labels**: All buttons properly labeled
3. **Keyboard Navigation**: Tab through buttons, Enter to activate
4. **Screen Reader**: All content readable by screen readers
5. **Focus States**: Clear visual focus indicators

## Performance Characteristics

### First Load (No Cache)
- Click button → 500ms loading → Display research
- Total time: ~0.5-1 second

### Cached Load
- Click button → Instant display
- Total time: <100ms

### Multiple Positions
- Each position tracked independently
- No performance degradation with 10+ positions
- Cache shared across page refreshes

## User Interaction Patterns

### Pattern 1: Quick Check
1. User spots interesting P/L
2. Clicks AI button
3. Reads quick summary + recommendation
4. Closes or leaves open

### Pattern 2: Deep Dive
1. User clicks AI button
2. Reads summary
3. Explores all 4 analysis tabs
4. Reviews position-specific advice
5. Makes trading decision

### Pattern 3: Comparison
1. User opens research for multiple positions
2. Compares ratings and recommendations
3. Prioritizes positions to manage
4. Takes action on highest priority

## Tips for Users

💡 **Pro Tips:**
- Click multiple AI buttons to compare stocks side-by-side
- Check cache expiration to know if data is fresh
- Position-specific advice is tailored to your strategy type
- Time-sensitive factors highlight urgent considerations
- Green/orange/red colors indicate quality at a glance

🎯 **Best Practices:**
- Review AI research before rolling positions
- Use technical tab for entry/exit timing
- Check sentiment before earnings
- Monitor options tab for IV changes
- Reference fundamental tab for long-term holds

---

**Note**: This is a text representation of the UI. The actual Streamlit interface will be more visually appealing with proper styling, spacing, and interactive elements.

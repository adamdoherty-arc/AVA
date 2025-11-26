# Feature Plan: Multi-Expiration Options Display with AI Analysis

**Created**: 2025-10-27
**Status**: Planning Phase
**Priority**: HIGH

---

## User Requirements

### Primary Goals
1. **Multiple Options Per Stock** - Show options for 7, 14, 30, and 45 DTE
2. **Delta Targeting** - Show multiple options around 0.30 delta for each expiration
3. **List Option Prices** - Clear display of premium, bid, ask for each option
4. **AI Analysis** - Prompt/window where user can ask "which ones are best to make money"
5. **Better Navigation** - Expandable rows to make navigation easier

### Current Limitations
- Dashboard shows only ONE option per stock (highest monthly return)
- Cannot compare different expirations side-by-side
- No way to see alternative strikes at same expiration
- No AI-powered opportunity ranking

---

## Current Data Analysis

### Data Available in Database
```
AAPL Example:
- DTE 10: 3 options (delta range 0.18-0.33)
- DTE 17: 2 options (delta range 0.21-0.30)
- DTE 24: 2 options (delta range 0.23-0.27)
- DTE 31: 2 options (delta range 0.21-0.28)
- DTE 38: 2 options (delta range 0.22-0.29)
```

**Observation**: We have LIMITED options per DTE (2-3 each), not dozens. System is already storing the best ~0.30 delta options.

### Current Database Schema
```sql
stock_premiums (
    symbol,
    strike_price,
    dte,
    expiration_date,
    premium,
    bid,
    ask,
    monthly_return,
    annual_return,
    delta,
    implied_volatility,
    volume,
    open_interest,
    strike_type  -- '30_delta' or '5%_OTM'
)
```

---

## Proposed Solution

### Architecture Design

#### Component 1: Grouped Stock View (Main Table)
**Purpose**: Show one row per stock with summary metrics

**Display**:
```
┌─────────────────────────────────────────────────────────────────┐
│ Symbol │ Price │ Best Premium │ Best Monthly % │ # Options │ ▼ │
├─────────────────────────────────────────────────────────────────┤
│ AAPL   │ $170  │    $285      │     3.2%       │    10     │ ▼ │  ← Click to expand
│ NVDA   │ $485  │    $980      │     4.1%       │    12     │ ▼ │
│ MSFT   │ $375  │    $595      │     2.8%       │     8     │ ▼ │
└─────────────────────────────────────────────────────────────────┘
```

#### Component 2: Expandable Multi-Expiration View
**Purpose**: When row is clicked, show ALL options for that stock grouped by DTE

**Expanded View**:
```
AAPL - $170.50
┌──────────────────────────────────────────────────────────────────────┐
│ 7-14 DAYS OUT (DTE 10)                                               │
├──────────────────────────────────────────────────────────────────────┤
│ Strike   │ Premium │ Bid/Ask    │ Delta  │ Monthly % │ IV    │ Vol  │
│ $165.00  │  $285   │ $2.80/2.90 │ -0.32  │   3.2%    │ 28%   │ 450  │
│ $162.50  │  $195   │ $1.90/2.00 │ -0.28  │   2.3%    │ 26%   │ 320  │
│ $160.00  │  $125   │ $1.20/1.30 │ -0.18  │   1.5%    │ 24%   │ 280  │
├──────────────────────────────────────────────────────────────────────┤
│ 14-21 DAYS OUT (DTE 17)                                              │
├──────────────────────────────────────────────────────────────────────┤
│ $165.00  │  $420   │ $4.15/4.25 │ -0.30  │   4.5%    │ 30%   │ 380  │
│ $162.50  │  $295   │ $2.90/3.00 │ -0.21  │   3.2%    │ 28%   │ 250  │
├──────────────────────────────────────────────────────────────────────┤
│ 30 DAYS OUT (DTE 31)                                                 │
├──────────────────────────────────────────────────────────────────────┤
│ $165.00  │  $685   │ $6.80/6.90 │ -0.28  │   5.8%    │ 32%   │ 420  │
│ $162.50  │  $495   │ $4.90/5.00 │ -0.21  │   4.2%    │ 30%   │ 310  │
├──────────────────────────────────────────────────────────────────────┤
│ 45 DAYS OUT (DTE 38)                                                 │
├──────────────────────────────────────────────────────────────────────┤
│ $165.00  │  $895   │ $8.90/9.00 │ -0.29  │   6.5%    │ 33%   │ 380  │
│ $162.50  │  $665   │ $6.60/6.70 │ -0.22  │   4.8%    │ 31%   │ 290  │
└──────────────────────────────────────────────────────────────────────┘

[Ask AI: Which option is best? 💬]
```

#### Component 3: AI Analysis Prompt
**Purpose**: Natural language interface to analyze and recommend options

**Interface**:
```
┌─────────────────────────────────────────────────────────────────┐
│ 🤖 AI Options Analyzer                                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Ask me anything about the options:                             │
│                                                                 │
│  • "Which option gives best risk/reward?"                       │
│  • "Show me the safest plays"                                   │
│  • "Which expiration maximizes premium?"                        │
│  • "I have $5000, what should I do?"                            │
│  • "What are the top 3 opportunities?"                          │
│                                                                 │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │ Which options are best to make money?                   │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                  [Analyze 🔍]   │
├─────────────────────────────────────────────────────────────────┤
│ 📊 AI Analysis Results:                                         │
│                                                                 │
│ Based on your current data, here are the top 3 opportunities:  │
│                                                                 │
│ 1. NVDA DTE 31, Strike $480 - $980 premium                     │
│    • Monthly Return: 4.1%                                       │
│    • Delta: -0.29 (good probability)                            │
│    • Reasoning: Best balance of premium and safety              │
│                                                                 │
│ 2. AAPL DTE 17, Strike $165 - $420 premium                     │
│    • Monthly Return: 4.5%                                       │
│    • Delta: -0.30 (ideal target)                                │
│    • Reasoning: Shorter DTE, high monthly return                │
│                                                                 │
│ 3. MSFT DTE 31, Strike $370 - $595 premium                     │
│    • Monthly Return: 3.5%                                       │
│    • Delta: -0.28 (safe)                                        │
│    • Reasoning: Stable stock, consistent premium                │
│                                                                 │
│ [Ask Another Question]                                          │
└─────────────────────────────────────────────────────────────────┘
```

---

## Implementation Plan

### Phase 1: Data Verification & Query Design
**Goal**: Ensure we can fetch multi-DTE options efficiently

**Tasks**:
1. Verify database has options for target DTEs (7, 14, 30, 45)
2. Create SQL query to fetch ALL options per stock grouped by DTE
3. Test query performance with full dataset
4. Validate delta filtering (~0.30 ±0.10 range)

**SQL Query Design**:
```sql
-- Get all options for a stock, grouped by expiration
SELECT
    sp.symbol,
    sd.current_price,
    sp.dte,
    sp.expiration_date,
    sp.strike_price,
    sp.premium,
    sp.bid,
    sp.ask,
    sp.monthly_return,
    sp.annual_return,
    sp.delta,
    sp.implied_volatility,
    sp.volume,
    sp.open_interest,
    sp.strike_type
FROM stock_premiums sp
LEFT JOIN stock_data sd ON sp.symbol = sd.symbol
WHERE sp.symbol = %s
  AND sp.delta IS NOT NULL
  AND ABS(sp.delta) BETWEEN 0.20 AND 0.40  -- Focus on ~30 delta ±10
  AND sp.dte IN (7, 10, 14, 17, 21, 24, 30, 31, 38, 45)  -- Target DTEs
ORDER BY sp.dte ASC, sp.monthly_return DESC
```

### Phase 2: UI Implementation - Expandable Rows
**Goal**: Create collapsible stock rows with expander component

**Streamlit Approach**:
```python
# Main table - one row per stock
stocks = get_unique_stocks()

for stock in stocks:
    col1, col2, col3, col4 = st.columns([2, 2, 2, 1])

    with col1:
        st.write(f"**{stock['symbol']}** ${stock['price']:.2f}")
    with col2:
        st.metric("Best Premium", f"${stock['best_premium']:.0f}")
    with col3:
        st.metric("Best Monthly %", f"{stock['best_monthly']:.2f}%")
    with col4:
        # Expandable trigger
        with st.expander("📊"):
            show_multi_expiration_view(stock['symbol'])
```

**Alternative Approach** (Better UX):
```python
# Use tabs for each stock
stock_tabs = st.tabs([f"{s['symbol']} (${s['price']:.0f})" for s in stocks])

for i, stock in enumerate(stocks):
    with stock_tabs[i]:
        show_multi_expiration_view(stock['symbol'])
```

### Phase 3: Multi-Expiration Display
**Goal**: Show all options grouped by DTE ranges

**Grouping Logic**:
- **7-14 Days**: DTE 7-14
- **14-21 Days**: DTE 14-21
- **21-30 Days**: DTE 21-30
- **30-45 Days**: DTE 30-45

**Display Function**:
```python
def show_multi_expiration_view(symbol: str):
    options = fetch_all_options_for_stock(symbol)

    # Group by DTE ranges
    dte_groups = {
        "7-14 Days": options[(options['dte'] >= 7) & (options['dte'] <= 14)],
        "14-21 Days": options[(options['dte'] > 14) & (options['dte'] <= 21)],
        "21-30 Days": options[(options['dte'] > 21) & (options['dte'] <= 30)],
        "30-45 Days": options[(options['dte'] > 30) & (options['dte'] <= 45)]
    }

    for group_name, group_df in dte_groups.items():
        if not group_df.empty:
            st.subheader(group_name)

            # Display as clean table
            display_df = group_df[[
                'Strike', 'Premium', 'Bid', 'Ask',
                'Delta', 'Monthly %', 'IV', 'Volume'
            ]]

            st.dataframe(
                display_df,
                hide_index=True,
                use_container_width=True,
                column_config={
                    "Strike": st.column_config.NumberColumn(format="$%.2f"),
                    "Premium": st.column_config.NumberColumn(format="$%.0f"),
                    "Bid": st.column_config.NumberColumn(format="$%.2f"),
                    "Ask": st.column_config.NumberColumn(format="$%.2f"),
                    "Delta": st.column_config.NumberColumn(format="%.3f"),
                    "Monthly %": st.column_config.NumberColumn(format="%.2f%%"),
                    "IV": st.column_config.NumberColumn(format="%.1f%%"),
                }
            )
```

### Phase 4: AI Analysis Integration
**Goal**: Add natural language interface for option analysis

**Two Approaches**:

**Approach A: Simple Rule-Based Analysis** (Fast, No API needed)
```python
def analyze_options(question: str, all_options_df):
    """Rule-based analysis based on keywords"""

    question_lower = question.lower()

    if "best" in question_lower and "money" in question_lower:
        # Sort by monthly return
        top = all_options_df.nlargest(3, 'monthly_return')
        return format_top_recommendations(top, "highest returns")

    elif "safe" in question_lower:
        # Lower delta, good liquidity
        safe = all_options_df[
            (all_options_df['delta'].abs() < 0.25) &
            (all_options_df['volume'] > 100)
        ].nlargest(3, 'monthly_return')
        return format_top_recommendations(safe, "safest plays")

    elif "premium" in question_lower and "max" in question_lower:
        # Highest absolute premium
        top = all_options_df.nlargest(3, 'premium')
        return format_top_recommendations(top, "highest premiums")

    else:
        # Default: best risk/reward
        return analyze_risk_reward(all_options_df)
```

**Approach B: LLM-Powered Analysis** (Better, Requires API)
```python
import openai

def ai_analyze_options(question: str, all_options_df):
    """Use LLM to analyze options and provide insights"""

    # Convert options to context
    context = options_to_context(all_options_df)

    prompt = f"""
You are an expert options trader analyzing cash-secured put opportunities.

Available Options:
{context}

User Question: {question}

Analyze the options and provide:
1. Top 3 recommendations
2. Reasoning for each
3. Risk/reward assessment
4. Expected outcomes

Format your response clearly with specific option details.
"""

    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}]
    )

    return response.choices[0].message.content
```

### Phase 5: Testing & Refinement
**Goal**: Ensure quality and usability

**Test Cases**:
1. Stock with many options (AAPL, NVDA) - verify all show
2. Stock with few options - verify graceful display
3. Expand/collapse functionality works
4. AI analysis returns sensible recommendations
5. Performance with 146 stocks
6. Mobile/tablet responsive design

---

## Database Requirements

### Current State
✅ Data already stored for multiple expirations per stock
✅ Delta values calculated
✅ Premium, bid, ask, volume, OI tracked

### Potential Enhancements
⚠️ **May need more options per DTE** - Currently only 2-3 options stored per expiration

**Solution**: Enhance `enhanced_options_fetcher.py` to store MORE strikes near 0.30 delta

```python
# Instead of finding BEST put, find TOP 5 puts near target delta
def find_top_puts_near_delta(self, strikes, target_delta=-0.30, top_n=5):
    """Find multiple puts near target delta"""

    puts_with_delta = []
    for put in strikes:
        delta = self.calculate_delta(...)
        delta_diff = abs(delta - target_delta)
        puts_with_delta.append((put, delta, delta_diff))

    # Sort by delta difference, take top N
    puts_with_delta.sort(key=lambda x: x[2])
    return puts_with_delta[:top_n]
```

---

## UI/UX Design Decisions

### Option 1: Expanders (Current Proposal)
**Pros**:
- Compact main view
- All stocks visible at once
- Click to drill down

**Cons**:
- Requires clicking each stock
- Can't compare multiple stocks simultaneously

### Option 2: Tabs
**Pros**:
- Full-screen space for each stock
- Clean navigation
- Easy to switch between stocks

**Cons**:
- Can't see multiple stocks at once
- More clicks to navigate

### Option 3: Side-by-Side Comparison
**Pros**:
- Compare 2-3 stocks directly
- Select which to compare

**Cons**:
- Complex UI
- Limited screen space

**RECOMMENDATION**: **Option 1 (Expanders)** with ability to "pin" stocks for comparison

---

## AI Analysis Features

### Question Types to Support

1. **Best for Making Money**
   - Rank by monthly return
   - Consider risk (delta)
   - Factor in liquidity (volume)

2. **Safest Plays**
   - Lower delta (< 0.25)
   - High volume/OI
   - Stable stocks

3. **Capital Optimization**
   - "I have $X, show best allocation"
   - Suggest portfolio of puts
   - Diversify across stocks/expirations

4. **Time-based**
   - "What's best for this week?"
   - "Best 30-day plays?"

5. **Stock-specific**
   - "Best AAPL strategy?"
   - "Compare NVDA vs MSFT"

### Analysis Metrics

**Scoring Formula**:
```python
def calculate_opportunity_score(option):
    """Score each option 0-100"""

    # Factors:
    # 1. Monthly return (40%)
    # 2. Delta proximity to 0.30 (30%)
    # 3. Liquidity - volume & OI (20%)
    # 4. IV level (10%)

    return_score = min(option['monthly_return'] / 0.05 * 40, 40)  # 5% = max
    delta_score = (1 - abs(abs(option['delta']) - 0.30) / 0.10) * 30
    liquidity_score = min(option['volume'] / 500, 1) * 20
    iv_score = min(option['iv'] / 0.50, 1) * 10

    return return_score + delta_score + liquidity_score + iv_score
```

---

## Success Criteria

### Functional Requirements
- [ ] Show 4+ expiration groups per stock
- [ ] Display 3-5 strikes per expiration
- [ ] All options near 0.30 delta (±0.10)
- [ ] Expandable/collapsible stock rows
- [ ] AI prompt accepts natural language
- [ ] AI returns top 3 recommendations with reasoning

### Performance Requirements
- [ ] Page loads < 3 seconds
- [ ] Expand operation < 500ms
- [ ] AI analysis < 5 seconds
- [ ] Handles 146 stocks smoothly

### UX Requirements
- [ ] Intuitive navigation
- [ ] Clear visual hierarchy
- [ ] Mobile-friendly (responsive)
- [ ] Accessible (keyboard navigation)

---

## Risks & Mitigation

### Risk 1: Limited Options in Database
**Issue**: Only 2-3 options per DTE currently stored
**Impact**: Can't show "multiple options around 0.30 delta"
**Mitigation**: Enhance fetcher to store top 5 strikes per DTE
**Effort**: Medium (modify enhanced_options_fetcher.py)

### Risk 2: Performance with Many Options
**Issue**: 146 stocks × 4 DTEs × 5 strikes = 2,920 options
**Impact**: Slow loading, memory issues
**Mitigation**:
- Load data progressively (only fetch when expanded)
- Use caching aggressively
- Limit to filtered stocks only
**Effort**: Low (good Streamlit caching)

### Risk 3: AI Analysis Costs
**Issue**: LLM API calls can be expensive
**Impact**: High costs with frequent queries
**Mitigation**:
- Start with rule-based analysis
- Add LLM as optional "advanced" mode
- Cache common queries
**Effort**: Low (implement both modes)

### Risk 4: UX Complexity
**Issue**: Too much data can overwhelm users
**Impact**: Poor user experience
**Mitigation**:
- Progressive disclosure (show summary first)
- Smart defaults
- Clear visual grouping
**Effort**: Medium (iterative design)

---

## Timeline Estimate

### Phase 1: Data Layer (1-2 hours)
- Enhance database queries
- Modify fetcher to store more strikes
- Test data availability

### Phase 2: UI - Expandable View (2-3 hours)
- Implement expander component
- Multi-DTE display
- Formatting and styling

### Phase 3: AI Analysis (2-3 hours)
- Rule-based analyzer
- Prompt interface
- Results formatting

### Phase 4: Testing & Polish (1-2 hours)
- QA testing
- Code review
- Performance optimization

**Total**: 6-10 hours

---

## Next Steps

1. **Review with Architect Agent** ✅ (This document)
2. **Verify data availability** (Run database queries)
3. **Get user approval** on design approach
4. **Implement Phase 1** (Data layer)
5. **Implement Phase 2** (UI)
6. **Implement Phase 3** (AI)
7. **Test with QA Agent**
8. **Review with Code Reviewer**
9. **Deploy**

---

## Open Questions for User

1. **Expanders vs Tabs**: Which UI approach do you prefer?
2. **AI Analysis**: Start with rule-based or integrate LLM immediately?
3. **Data Enhancement**: Should we enhance fetcher to get MORE strikes per DTE? (Currently only 2-3)
4. **DTE Grouping**: Are the ranges (7-14, 14-21, 21-30, 30-45) what you want?
5. **Capital**: Do you want the AI to suggest position sizing/allocation?

---

**Status**: Ready for Architect Review 📋

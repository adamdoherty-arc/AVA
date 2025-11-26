# Option Strategies Enhancement Plan

## Overview

Enhance Magnus with comprehensive option strategy analysis, modular AI research, and TradingView watchlist integration.

---

## Phase 1: Modular AI Research Component

### Goal
Make AI Research reusable across multiple pages (Positions, TradingView Watchlists, Calendar Spreads, etc.)

### Implementation

**Create**: `src/components/ai_research_widget.py`

```python
"""
Modular AI Research Widget
Can be imported and used on any Streamlit page
"""

import streamlit as st
from datetime import datetime
from src.ai_research_service import get_research_service

def display_ai_research_button(symbol: str, key_prefix: str = ""):
    """
    Display a single AI Research button

    Args:
        symbol: Stock ticker
        key_prefix: Unique prefix for button key (e.g., 'watchlist', 'positions')
    """
    if st.button(f"🤖 {symbol}", key=f"{key_prefix}_ai_{symbol}"):
        st.session_state[f'show_research_{key_prefix}_{symbol}'] = True

def display_ai_research_expander(symbol: str, key_prefix: str = "", position_type: str = None):
    """
    Display AI Research in an expander

    Args:
        symbol: Stock ticker
        key_prefix: Unique prefix for session state
        position_type: Type of position for context-aware advice
    """
    if st.session_state.get(f'show_research_{key_prefix}_{symbol}', False):
        with st.expander(f"🤖 AI Research: {symbol}", expanded=True):
            display_ai_research_analysis(symbol, position_type)

def display_ai_research_analysis(symbol: str, position_type: str = None):
    """Core AI research display logic"""
    research_service = get_research_service()

    with st.spinner(f"Loading AI research for {symbol}..."):
        try:
            report = research_service.get_research_report(symbol)

            if not report:
                st.error("Failed to load research report")
                return

            # Display report (full implementation from positions_page_improved.py)
            # ... [rest of display logic]

        except Exception as e:
            st.error(f"Error loading research: {e}")

def display_consolidated_ai_research_section(symbols: list, key_prefix: str = ""):
    """
    Display consolidated AI Research section for multiple symbols

    Args:
        symbols: List of stock tickers
        key_prefix: Unique prefix for this section
    """
    if not symbols:
        return

    st.markdown("---")
    st.markdown("### 🤖 AI Research")
    st.caption(f"Analyzing {len(symbols)} symbols")

    # Display buttons (5 per row)
    cols = st.columns(min(len(symbols), 5))
    for idx, symbol in enumerate(symbols):
        col_idx = idx % 5
        with cols[col_idx]:
            display_ai_research_button(symbol, key_prefix)

    # Display expanders
    for symbol in symbols:
        display_ai_research_expander(symbol, key_prefix)
```

### Usage Examples

**In Positions Page**:
```python
from src.components.ai_research_widget import display_consolidated_ai_research_section

# Collect symbols from all positions
all_symbols = get_all_position_symbols()
display_consolidated_ai_research_section(all_symbols, key_prefix="positions")
```

**In TradingView Watchlists**:
```python
from src.components.ai_research_widget import display_consolidated_ai_research_section

# Get symbols from selected watchlist
watchlist_symbols = get_watchlist_symbols(selected_watchlist)
display_consolidated_ai_research_section(watchlist_symbols, key_prefix="watchlist")
```

---

## Phase 2: Calendar Spreads in TradingView Watchlists

### Goal
Move Calendar Spreads functionality into TradingView Watchlists page as a new tab

### Implementation

**Modify**: `dashboard.py` - Remove Calendar Spreads navigation button
**Modify**: TradingView Watchlists page - Add "Calendar Spreads" tab

**New Tab Structure**:
```
TradingView Watchlists Page
├── 📋 Watchlists Tab (existing)
├── 🔄 Auto-Sync Tab (existing)
└── 📆 Calendar Spreads Tab (NEW)
    ├── Watchlist Selector
    ├── Strategy Parameters
    ├── Calendar Spread Finder
    └── Results Table (sortable by profit)
```

---

## Phase 3: Calendar Spread Evaluator

### Goal
Evaluate each symbol from watchlists and find optimal calendar spreads

### Data Model

```python
@dataclass
class CalendarSpreadOpportunity:
    symbol: str
    stock_price: float

    # Near-term leg (sell)
    near_strike: float
    near_expiration: date
    near_dte: int
    near_premium: float
    near_iv: float

    # Far-term leg (buy)
    far_strike: float
    far_expiration: date
    far_dte: int
    far_premium: float
    far_iv: float

    # Spread metrics
    net_debit: float  # Cost to open spread
    max_profit: float  # At expiration if stock at strike
    max_loss: float  # = net_debit
    profit_potential: float  # Max profit / max loss ratio
    probability_profit: float  # % chance of profit (0-100)
    breakeven_range: tuple  # (lower, upper) prices for profit

    # Greeks
    theta: float  # Time decay advantage
    vega: float  # Volatility advantage

    # Scoring
    opportunity_score: float  # 0-100 composite score
    rank: int  # Relative ranking
```

### Evaluation Algorithm

```python
def evaluate_calendar_spread(symbol: str, stock_price: float) -> List[CalendarSpreadOpportunity]:
    """
    Find and evaluate all calendar spread opportunities for a symbol

    Strategy:
    1. Sell near-term option (30-45 DTE)
    2. Buy far-term option (60-90 DTE) at same strike
    3. Profit from faster theta decay of near-term option

    Returns:
        List of opportunities sorted by opportunity_score
    """

    # Get options chain
    near_options = get_options_chain(symbol, min_dte=30, max_dte=45)
    far_options = get_options_chain(symbol, min_dte=60, max_dte=90)

    opportunities = []

    # Try strikes near current price (ATM ±10%)
    for strike in get_strikes_near_price(stock_price, range_pct=0.10):

        # Find best near-term option to sell
        near_option = find_best_option(near_options, strike, option_type='call')
        if not near_option:
            continue

        # Find best far-term option to buy
        far_option = find_best_option(far_options, strike, option_type='call')
        if not far_option:
            continue

        # Calculate spread metrics
        net_debit = far_option.premium - near_option.premium

        # Max profit occurs when near-term expires worthless but far-term retains value
        # Estimate: far-term intrinsic value at near-term expiration
        max_profit = calculate_max_profit(near_option, far_option, stock_price)

        # Probability of profit (Monte Carlo or Black-Scholes)
        prob_profit = calculate_probability_profit(near_option, far_option, stock_price)

        # Opportunity score (0-100)
        score = calculate_opportunity_score(
            profit_potential=max_profit / net_debit,
            probability=prob_profit,
            theta_advantage=near_option.theta - far_option.theta,
            liquidity=min(near_option.volume, far_option.volume)
        )

        opportunities.append(CalendarSpreadOpportunity(
            symbol=symbol,
            stock_price=stock_price,
            near_strike=strike,
            near_expiration=near_option.expiration,
            near_dte=near_option.dte,
            near_premium=near_option.premium,
            near_iv=near_option.iv,
            far_strike=strike,
            far_expiration=far_option.expiration,
            far_dte=far_option.dte,
            far_premium=far_option.premium,
            far_iv=far_option.iv,
            net_debit=net_debit,
            max_profit=max_profit,
            max_loss=net_debit,
            profit_potential=max_profit / net_debit,
            probability_profit=prob_profit,
            opportunity_score=score
        ))

    # Sort by opportunity score (highest first)
    opportunities.sort(key=lambda x: x.opportunity_score, reverse=True)

    # Assign ranks
    for idx, opp in enumerate(opportunities, 1):
        opp.rank = idx

    return opportunities
```

### Display Table

```python
def display_calendar_spreads_table(opportunities: List[CalendarSpreadOpportunity]):
    """Display sortable table of calendar spread opportunities"""

    df = pd.DataFrame([
        {
            'Rank': opp.rank,
            'Symbol': opp.symbol,
            'Stock $': opp.stock_price,
            'Strike': opp.near_strike,
            'Near Exp': opp.near_expiration.strftime('%m/%d'),
            'Far Exp': opp.far_expiration.strftime('%m/%d'),
            'Net Debit': opp.net_debit,
            'Max Profit': opp.max_profit,
            'Profit %': opp.profit_potential * 100,
            'Prob Profit': opp.probability_profit,
            'Score': opp.opportunity_score,
            'Chart': f"https://www.tradingview.com/chart/?symbol={opp.symbol}"
        }
        for opp in opportunities
    ])

    st.dataframe(
        df,
        hide_index=True,
        use_container_width=True,
        column_config={
            "Rank": st.column_config.NumberColumn("Rank", width="small"),
            "Symbol": st.column_config.TextColumn("Symbol", width="small"),
            "Stock $": st.column_config.NumberColumn("Stock $", format="$%.2f"),
            "Strike": st.column_config.NumberColumn("Strike", format="$%.2f"),
            "Net Debit": st.column_config.NumberColumn("Debit", format="$%.2f"),
            "Max Profit": st.column_config.NumberColumn("Max Profit", format="$%.2f"),
            "Profit %": st.column_config.NumberColumn("Profit %", format="%.1f%%"),
            "Prob Profit": st.column_config.ProgressColumn(
                "Prob Profit",
                format="%.0f%%",
                min_value=0,
                max_value=100
            ),
            "Score": st.column_config.ProgressColumn(
                "Score",
                format="%.0f",
                min_value=0,
                max_value=100
            ),
            "Chart": st.column_config.LinkColumn("Chart", display_text="📈")
        }
    )
```

---

## Phase 4: Additional Option Strategies

### Strategies to Add

#### 1. **Iron Condor**
- Sell OTM call + put (collect premium)
- Buy further OTM call + put (limit risk)
- Profit from stock staying in range
- High probability, limited profit

#### 2. **Butterfly Spread**
- Sell 2 ATM options
- Buy 1 ITM + 1 OTM option
- Profit from minimal stock movement
- Low cost, limited profit/loss

#### 3. **Vertical Spreads**
- **Bull Call Spread**: Buy lower strike call, sell higher strike call
- **Bear Put Spread**: Buy higher strike put, sell lower strike put
- Directional bets with defined risk

#### 4. **Diagonal Spread**
- Like calendar spread but different strikes
- Sell near-term OTM option
- Buy far-term further OTM option
- Directional + time decay

#### 5. **Ratio Spread**
- Buy 1 option, sell 2-3 options at different strike
- Profit from specific price movement
- Unlimited risk on one side

### Strategy Selector

```python
st.selectbox("Select Strategy", [
    "Calendar Spread",
    "Iron Condor",
    "Butterfly Spread",
    "Bull Call Spread",
    "Bear Put Spread",
    "Diagonal Spread",
    "Ratio Spread"
])
```

### Universal Strategy Evaluator

```python
@dataclass
class StrategyOpportunity:
    """Universal data model for any option strategy"""
    symbol: str
    strategy_name: str
    stock_price: float

    # Legs (can have 2-4 legs)
    legs: List[OptionLeg]

    # Metrics
    net_cost: float  # Debit or credit
    max_profit: float
    max_loss: float
    breakeven: Union[float, tuple]  # Single or range
    probability_profit: float

    # Greeks
    delta: float
    theta: float
    vega: float
    gamma: float

    # Scoring
    opportunity_score: float
    rank: int

@dataclass
class OptionLeg:
    """Single option leg in a strategy"""
    action: str  # 'buy' or 'sell'
    option_type: str  # 'call' or 'put'
    strike: float
    expiration: date
    premium: float
    quantity: int
    iv: float
```

---

## Phase 5: UI Integration

### TradingView Watchlists Page Layout

```
┌─────────────────────────────────────────────────────────────┐
│ TradingView Watchlists                                      │
├─────────────────────────────────────────────────────────────┤
│ [📋 Watchlists] [🔄 Auto-Sync] [📆 Calendar Spreads] [📊 Strategies] │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│ 📆 Calendar Spreads Tab                                     │
├─────────────────────────────────────────────────────────────┤
│ Select Watchlist: [High IV Wheel Stocks ▼]                 │
│ Strategy: [Calendar Spread ▼]                              │
│                                                             │
│ Parameters:                                                 │
│ ├─ Near-term DTE: [30-45 days]                            │
│ ├─ Far-term DTE: [60-90 days]                             │
│ ├─ Min Liquidity: [100 contracts/day]                     │
│ └─ Strike Range: [±10% from current price]                │
│                                                             │
│ [🔍 Find Opportunities] (scans all symbols in watchlist)  │
├─────────────────────────────────────────────────────────────┤
│ 📊 Opportunities Found: 15                                  │
├─────────────────────────────────────────────────────────────┤
│ Rank │ Symbol │ Strike │ Debit │ Profit │ % │ Prob │ Score │
│   1  │  AAPL  │  $180  │ $250  │  $450  │180%│ 65% │  92   │
│   2  │  NVDA  │  $850  │ $450  │  $700  │156%│ 58% │  87   │
│   3  │  TSLA  │  $240  │ $320  │  $480  │150%│ 62% │  85   │
│  ... │  ...   │  ...   │  ...  │  ...   │...│ ... │  ...  │
├─────────────────────────────────────────────────────────────┤
│ 🤖 AI Research: [AAPL] [NVDA] [TSLA] [AMD] [MSFT]         │
└─────────────────────────────────────────────────────────────┘
```

---

## Implementation Timeline

| Phase | Task | Time | Status |
|-------|------|------|--------|
| 1 | Extract modular AI Research | 1 hour | Pending |
| 2 | Move Calendar Spreads to Watchlists | 30 min | Pending |
| 3 | Build Calendar Spread Evaluator | 2 hours | Pending |
| 4 | Add Iron Condor Strategy | 1 hour | Pending |
| 4 | Add Butterfly Strategy | 1 hour | Pending |
| 4 | Add Vertical Spreads | 1 hour | Pending |
| 5 | Create sortable tables | 30 min | Pending |
| 5 | Testing & Documentation | 1 hour | Pending |

**Total**: ~8 hours

---

## Success Criteria

✅ AI Research widget is modular and reusable
✅ Calendar Spreads integrated into TradingView Watchlists
✅ Evaluates all symbols in selected watchlist
✅ Calculates profit potential and probability
✅ Sortable table by score/profit/probability
✅ At least 4 option strategies implemented
✅ AI Research available on all strategy pages
✅ Fast performance (<5 seconds for full watchlist scan)

---

## Technical Stack

- **Options Data**: yfinance + mibian for Greeks
- **Probability Calculations**: Black-Scholes + Monte Carlo
- **UI**: Streamlit with sortable dataframes
- **Caching**: Redis for options chains (5-min TTL)
- **AI Research**: Existing modular widget
- **Database**: PostgreSQL for historical tracking

---

## Files to Create/Modify

### New Files
- `src/components/ai_research_widget.py` - Modular AI Research
- `src/strategies/calendar_spread_finder.py` - Calendar spread logic
- `src/strategies/iron_condor_finder.py` - Iron condor logic
- `src/strategies/butterfly_finder.py` - Butterfly logic
- `src/strategies/vertical_spread_finder.py` - Vertical spreads
- `src/strategies/strategy_evaluator.py` - Universal evaluator
- `src/strategies/__init__.py` - Package exports

### Modified Files
- `dashboard.py` - Remove Calendar Spreads nav button
- TradingView Watchlists page - Add strategies tab
- `positions_page_improved.py` - Use modular AI widget

---

## Next Steps

1. Create modular AI Research widget
2. Extract and refactor existing AI Research code
3. Build calendar spread evaluator
4. Integrate into TradingView Watchlists
5. Add additional strategies one by one
6. Test with real data
7. Optimize performance
8. Document usage

# Trade History & Theta Decay Enhancement Plan

## User Requirements

### 1. Historical Trade Tracking
**Problem**: No visibility into closed positions, especially cash-secured puts closed early for profit
**Need**: Track all closed trades to measure progress and performance

**Required Features**:
- Record when positions are opened (sell put option)
- Record when positions are closed (buy back put option early OR let expire)
- Track profit/loss on each trade
- Show trade history on Dashboard
- Show trade history on TradingView Watchlists page
- Calculate cumulative P&L
- Show win rate, average profit, best/worst trades

### 2. Extended Theta Decay Chart
**Problem**: Theta decay only shows 7 days, but some positions go out 30-45 days
**Need**: Show theta decay all the way to expiration

**Required Features**:
- Extend theta chart from 7 days to full expiration (up to 60 days)
- Show cumulative theta decay value
- Highlight current day on chart
- Show projected profit if held to expiration

---

## Implementation Plan

### Phase 1: Database Schema for Trade History

Create `trade_history` table:

```sql
CREATE TABLE trade_history (
    id SERIAL PRIMARY KEY,
    symbol VARCHAR(10) NOT NULL,
    strategy_type VARCHAR(20) DEFAULT 'cash_secured_put',

    -- Opening trade
    open_date TIMESTAMP WITH TIME ZONE NOT NULL,
    strike_price DECIMAL(10, 2) NOT NULL,
    expiration_date DATE NOT NULL,
    premium_collected DECIMAL(10, 2) NOT NULL,
    contracts INTEGER DEFAULT 1,

    -- Closing trade (NULL if still open)
    close_date TIMESTAMP WITH TIME ZONE,
    close_price DECIMAL(10, 2),
    close_reason VARCHAR(20), -- 'early_close', 'expiration', 'assignment'

    -- P&L
    days_held INTEGER,
    profit_loss DECIMAL(10, 2),
    profit_loss_percent DECIMAL(10, 4),
    annualized_return DECIMAL(10, 4),

    -- Status
    status VARCHAR(20) DEFAULT 'open', -- 'open', 'closed', 'assigned'

    -- Metadata
    notes TEXT,
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

CREATE INDEX idx_trade_history_symbol ON trade_history(symbol);
CREATE INDEX idx_trade_history_status ON trade_history(status);
CREATE INDEX idx_trade_history_close_date ON trade_history(close_date DESC);
```

### Phase 2: Dashboard Trade History Section

Add new section to Dashboard page showing:

**Metrics Row**:
- Total Closed Trades
- Total P/L ($)
- Win Rate (%)
- Avg Days Held

**Trade History Table**:
Columns: Symbol, Open Date, Close Date, Strike, Premium, Close Price, Days Held, P/L, P/L %, Annualized Return, Status

**Charts**:
- Cumulative P/L over time (line chart)
- P/L by symbol (bar chart)
- Win/Loss distribution (pie chart)

**Filters**:
- Date range
- Symbol
- Status (all, closed, open)
- Min/Max P/L

### Phase 3: TradingView Watchlist Trade History

Add tab to TradingView Watchlists showing:
- Trade history for current watchlist symbols
- Quick stats per symbol
- Link to add new trade manually

### Phase 4: Manual Trade Entry Form

Add button to manually log trades:
- Symbol (dropdown from database)
- Open Date
- Strike Price
- Expiration Date
- Premium Collected
- Contracts
- [Optional] Close Date, Close Price, Reason

### Phase 5: Theta Decay Extension

**Current Code** (7 days):
```python
days_range = list(range(0, min(8, int(dte) + 1)))
```

**New Code** (all days to expiration):
```python
days_range = list(range(0, int(dte) + 1))  # All days to expiration
```

**Enhanced Chart**:
- X-axis: Days from now (0 to DTE)
- Y-axis: Option value ($)
- Green line: Theta decay curve
- Red vertical line: Current day marker
- Shaded area: Projected profit zone
- Annotations: DTE milestones (7d, 14d, 21d, 30d)

---

## File Changes Required

### New Files
1. `trade_history_manager.py` - Class to manage trade CRUD operations
2. `trade_history_schema.sql` - Database schema
3. `trade_history_form.py` - Streamlit form for manual entry

### Modified Files
1. `dashboard.py` - Add trade history section
2. `dashboard.py` - Extend theta decay chart (TradingView Watchlists section)
3. `dashboard.py` - Add trade history to TradingView Watchlists

---

## Detailed Implementation

### Trade History Manager Class

```python
class TradeHistoryManager:
    def __init__(self):
        self.db = TradingViewDBManager()

    def add_trade(self, symbol, strike, expiration, premium, contracts=1, open_date=None):
        """Open a new trade"""
        pass

    def close_trade(self, trade_id, close_price, close_reason='early_close'):
        """Close an existing trade"""
        pass

    def get_open_trades(self):
        """Get all open positions"""
        pass

    def get_closed_trades(self, limit=100, symbol=None, date_from=None):
        """Get trade history"""
        pass

    def get_trade_stats(self):
        """Calculate aggregate statistics"""
        return {
            'total_trades': 0,
            'total_pl': 0.0,
            'win_rate': 0.0,
            'avg_days_held': 0,
            'best_trade': {},
            'worst_trade': {}
        }

    def get_cumulative_pl(self):
        """Get cumulative P/L over time for charting"""
        pass
```

### Dashboard Trade History Section

Location: Right after the main dashboard metrics

```python
st.markdown("---")
st.subheader("📊 Trade History")

from trade_history_manager import TradeHistoryManager
th_mgr = TradeHistoryManager()

# Metrics
stats = th_mgr.get_trade_stats()
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Total Trades", stats['total_trades'])
with col2:
    st.metric("Total P/L", f"${stats['total_pl']:,.2f}")
with col3:
    st.metric("Win Rate", f"{stats['win_rate']:.1f}%")
with col4:
    st.metric("Avg Days Held", stats['avg_days_held'])

# Add trade button
if st.button("+ Add Trade", type="primary"):
    st.session_state.show_trade_form = True

# Trade form modal
if st.session_state.get('show_trade_form'):
    with st.form("trade_entry"):
        # Form fields...
        submitted = st.form_submit_button("Save Trade")

# Trade history table
trades = th_mgr.get_closed_trades(limit=50)
if trades:
    df = pd.DataFrame(trades)
    st.dataframe(df)
else:
    st.info("No closed trades yet. Click '+ Add Trade' to log your first trade!")
```

### Extended Theta Decay Chart

Location: TradingView Watchlists → Auto-Sync tab

Find current theta decay code and modify:

```python
# OLD CODE (7 days only):
days_range = list(range(0, min(8, int(dte) + 1)))

# NEW CODE (all days to expiration):
max_days = int(dte) if dte <= 60 else 60  # Cap at 60 days for chart readability
days_range = list(range(0, max_days + 1))

# Calculate theta decay values for all days
theta_values = []
for days_passed in days_range:
    days_remaining = dte - days_passed
    if days_remaining >= 0:
        # Exponential decay model
        value = premium * (days_remaining / dte) ** 1.5
        theta_values.append(value)
    else:
        theta_values.append(0)

# Enhanced chart
fig = go.Figure()

# Theta decay line
fig.add_trace(go.Scatter(
    x=days_range,
    y=theta_values,
    mode='lines+markers',
    name='Option Value',
    line=dict(color='green', width=3),
    fill='tozeroy',
    fillcolor='rgba(0, 255, 0, 0.1)'
))

# Current day marker
fig.add_vline(x=0, line_dash="dash", line_color="red", annotation_text="Today")

# Milestone annotations
for milestone in [7, 14, 21, 30]:
    if milestone <= max_days:
        fig.add_vline(x=milestone, line_dash="dot", line_color="gray", opacity=0.3)

fig.update_layout(
    title=f"Theta Decay: {symbol} ${strike} Put (${premium} premium, {dte} DTE)",
    xaxis_title="Days from Now",
    yaxis_title="Option Value ($)",
    hovermode='x unified'
)

st.plotly_chart(fig, use_container_width=True)
```

---

## Testing Checklist

- [ ] Trade history table created successfully
- [ ] Can add new trade manually
- [ ] Can close trade and calculate P/L correctly
- [ ] Trade history displays on Dashboard
- [ ] Trade history displays on TradingView Watchlists
- [ ] Theta decay shows full expiration period (not just 7 days)
- [ ] Charts render correctly
- [ ] Statistics calculate correctly
- [ ] Export trade history to CSV works

---

## Future Enhancements (v2)

1. **Robinhood Integration** - Auto-import trades from Robinhood API
2. **Trade Alerts** - Notify when position hits profit target
3. **Performance Analytics** - Compare to benchmarks (SPY, QQQ)
4. **Tax Reporting** - Export trades for tax purposes
5. **Trade Journal** - Add notes/tags to trades
6. **Strategy Comparison** - Compare CSP vs Covered Call performance
7. **Risk Metrics** - Max drawdown, Sharpe ratio, win streaks

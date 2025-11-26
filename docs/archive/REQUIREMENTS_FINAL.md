# Final Requirements - TradingView Watchlists Options Table

## What You Want (Based on Your Feedback)

### PRIMARY TABLE (Main View)
A **sortable table** showing cash-secured put opportunities with these columns:

| Symbol | Stock Price | 7 DTE | 14 DTE | 30 DTE | Score | [Expand] |
|--------|-------------|-------|--------|--------|-------|----------|
| AAPL   | $150.00     | $2.50 | $3.20  | $5.00  | 85    | [+]      |
| NVDA   | $450.00     | $8.00 | $12.50 | $18.00 | 92    | [+]      |

**Requirements:**
- ONE row per stock symbol
- Shows best premium for 7, 14, and 30 DTE
- Click column headers to sort
- Click [+] button or anywhere on row to expand

### FILTERS (Above Table)
```
[Min Stock Price: $___] [Max Stock Price: $___] [Delta Range: 0.__ to 0.__]
[Min Premium: $___] [Min Monthly %: ___%] [Max DTE: ___]
```

All filters apply when you change them (live filtering).

### EXPANDED ROW VIEW (When You Click a Row)

When you click AAPL row, show below it:

```
┌─────────────────────────────────────────────────────────────────────────┐
│ AAPL - Stock Price: $150.00                                    [Close X]│
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  📅 7 Days to Expiration                                                │
│  ┌────────┬─────────┬───────┬──────┬──────┬──────┬────────┬──────┐    │
│  │ Strike │ Premium │  Bid  │ Ask  │Delta │ IV % │ Volume │  OI  │    │
│  ├────────┼─────────┼───────┼──────┼──────┼──────┼────────┼──────┤    │
│  │ $148   │  $1.80  │ $1.75 │$1.85 │-0.28 │ 24.5 │  450   │ 230  │    │
│  │ $146   │  $2.20  │ $2.15 │$2.25 │-0.32 │ 26.2 │  890   │ 450  │    │
│  │ $144   │  $2.80  │ $2.75 │$2.85 │-0.36 │ 28.1 │ 1100   │ 670  │    │
│  └────────┴─────────┴───────┴──────┴──────┴──────┴────────┴──────┘    │
│                                                                          │
│  📅 14 Days to Expiration                                               │
│  ┌────────┬─────────┬───────┬──────┬──────┬──────┬────────┬──────┐    │
│  │ Strike │ Premium │  Bid  │ Ask  │Delta │ IV % │ Volume │  OI  │    │
│  ├────────┼─────────┼───────┼──────┼──────┼──────┼────────┼──────┤    │
│  │ $148   │  $2.80  │ $2.75 │$2.85 │-0.29 │ 25.8 │  560   │ 340  │    │
│  │ $146   │  $3.40  │ $3.35 │$3.45 │-0.33 │ 27.5 │  780   │ 480  │    │
│  │ $144   │  $4.10  │ $4.05 │$4.15 │-0.37 │ 29.3 │  920   │ 590  │    │
│  └────────┴─────────┴───────┴──────┴──────┴──────┴────────┴──────┘    │
│                                                                          │
│  📅 30 Days to Expiration                                               │
│  ┌────────┬─────────┬───────┬──────┬──────┬──────┬────────┬──────┐    │
│  │ Strike │ Premium │  Bid  │ Ask  │Delta │ IV % │ Volume │  OI  │    │
│  ├────────┼─────────┼───────┼──────┼──────┼──────┼────────┼──────┤    │
│  │ $148   │  $4.50  │ $4.45 │$4.55 │-0.30 │ 26.5 │  890   │ 560  │    │
│  │ $146   │  $5.40  │ $5.35 │$5.45 │-0.34 │ 28.2 │ 1200   │ 780  │    │
│  │ $144   │  $6.50  │ $6.45 │$6.55 │-0.38 │ 30.1 │ 1450   │ 920  │    │
│  └────────┴─────────┴───────┴──────┴──────┴──────┴────────┴──────┘    │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

**Key Points:**
- Show **multiple strike prices** for each DTE (not just one)
- Show strikes around 0.3 delta (e.g., 0.28, 0.32, 0.36 delta strikes)
- Show ALL greeks and prices: Premium, Bid, Ask, Delta, IV, Volume, OI
- Group by DTE: 7 days, 14 days, 30 days
- Each group shows 3-5 strike price options

## What Got Lost

You mentioned "cash secured puts 7, 14, and 30 days out" - this needs to be the main table showing:
1. **7 DTE column** - best premium around 7 days
2. **14 DTE column** - best premium around 14 days
3. **30 DTE column** - best premium around 30 days

And when you expand, you see ALL the strikes for each DTE, not just the best one.

## Database Schema Reference

Based on `stock_premiums` table:
- `symbol` - Stock ticker
- `current_price` - Stock price (from stock_data join)
- `strike_price` - Put strike
- `dte` - Days to expiration
- `premium` - Option premium (mid price)
- `bid` - Bid price
- `ask` - Ask price
- `delta` - Option delta
- `implied_volatility` - IV %
- `volume` - Daily volume
- `open_interest` - Open interest
- `monthly_return` - Monthly return %

## Implementation Plan

### 1. Main Table Query
```sql
-- Get best option for each symbol at 7, 14, and 30 DTE
WITH Options7 AS (
    SELECT symbol, premium, delta
    FROM stock_premiums
    WHERE dte BETWEEN 5 AND 9
    AND ABS(delta) BETWEEN 0.25 AND 0.35
    ORDER BY monthly_return DESC
    LIMIT 1 PER symbol
),
Options14 AS (
    SELECT symbol, premium, delta
    FROM stock_premiums
    WHERE dte BETWEEN 12 AND 16
    AND ABS(delta) BETWEEN 0.25 AND 0.35
    ORDER BY monthly_return DESC
    LIMIT 1 PER symbol
),
Options30 AS (
    SELECT symbol, premium, delta
    FROM stock_premiums
    WHERE dte BETWEEN 28 AND 32
    AND ABS(delta) BETWEEN 0.25 AND 0.35
    ORDER BY monthly_return DESC
    LIMIT 1 PER symbol
)
SELECT
    sd.symbol,
    sd.current_price,
    o7.premium as premium_7dte,
    o14.premium as premium_14dte,
    o30.premium as premium_30dte,
    (calculate score based on premiums and deltas)
FROM stock_data sd
LEFT JOIN Options7 o7 ON sd.symbol = o7.symbol
LEFT JOIN Options14 o14 ON sd.symbol = o14.symbol
LEFT JOIN Options30 o30 ON sd.symbol = o30.symbol
WHERE sd.symbol = ANY(%s)  -- watchlist symbols
AND sd.current_price BETWEEN %s AND %s  -- price filter
```

### 2. Expanded View Query
```sql
-- When user clicks AAPL, get ALL strikes for 7, 14, 30 DTE
SELECT
    strike_price,
    dte,
    premium,
    bid,
    ask,
    delta,
    implied_volatility,
    volume,
    open_interest
FROM stock_premiums
WHERE symbol = 'AAPL'
AND dte IN (
    -- Get closest to 7 days
    (SELECT dte FROM stock_premiums WHERE symbol = 'AAPL' ORDER BY ABS(dte - 7) LIMIT 1),
    -- Get closest to 14 days
    (SELECT dte FROM stock_premiums WHERE symbol = 'AAPL' ORDER BY ABS(dte - 14) LIMIT 1),
    -- Get closest to 30 days
    (SELECT dte FROM stock_premiums WHERE symbol = 'AAPL' ORDER BY ABS(dte - 30) LIMIT 1)
)
AND ABS(delta) BETWEEN 0.25 AND 0.40  -- Show range of deltas
ORDER BY dte, strike_price
```

### 3. Filters to Implement
- **Min Stock Price** - Filter stocks by price (e.g., only stocks > $50)
- **Max Stock Price** - Upper price limit (e.g., stocks < $500)
- **Delta Range** - Slider for delta range (default 0.25 - 0.35)
- **Min Premium** - Minimum option premium (e.g., > $2.00)
- **Min Monthly %** - Minimum monthly return % (e.g., > 3%)
- **Max DTE** - Maximum days to expiration to include

### 4. UI Components

**Filters Section:**
```python
col1, col2, col3 = st.columns(3)
with col1:
    min_stock_price = st.number_input("Min Stock Price", value=10.0)
    max_stock_price = st.number_input("Max Stock Price", value=1000.0)
with col2:
    delta_range = st.slider("Delta Range", 0.20, 0.40, (0.25, 0.35))
    min_premium = st.number_input("Min Premium", value=2.0)
with col3:
    min_monthly = st.number_input("Min Monthly %", value=3.0)
    max_dte = st.number_input("Max DTE", value=45)
```

**Main Table:**
```python
# Show table with 7, 14, 30 DTE columns
st.dataframe(
    df[['Symbol', 'Stock Price', '7 DTE', '14 DTE', '30 DTE', 'Score']],
    on_select="rerun",
    selection_mode="single-row"
)
```

**Expanded View:**
```python
if selected_symbol:
    st.markdown(f"### {selected_symbol} - Stock Price: ${stock_price}")

    # Show 7 DTE options
    st.markdown("#### 📅 7 Days to Expiration")
    st.dataframe(options_7dte)

    # Show 14 DTE options
    st.markdown("#### 📅 14 Days to Expiration")
    st.dataframe(options_14dte)

    # Show 30 DTE options
    st.markdown("#### 📅 30 Days to Expiration")
    st.dataframe(options_30dte)
```

## Success Criteria

✅ Main table shows: Symbol, Stock Price, 7 DTE premium, 14 DTE premium, 30 DTE premium, Score
✅ Table is sortable by clicking column headers
✅ Filters work: Stock price range, delta range, min premium, min monthly %, max DTE
✅ Click any row to expand
✅ Expanded view shows 3 sections: 7 DTE, 14 DTE, 30 DTE
✅ Each section shows multiple strikes with: Premium, Bid, Ask, Delta, IV, Volume, OI
✅ Show 3-5 strike prices per DTE group
✅ All prices and greeks are visible

## What NOT to Do

❌ Don't remove the table
❌ Don't make it card-based
❌ Don't show only one option per stock in expanded view
❌ Don't hide the greeks (delta, IV)
❌ Don't hide bid/ask prices
❌ Don't make filters optional - they must all be there

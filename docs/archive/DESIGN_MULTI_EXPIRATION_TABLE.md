# Multi-Expiration Options Table - Design Document

## Current Problem

The current implementation **broke** the existing table functionality:
- ❌ Removed the sortable table
- ❌ Removed the filtering system (min premium, max DTE, delta range, min monthly %)
- ❌ Changed to expandable cards instead of keeping table format
- ❌ Lost ability to sort by any column

## User Requirements (Correctly Interpreted)

### What User Actually Wants

1. **KEEP the existing sortable table** - This is the PRIMARY interface
2. **KEEP all existing filters working** (min premium, max DTE, delta range, min monthly %)
3. **ADD expandable row functionality** - Click a row to see MORE options for that same stock
4. **Show multiple expirations per stock** - 7, 14, 30, 45 days out
5. **Show multiple strikes around 0.3 delta** for each expiration
6. **AI analysis capability** - Ask which options are best

### What Should NOT Change

- ❌ Do NOT remove the table
- ❌ Do NOT remove filtering
- ❌ Do NOT remove sorting
- ❌ Do NOT change to card/expander-based primary view

## Proposed Architecture

### 1. Main Table (PRIMARY VIEW - UNCHANGED)

**Keep existing functionality:**
```
[Filters: Min Premium | Max DTE | Delta Range | Min Monthly %]

| Symbol | Price | Strike | DTE | Premium | Monthly % | Delta | IV | Bid | Ask | Volume | OI | [+] |
|--------|-------|--------|-----|---------|-----------|-------|----|----|-----|--------|----|----|
| AAPL   | 150   | 145    | 30  | 2.50    | 5.2%      | 0.32  | 25 | 2.45| 2.55| 1200  | 500| [+]|
| NVDA   | 450   | 440    | 28  | 8.75    | 6.8%      | 0.30  | 45 | 8.70| 8.80| 3400  | 890| [+]|
...
```

**Behavior:**
- Fully sortable by ANY column (click headers)
- Filters apply to visible rows
- Shows ONE "best" option per stock (highest monthly %)
- New column: `[+]` button to expand and see more options

### 2. Expanded Row View (NEW FUNCTIONALITY)

**When user clicks `[+]` button on a row:**

```
| Symbol | Price | Strike | DTE | Premium | Monthly % | Delta | IV | Bid | Ask | Volume | OI | [-] |
|--------|-------|--------|-----|---------|-----------|-------|----|----|-----|--------|----|----|
| AAPL   | 150   | 145    | 30  | 2.50    | 5.2%      | 0.32  | 25 | 2.45| 2.55| 1200  | 500| [-]|

    ┌─────────────────────────────────────────────────────────────────────────────────────┐
    │ Additional Options for AAPL (Stock Price: $150.00)                                  │
    ├─────────────────────────────────────────────────────────────────────────────────────┤
    │                                                                                      │
    │  7-14 Days Out:                                                                      │
    │  ┌───────┬──────┬─────────┬───────────┬───────┬────┬────┬────────┬──────┐         │
    │  │Strike │ DTE  │ Premium │ Monthly % │ Delta │ Bid│Ask │ Volume │  OI  │         │
    │  ├───────┼──────┼─────────┼───────────┼───────┼────┼────┼────────┼──────┤         │
    │  │ 148   │  10  │  1.85   │   4.8%    │ 0.28  │1.80│1.90│   450  │  230 │         │
    │  │ 146   │  12  │  2.10   │   5.1%    │ 0.31  │2.05│2.15│   890  │  450 │         │
    │  │ 144   │  14  │  2.35   │   5.3%    │ 0.34  │2.30│2.40│  1100  │  670 │         │
    │  └───────┴──────┴─────────┴───────────┴───────┴────┴────┴────────┴──────┘         │
    │                                                                                      │
    │  15-30 Days Out:                                                                     │
    │  ┌───────┬──────┬─────────┬───────────┬───────┬────┬────┬────────┬──────┐         │
    │  │Strike │ DTE  │ Premium │ Monthly % │ Delta │ Bid│Ask │ Volume │  OI  │         │
    │  ├───────┼──────┼─────────┼───────────┼───────┼────┼────┼────────┼──────┤         │
    │  │ 147   │  21  │  3.20   │   5.5%    │ 0.29  │3.15│3.25│   670  │  340 │         │
    │  │ 145   │  30  │  2.50   │   5.2%    │ 0.32  │2.45│2.55│  1200  │  500 │ ← MAIN  │
    │  │ 143   │  28  │  3.85   │   5.7%    │ 0.35  │3.80│3.90│   890  │  450 │         │
    │  └───────┴──────┴─────────┴───────────┴───────┴────┴────┴────────┴──────┘         │
    │                                                                                      │
    │  31-45 Days Out:                                                                     │
    │  ┌───────┬──────┬─────────┬───────────┬───────┬────┬────┬────────┬──────┐         │
    │  │Strike │ DTE  │ Premium │ Monthly % │ Delta │ Bid│Ask │ Volume │  OI  │         │
    │  ├───────┼──────┼─────────┼───────────┼───────┼────┼────┼────────┼──────┤         │
    │  │ 148   │  35  │  4.20   │   5.4%    │ 0.30  │4.15│4.25│   560  │  280 │         │
    │  │ 146   │  42  │  4.85   │   5.6%    │ 0.33  │4.80│4.90│   780  │  390 │         │
    │  └───────┴──────┴─────────┴───────────┴───────┴────┴────┴────────┴──────┘         │
    │                                                                                      │
    │  [Close Details]                                                                     │
    └─────────────────────────────────────────────────────────────────────────────────────┘
```

**Expanded View Details:**
- Shows grouped by DTE ranges: 7-14, 15-30, 31-45 days
- Each group shows multiple strikes around 0.3 delta
- Indicates which option is shown in main table (← MAIN marker)
- Click `[-]` button to collapse back to single row

### 3. AI Analysis (SEPARATE SECTION)

**Location:** Above the table (not inside expanded rows)

```
┌─────────────────────────────────────────────────────────────────────┐
│ 🤖 AI Options Analyzer                                              │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  Ask a question about the options:                                  │
│  [Which options are best for making money in the next 30 days?___] │
│                                                                      │
│  [Analyze Options]                                                  │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘

Results: (shown below after clicking Analyze)
┌─────────────────────────────────────────────────────────────────────┐
│ Top 5 Recommendations for "best 30-day plays":                      │
├─────────────────────────────────────────────────────────────────────┤
│ 1. NVDA $440 strike, 28 DTE - 6.8% monthly, 0.30 delta             │
│ 2. AAPL $145 strike, 30 DTE - 5.2% monthly, 0.32 delta             │
│ ...                                                                  │
└─────────────────────────────────────────────────────────────────────┘
```

## Technical Implementation Plan

### Phase 1: Database Query Changes

**Current Query (WRONG):**
```python
# Gets ALL options for ALL stocks - too much data
query = """
    SELECT ... FROM stock_premiums sp
    JOIN stock_data sd ON sp.symbol = sd.symbol
    WHERE sp.dte BETWEEN 7 AND 45
    AND sp.delta BETWEEN 0.25 AND 0.35
"""
```

**New Queries (CORRECT):**

**Query 1 - Main Table Data (Fast):**
```python
# Get ONLY the best option per stock for main table
query_main = """
    WITH RankedOptions AS (
        SELECT
            sp.*,
            sd.current_price as stock_price,
            ROW_NUMBER() OVER (
                PARTITION BY sp.symbol
                ORDER BY sp.monthly_return DESC
            ) as rn
        FROM stock_premiums sp
        JOIN stock_data sd ON sp.symbol = sd.symbol
        WHERE sp.dte BETWEEN {min_dte} AND {max_dte}
        AND sp.delta BETWEEN {min_delta} AND {max_delta}
        AND sp.premium >= {min_premium}
        AND sp.monthly_return >= {min_monthly}
    )
    SELECT * FROM RankedOptions WHERE rn = 1
"""
```

**Query 2 - Expanded Row Data (On-Demand):**
```python
# Get ALL options for specific symbol when user expands row
query_expanded = """
    SELECT
        sp.strike,
        sp.dte,
        sp.premium,
        sp.monthly_return,
        sp.delta,
        sp.iv,
        sp.bid,
        sp.ask,
        sp.volume,
        sp.open_interest
    FROM stock_premiums sp
    WHERE sp.symbol = %s
    AND sp.dte BETWEEN 7 AND 45
    AND sp.delta BETWEEN 0.25 AND 0.35
    ORDER BY sp.dte, sp.strike
"""
```

### Phase 2: UI State Management

**Add Session State for Row Expansion:**
```python
# Track which rows are expanded
if 'expanded_rows' not in st.session_state:
    st.session_state.expanded_rows = set()

def toggle_row(symbol):
    if symbol in st.session_state.expanded_rows:
        st.session_state.expanded_rows.remove(symbol)
    else:
        st.session_state.expanded_rows.add(symbol)
```

### Phase 3: Table Rendering

**Main Table with Expand Buttons:**
```python
# Add expand button column to dataframe
df['Expand'] = df['Symbol'].apply(
    lambda s: '[-]' if s in st.session_state.expanded_rows else '[+]'
)

# Display table with column config
st.dataframe(
    df,
    column_config={
        'Expand': st.column_config.TextColumn('', width=50)
    },
    on_select='rerun',  # Detect row clicks
    use_container_width=True,
    hide_index=True
)

# Handle row expansion
selected_rows = st.session_state.get('dataframe_selected_rows', [])
for row_idx in selected_rows:
    symbol = df.iloc[row_idx]['Symbol']

    if symbol in st.session_state.expanded_rows:
        # Show expanded details
        expanded_data = fetch_expanded_options(symbol)
        render_expanded_view(symbol, expanded_data)
```

### Phase 4: Expanded View Rendering

**Group by DTE and Display:**
```python
def render_expanded_view(symbol, options_df):
    st.markdown(f"#### Additional Options for {symbol}")

    # Group by DTE ranges
    dte_groups = {
        '7-14 Days': options_df[(options_df['DTE'] >= 7) & (options_df['DTE'] <= 14)],
        '15-30 Days': options_df[(options_df['DTE'] >= 15) & (options_df['DTE'] <= 30)],
        '31-45 Days': options_df[(options_df['DTE'] >= 31) & (options_df['DTE'] <= 45)]
    }

    for group_name, group_df in dte_groups.items():
        if not group_df.empty:
            st.markdown(f"**{group_name}:**")
            st.dataframe(
                group_df[['Strike', 'DTE', 'Premium', 'Monthly %',
                         'Delta', 'Bid', 'Ask', 'Volume', 'OI']],
                hide_index=True
            )
```

## Implementation Todos

### Todo List

1. **Database Query Refactoring**
   - [ ] Create `get_main_table_options()` function - returns ONE best option per stock
   - [ ] Create `get_expanded_options(symbol)` function - returns ALL options for specific stock
   - [ ] Add proper numeric type conversion after query (pd.to_numeric)
   - [ ] Test both queries with real database

2. **Session State Management**
   - [ ] Add `expanded_rows` set to session state
   - [ ] Create `toggle_row(symbol)` function
   - [ ] Ensure state persists across reruns

3. **Main Table UI**
   - [ ] Keep existing filter inputs (min premium, max DTE, delta range, min monthly %)
   - [ ] Keep existing table display with st.dataframe()
   - [ ] Add '[+]' expand button column
   - [ ] Ensure all columns remain sortable
   - [ ] Apply filters to main query

4. **Expanded Row UI**
   - [ ] Detect row clicks/expand button clicks
   - [ ] Fetch expanded data only when row is expanded (lazy loading)
   - [ ] Group options by DTE ranges (7-14, 15-30, 31-45)
   - [ ] Display each group as sub-table
   - [ ] Add '[-]' collapse button

5. **AI Analysis Section**
   - [ ] Keep existing AI analyzer above table
   - [ ] Query should analyze ALL available options (not just visible ones)
   - [ ] Display top 5 results with full details

6. **Testing & Validation**
   - [ ] Test that filters work correctly
   - [ ] Test that sorting works on all columns
   - [ ] Test expand/collapse functionality
   - [ ] Test with real database data
   - [ ] Verify performance (main table should load fast <1s)
   - [ ] Verify expanded rows load quickly (<500ms per symbol)

## Data Flow

```
User Opens Page
    ↓
Execute Query 1 (get_main_table_options)
    ↓
Display Main Table (ONE row per stock)
    ↓
User Applies Filters → Re-execute Query 1 with new parameters
    ↓
User Clicks Sort → Streamlit handles (no re-query needed)
    ↓
User Clicks [+] on AAPL row
    ↓
Execute Query 2 (get_expanded_options('AAPL'))
    ↓
Render Expanded View below AAPL row
    ↓
User Clicks [-] → Remove expanded view, keep main table
```

## Performance Considerations

1. **Main Table Load Time:** <1 second
   - Query returns ~150 rows (one per stock)
   - Existing indexes optimize this query

2. **Expanded Row Load Time:** <500ms per symbol
   - Query returns ~20-50 rows per symbol
   - Lazy loading - only fetch when expanded
   - Cache expanded data in session state

3. **Memory Usage:** Minimal
   - Main table: ~150 rows × 12 columns = ~1,800 cells
   - Expanded rows: Only loaded symbols (~30 rows × 9 columns per symbol)
   - Total: <100KB in memory

## Success Criteria

✅ **Must Have:**
- [ ] Main table shows ONE best option per stock
- [ ] All existing filters work (min premium, max DTE, delta range, min monthly %)
- [ ] All columns are sortable by clicking headers
- [ ] Clicking [+] shows multiple options grouped by DTE
- [ ] Expanded view shows 7-14, 15-30, 31-45 day groups
- [ ] Multiple strikes around 0.3 delta visible in expanded view
- [ ] All option prices shown (premium, bid, ask)
- [ ] AI analysis works and shows top 5 recommendations
- [ ] No breaking changes to existing functionality

✅ **Performance:**
- [ ] Main table loads in <1 second
- [ ] Expanded rows load in <500ms
- [ ] Sorting is instant (client-side)
- [ ] Filtering triggers re-query in <1 second

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│ TradingView Watchlists Page (Auto-Sync Tab)                    │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌────────────────────────────────────────────────────────┐    │
│  │ 🤖 AI Options Analyzer                                 │    │
│  │ [Ask question...] [Analyze]                            │    │
│  └────────────────────────────────────────────────────────┘    │
│                                                                  │
│  ┌────────────────────────────────────────────────────────┐    │
│  │ Filters                                                 │    │
│  │ Min Premium: [2.0] Max DTE: [45] Delta: [0.25-0.35]   │    │
│  │ Min Monthly %: [3.0]                                    │    │
│  └────────────────────────────────────────────────────────┘    │
│                                                                  │
│  ┌────────────────────────────────────────────────────────┐    │
│  │ Main Options Table (Sortable)                          │    │
│  │ ┌──────┬───────┬────────┬─────┬─────────┬────┬────┐   │    │
│  │ │Symbol│ Price │ Strike │ DTE │ Premium │... │[+] │   │    │
│  │ ├──────┼───────┼────────┼─────┼─────────┼────┼────┤   │    │
│  │ │ AAPL │ 150   │ 145    │ 30  │  2.50   │... │[+] │   │    │
│  │ │ NVDA │ 450   │ 440    │ 28  │  8.75   │... │[-] │ ←─┼─── Expanded
│  │ │      │       │        │     │         │    │    │   │
│  │ │      ┌─────────────────────────────────────────┐   │   │
│  │ │      │ Expanded: NVDA Options                  │   │   │
│  │ │      │ 7-14 Days: [3 options]                  │   │   │
│  │ │      │ 15-30 Days: [5 options]                 │   │   │
│  │ │      │ 31-45 Days: [4 options]                 │   │   │
│  │ │      └─────────────────────────────────────────┘   │   │
│  │ │      │       │        │     │         │    │    │   │
│  │ │ TSLA │ 210   │ 205    │ 35  │  4.20   │... │[+] │   │    │
│  │ └──────┴───────┴────────┴─────┴─────────┴────┴────┘   │    │
│  └────────────────────────────────────────────────────────┘    │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

## Risk Assessment

### High Risk
- **Breaking existing filters** - Mitigation: Keep query structure, only modify SELECT clause
- **Performance degradation** - Mitigation: Use two-query approach, lazy load expanded rows

### Medium Risk
- **UI complexity with expandable rows** - Mitigation: Use session state properly, test thoroughly
- **Expanded row state management** - Mitigation: Simple set-based tracking

### Low Risk
- **AI analysis accuracy** - Already working, no changes needed
- **Database connection** - Already working, no schema changes

## Next Steps

1. **Review this document** - User approval required before implementation
2. **Implement Phase 1** - Database query refactoring
3. **Implement Phase 2** - Session state management
4. **Implement Phase 3** - Main table UI (no breaking changes)
5. **Implement Phase 4** - Expanded row UI
6. **Test thoroughly** - All filters, sorting, expansion
7. **QA review** - Full integration test

---

**IMPORTANT:** No code changes will be made until this design is approved. This prevents breaking changes and ensures we build exactly what the user needs.

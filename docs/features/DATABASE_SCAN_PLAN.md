# Database Scan Implementation Plan

## Current Problem
Both "TradingView Watchlists" and "Database Scan" are pulling from the same `stock_premiums` table, making them identical. User wants Database Scan to be a SEPARATE interface that shows ALL 1,205 stocks from the `stocks` table.

## Current State Analysis

### TradingView Watchlists → Auto-Sync Tab
**Data Source**:
- Queries `stock_premiums` table
- Filters by `WHERE sp.symbol = ANY(%s)` (only shows symbols from selected watchlist)
- Shows ~114-150 stocks (only NVDA watchlist symbols)

**Query**:
```sql
SELECT DISTINCT ON (sp.symbol) ...
FROM stock_premiums sp
WHERE sp.symbol = ANY(%s)  -- ← FILTERS by watchlist
  AND sp.dte BETWEEN 28 AND 32
  AND ABS(sp.delta) BETWEEN 0.25 AND 0.40
```

### Database Scan → Scan Premiums Tab (CURRENT - WRONG)
**Data Source**:
- Also queries `stock_premiums` table
- NO watchlist filter - shows ALL stocks in stock_premiums
- Shows same ~114 stocks (only stocks that have been synced)

**Query**:
```sql
SELECT DISTINCT ON (sp.symbol) ...
FROM stock_premiums sp
WHERE sp.dte BETWEEN 29 AND 33  -- ← NO watchlist filter, but...
  AND ABS(sp.delta) BETWEEN 0.25 AND 0.40
-- Still only shows stocks that have been synced to stock_premiums!
```

**Problem**: Both show the same data because `stock_premiums` only has 89 unique stocks synced.

## What User Actually Wants

**Database Scan should**:
1. Show ALL 1,205 stocks from `stocks` table
2. Have a sync button to fetch options for ALL of them
3. Display premiums in same format as TradingView watchlist
4. Allow filtering and sorting to find best premiums across ALL stocks

## Implementation Plan

### Phase 1: Verify Current Data (5 min)
- [ ] Check how many stocks are in `stocks` table (should be 1,205)
- [ ] Check how many stocks are in `stock_premiums` table (currently 89-114)
- [ ] Confirm the gap: 1,205 - 89 = 1,116 stocks need syncing

### Phase 2: Create Sync for All Database Stocks (10 min)
- [ ] Create script to sync ALL 999 stocks from `stocks` table
- [ ] Run sync in background (will take 10-15 minutes)
- [ ] Monitor progress and ensure it's populating `stock_premiums`

### Phase 3: Verify Sync Completed (5 min)
- [ ] Check `stock_premiums` row count (should increase from 884 to 5000+)
- [ ] Check unique symbols in `stock_premiums` (should increase from 89 to 500+)
- [ ] Verify new stocks have options data

### Phase 4: Test Both Interfaces (5 min)
- [ ] Open TradingView Watchlists → Auto-Sync → Select "NVDA" watchlist
  - Should show ~150 stocks (only NVDA watchlist symbols)
- [ ] Open Database Scan → Scan Premiums
  - Should show 500+ stocks (ALL synced stocks from database)
- [ ] Confirm they show different numbers

### Phase 5: Review with Code Agent (5 min)
- [ ] Document what each section does
- [ ] Confirm Database Scan is pulling from ALL stocks
- [ ] Verify sync button works

## Success Criteria

✅ **TradingView Watchlists**: Shows 114-150 stocks (filtered by selected watchlist)
✅ **Database Scan**: Shows 500+ stocks (ALL stocks from stocks table that have options)
✅ **Sync Button**: Works in Database Scan to fetch options for all 1,205 stocks
✅ **Different Data**: The two sections show different stock counts

## Key Difference

| Feature | TradingView Watchlists | Database Scan |
|---------|----------------------|---------------|
| Data Source | `stock_premiums` filtered by watchlist | `stock_premiums` (all stocks) |
| Stock Count | ~150 (watchlist only) | 500+ (all database stocks) |
| Sync Button | Syncs selected watchlist | Syncs ALL 1,205 stocks |
| Purpose | Track specific watchlist | Find best premiums across entire universe |

## Current Action Required

1. **RUN SYNC NOW** for all 999 stocks in `stocks` table
2. **VERIFY** sync is populating `stock_premiums` table
3. **TEST** both interfaces show different data
4. **REVIEW** with code agent to confirm implementation

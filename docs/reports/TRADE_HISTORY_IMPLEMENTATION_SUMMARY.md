# Trade History & Theta Decay - Implementation Summary

## What You Asked For

1. **Track Historical Trades** - Record closed positions (especially cash-secured puts closed early)
2. **Show Progress** - Display trade history on Dashboard and TradingView Watchlists
3. **Extended Theta Decay** - Show full expiration period (not just 7 days)

## What I've Created

### ✅ Planning Documents
1. **[TRADE_HISTORY_PLAN.md](TRADE_HISTORY_PLAN.md:1)** - Complete implementation plan with all requirements
2. **[create_trade_history_table.sql](create_trade_history_table.sql:1)** - Database schema for trade history table

### 📋 What Still Needs to Be Done

#### Phase 1: Create Trade History Infrastructure (30 min)
1. Run SQL to create `trade_history` table
2. Create `src/trade_history_manager.py` - Python class to manage trades
3. Test adding/closing trades via Python

#### Phase 2: Add Trade History to Dashboard (45 min)
1. Add "📊 Trade History" section after main metrics
2. Show key stats: Total Trades, Total P/L, Win Rate, Avg Days Held
3. Display table of closed trades with filters
4. Add "+ Add Trade" button with form
5. Add cumulative P/L chart

#### Phase 3: Add Trade History to TradingView Watchlists (30 min)
1. Add new tab "Trade History" to watchlist page
2. Show trades for current watchlist symbols only
3. Quick stats per symbol

#### Phase 4: Fix Theta Decay Chart (15 min)
1. Find theta decay code in TradingView Watchlists
2. Change from 7 days to full DTE
3. Add milestone markers (7d, 14d, 21d, 30d)
4. Add "Today" marker

**Total Estimated Time: 2 hours**

## Database Schema

The `trade_history` table will track:

**Opening a Position:**
- Symbol, Strike Price, Expiration Date
- Premium Collected
- Number of Contracts
- DTE at open

**Closing a Position:**
- Close Date, Close Price
- Reason (early close, expiration, assignment)
- Days Held
- Profit/Loss ($)
- Profit/Loss (%)
- Annualized Return (%)

## UI Mockup

### Dashboard - Trade History Section

```
📊 Trade History
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

[📈 Total Trades]  [💰 Total P/L]   [✅ Win Rate]    [📅 Avg Days]
      42           $12,450.00         85.7%             18 days

[+ Add Trade]  [📥 Export CSV]

Filters: [Date Range ▼] [Symbol ▼] [Status ▼] [Min P/L $____]

┌──────────────────────────────────────────────────────────────────────────┐
│ Symbol │ Open Date  │ Close Date │ Strike │ Premium │ Days │ P/L    │ %  │
├────────┼────────────┼────────────┼────────┼─────────┼──────┼────────┼────┤
│ NVDA   │ 2024-10-01 │ 2024-10-15 │ $180   │ $610    │  14  │ +$305  │50% │
│ AMD    │ 2024-10-05 │ 2024-10-20 │ $250   │ $1480   │  15  │ +$740  │50% │
│ AAPL   │ 2024-10-10 │ 2024-10-25 │ $260   │ $178    │  15  │ +$89   │50% │
└────────┴────────────┴────────────┴────────┴─────────┴──────┴────────┴────┘

[Cumulative P/L Chart showing growth over time]
```

### TradingView Watchlists - Extended Theta Decay

```
Current: 7-day view
┌─────────────┐
│ 0-7 days    │
└─────────────┘

New: Full expiration view
┌──────────────────────────────────────────┐
│ 0──7──14──21──28──35──42──49──56 days   │
│ │  │   │   │Today│   │   │   │          │
│ ╲  ╲   ╲   ╲   │ ╲   ╲   ╲   ╲          │
│  ╲  ╲   ╲   ╲  │  ╲   ╲   ╲   ╲         │
│   ╲  ╲   ╲   ╲ │   ╲   ╲   ╲   ╲        │
│    ╲  ╲   ╲   ╲│    ╲   ╲   ╲   ╲       │
└────────────────▼─────────────────────────┘
            Current Day
```

## Manual Trade Entry Form

When user clicks "+ Add Trade":

```
━━━━━━━━━━━━━━━━━━━━━━━━━━
    Add New Trade
━━━━━━━━━━━━━━━━━━━━━━━━━━

Symbol:        [NVDA ▼]
Strike Price:  [$____]
Expiration:    [2024-12-20 📅]
Premium:       [$____]
Contracts:     [1    ]
Open Date:     [2024-10-28 📅]

[Advanced ▼]
  Close Date:  [________]  (optional)
  Close Price: [$____]     (optional)
  Notes:       [________]  (optional)

[Cancel]  [Save Trade]
```

## Implementation Steps (For You or Me)

### Option 1: I Implement Everything Now (2 hours)
- Create all files
- Integrate into dashboard
- Test thoroughly
- Risk: Might have bugs that need fixing

### Option 2: I Implement in Phases (Safer)
- **Phase 1 Today**: Create database + manager class (30 min)
- **Phase 2 Tomorrow**: Add to Dashboard (45 min)
- **Phase 3 Next**: Add to Watchlists + Fix Theta (45 min)
- Benefit: Test each phase before moving on

### Option 3: You Review Plan First
- Review this document and [TRADE_HISTORY_PLAN.md](TRADE_HISTORY_PLAN.md:1)
- Tell me any changes needed
- Then I implement with your feedback

## Quick Theta Decay Fix (I can do this now - 5 min)

This is the easiest change. Want me to just fix this one first?

Current code location: `dashboard.py` line ~1200-1250 (in TradingView Watchlists section)

Change:
```python
# OLD (7 days only):
days_range = list(range(0, min(8, int(dte) + 1)))

# NEW (all days to expiration):
days_range = list(range(0, int(dte) + 1))
```

This will immediately show the full theta decay curve to expiration.

## Decision Point

**What would you like me to do?**

A) Implement everything now (2 hours, might have bugs)
B) Start with Phase 1 only (30 min, safer)
C) Just fix theta decay first (5 min, immediate value)
D) Review plan more and make changes

Let me know and I'll proceed accordingly!

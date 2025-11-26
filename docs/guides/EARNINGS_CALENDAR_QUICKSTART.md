# Earnings Calendar - Quick Start Guide

## 5-Minute Setup

### Step 1: Install (30 seconds)
```bash
pip install streamlit pandas plotly robin-stocks-py python-dotenv psycopg2-binary
```

### Step 2: Configure (1 minute)
Add to `.env`:
```env
ROBINHOOD_USERNAME=your_email@example.com
ROBINHOOD_PASSWORD=your_password
```

### Step 3: Initialize (2 minutes)
```bash
python sync_earnings_demo.py
```

### Step 4: Run (30 seconds)
```bash
streamlit run pages/earnings_calendar.py
```

### Step 5: Done!
Open browser to `http://localhost:8501`

---

## Files Created

```
✓ pages/earnings_calendar.py           Main Streamlit page
✓ src/earnings_manager.py              Data manager
✓ sync_earnings_demo.py                Demo data script
✓ check_earnings_tables.py             Verification script
✓ EARNINGS_CALENDAR_README.md          Full docs
✓ INTEGRATE_EARNINGS_CALENDAR.md       Integration guide
✓ EARNINGS_CALENDAR_LAYOUT.md          Design specs
✓ EARNINGS_CALENDAR_SUMMARY.md         Implementation summary
✓ EARNINGS_CALENDAR_QUICKSTART.md      This file
```

---

## Key Commands

| Command | Purpose |
|---------|---------|
| `python sync_earnings_demo.py` | Populate sample data |
| `streamlit run pages/earnings_calendar.py` | Run standalone |
| `python check_earnings_tables.py` | Verify setup |
| `python -c "from src.earnings_manager import EarningsManager; EarningsManager()"` | Create tables only |

---

## Features at a Glance

| Feature | Status |
|---------|--------|
| 📅 Calendar View | ✅ Done |
| 📋 List View | ✅ Done |
| 📈 Historical Charts | ✅ Done |
| 📊 Analytics | ✅ Done |
| 🔄 Robinhood Sync | ✅ Done |
| 🎯 Filters | ✅ Done |
| 📥 CSV Export | ✅ Done |
| 🔔 Alerts | ⏳ Planned |

---

## Database Tables

### earnings_events
Upcoming and past earnings with estimates/actuals

### earnings_history
Historical data synced from Robinhood (8 quarters)

---

## Common Tasks

### Add Earnings Event
```python
from src.earnings_manager import EarningsManager
from datetime import datetime

em = EarningsManager()
em.add_earnings_event(
    symbol='AAPL',
    earnings_date=datetime(2025, 10, 28, 16, 0),
    earnings_time='AMC',
    eps_estimate=2.10
)
em.close()
```

### Get This Week's Earnings
```python
from datetime import date, timedelta

em = EarningsManager()
df = em.get_earnings_events(
    start_date=date.today(),
    end_date=date.today() + timedelta(days=7)
)
print(df[['symbol', 'earnings_date', 'eps_estimate']])
em.close()
```

### Sync Specific Stocks
```python
em = EarningsManager()
result = em.sync_robinhood_earnings(['AAPL', 'MSFT', 'GOOGL'])
print(f"Synced: {result['synced']}, Errors: {result['errors']}")
em.close()
```

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| Empty calendar | Run `python sync_earnings_demo.py` |
| Sync fails | Check `.env` credentials |
| Import errors | Ensure `src/` in Python path |
| No data | Adjust date filters |
| Database error | Verify PostgreSQL running |

---

## Integration Code

Add to `dashboard.py` sidebar:
```python
if st.sidebar.button("📅 Earnings", use_container_width=True):
    st.session_state.page = "Earnings"
```

Add to main routing:
```python
elif st.session_state.page == "Earnings":
    from pages.earnings_calendar import main
    main()
```

---

## Status Colors

| Color | Meaning |
|-------|---------|
| 🟢 Green | Beat estimates |
| 🔴 Red | Missed estimates |
| ⚪ Gray | Pending/Not reported |
| 🟡 Yellow | Inline with estimates |

---

## Filters Available

- **Date Range**: This Week, Next Week, This Month, Next Month, Custom
- **Time**: BMO (Before Market), AMC (After Market Close), All
- **Sector**: Technology, Healthcare, Finance, etc.

---

## Key Metrics

- **Total Events**: Count of earnings in date range
- **Pending**: Upcoming not yet reported
- **Beat Rate**: % of stocks that beat estimates
- **Avg Surprise**: Mean EPS surprise percentage
- **Beat/Miss**: Ratio of beats to misses

---

## API Endpoints

### Robinhood
```python
rh.login(user, pass)
rh.get_earnings('AAPL')  # Returns 8 quarters
rh.logout()
```

### EarningsManager
```python
em.get_earnings_events(start_date, end_date, time_filter, sector_filter)
em.get_historical_earnings(symbol, limit)
em.get_analytics(df)
em.sync_robinhood_earnings(symbols, callback)
em.add_earnings_event(symbol, date, **kwargs)
```

---

## File Locations

```
C:\Code\WheelStrategy\
├── pages\earnings_calendar.py
├── src\earnings_manager.py
├── sync_earnings_demo.py
├── check_earnings_tables.py
└── EARNINGS_CALENDAR_*.md
```

---

## Environment Variables

```env
# Database
DB_HOST=localhost
DB_PORT=5432
DB_NAME=magnus
DB_USER=postgres
DB_PASSWORD=your_password

# Robinhood
ROBINHOOD_USERNAME=your_email@example.com
ROBINHOOD_PASSWORD=your_password
```

---

## SQL Queries

### View All Earnings
```sql
SELECT * FROM earnings_events ORDER BY earnings_date;
```

### This Week's Earnings
```sql
SELECT * FROM earnings_events
WHERE earnings_date >= NOW()
AND earnings_date < NOW() + INTERVAL '7 days';
```

### Beat Rate by Symbol
```sql
SELECT
    symbol,
    COUNT(*) FILTER (WHERE eps_actual > eps_estimate) * 100.0 / COUNT(*) as beat_rate
FROM earnings_events
WHERE eps_actual IS NOT NULL AND eps_estimate IS NOT NULL
GROUP BY symbol
HAVING COUNT(*) >= 4
ORDER BY beat_rate DESC;
```

---

## Python One-Liners

### Check Setup
```bash
python -c "from src.earnings_manager import EarningsManager; em = EarningsManager(); print(f'Tables ready!'); em.close()"
```

### Quick Sync
```bash
python -c "from src.earnings_manager import EarningsManager; em = EarningsManager(); r = em.sync_robinhood_earnings(['AAPL']); print(r); em.close()"
```

### Export to CSV
```bash
python -c "from src.earnings_manager import EarningsManager; import pandas as pd; em = EarningsManager(); df = em.get_earnings_events(); df.to_csv('earnings.csv'); em.close()"
```

---

## Streamlit Tips

### Cache Data
```python
@st.cache_data(ttl=300)
def load_earnings():
    return manager.get_earnings_events()
```

### Progress Bar
```python
progress = st.progress(0)
for i in range(100):
    progress.progress(i + 1)
```

### Columns
```python
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Label", "Value")
```

---

## Performance Tips

1. **Index Usage**: Already created on `symbol` and `earnings_date`
2. **Batch Sync**: Limit to 50 stocks at a time
3. **Cache Results**: Use `@st.cache_data` for expensive queries
4. **Pagination**: Show 50 rows, load more on scroll (future)

---

## Trading Strategies

### Pre-Earnings
- Check calendar for upcoming earnings
- Close risky positions 2 days before
- Look for high IV to sell options

### Post-Earnings
- Identify IV crush opportunities
- Sell covered calls on holdings
- Look for beaten-down stocks (puts)

### Pattern Trading
- Focus on consistent beaters (80%+ beat rate)
- Avoid chronic missers
- Track sector trends

---

## Documentation

| File | Purpose |
|------|---------|
| `EARNINGS_CALENDAR_README.md` | Full feature docs |
| `INTEGRATE_EARNINGS_CALENDAR.md` | Setup instructions |
| `EARNINGS_CALENDAR_LAYOUT.md` | Design guide |
| `EARNINGS_CALENDAR_SUMMARY.md` | Implementation details |
| `EARNINGS_CALENDAR_QUICKSTART.md` | This cheat sheet |

---

## Support

- **Check Logs**: Terminal shows errors
- **Verify DB**: `python check_earnings_tables.py`
- **Test API**: Test Robinhood login separately
- **Review Docs**: Read full README for details

---

## Version Info

- **Version**: 1.0.0
- **Date**: October 2025
- **Python**: 3.8+
- **Streamlit**: 1.28+
- **PostgreSQL**: 12+

---

## Next Steps

1. ✅ Run `sync_earnings_demo.py`
2. ✅ Test page: `streamlit run pages/earnings_calendar.py`
3. ✅ Integrate into dashboard
4. ✅ Sync your stocks
5. ✅ Start trading!

---

## Quick Reference URLs

When running:
- **Dashboard**: http://localhost:8501
- **Streamlit Docs**: https://docs.streamlit.io
- **Robinhood Docs**: https://robin-stocks.readthedocs.io

---

That's it! You're ready to track earnings and improve your trading strategy. 🚀
